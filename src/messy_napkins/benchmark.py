from __future__ import annotations

import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .config import BenchmarkCase, BenchmarkConfig

MIN_DURATION_SECONDS = 0.0001
CHUNK_SIZE_BYTES = 4096
STREAM_JOIN_TIMEOUT_SECONDS = 1.0


def estimate_token_count(text: str) -> int:
    """Estimate token count via whitespace chunks.

    This is a fast approximation for cross-model comparison only and will not
    match exact tokenizer counts for subword/tokenizer-specific schemes.  Output
    rows record this as ``estimated_tokens`` to make the approximation explicit.
    """

    return max(1, len(re.findall(r"\S+", text)))


def run_prompt(command: list[str], prompt: str, timeout_seconds: int) -> tuple[str, float, float]:
    start = time.perf_counter()
    with subprocess.Popen(
        [*command, prompt], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Prompt command must expose stdout and stderr pipes for telemetry.")

        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        first_output_at: float | None = None
        first_output_lock = threading.Lock()

        def read_stream(
            stream: BinaryIO,
            chunks: list[bytes],
            on_chunk: Callable[[bytes], None] | None = None,
        ) -> None:
            while True:
                chunk = stream.read(CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                chunks.append(chunk)
                if on_chunk is not None:
                    on_chunk(chunk)

        def maybe_record_first_output(chunk: bytes) -> None:
            nonlocal first_output_at
            if not chunk or chunk.isspace():
                return
            with first_output_lock:
                if first_output_at is None:
                    first_output_at = time.perf_counter()

        def join_reader_threads() -> list[str]:
            stdout_thread.join(timeout=STREAM_JOIN_TIMEOUT_SECONDS)
            stderr_thread.join(timeout=STREAM_JOIN_TIMEOUT_SECONDS)
            alive_streams = []
            if stdout_thread.is_alive():
                alive_streams.append("stdout")
            if stderr_thread.is_alive():
                alive_streams.append("stderr")
            return alive_streams

        stdout_thread = threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_chunks, maybe_record_first_output),
        )
        stderr_thread = threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_chunks),
        )
        stdout_thread.start()
        stderr_thread.start()

        timeout_exc: subprocess.TimeoutExpired | None = None
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            timeout_exc = exc
            process.kill()
            process.wait()
        finally:
            alive_streams = join_reader_threads()
        if alive_streams:
            raise RuntimeError(
                "Prompt command stream reader threads did not terminate within "
                f"{STREAM_JOIN_TIMEOUT_SECONDS} second(s) ({', '.join(alive_streams)})."
            )

        if timeout_exc is not None:
            raise RuntimeError(
                f"Prompt command timed out after {timeout_seconds} seconds."
            ) from timeout_exc

        end = time.perf_counter()

    if return_code != 0:
        stderr_output = b"".join(stderr_chunks).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Prompt command failed with exit code {return_code}: {stderr_output.strip()}"
        )

    total_seconds = max(MIN_DURATION_SECONDS, end - start)
    ttft_seconds = (
        total_seconds
        if first_output_at is None
        else max(MIN_DURATION_SECONDS, first_output_at - start)
    )
    output = b"".join(stdout_chunks).decode("utf-8", errors="replace").strip()
    return output, ttft_seconds, total_seconds


def run_prompt_http(
    url: str,
    model_id: str,
    prompt: str,
    parameters: dict[str, Any],
    seed: int | None,
    max_tokens: int | None,
    timeout_seconds: int,
) -> tuple[str, float, float]:
    """Run a prompt via an OpenAI-compatible /v1/chat/completions endpoint.

    Unlike the subprocess runner, all parameters in this payload are provably
    the exact configuration that executed — there is no external wrapper script
    that could diverge from what is logged.

    Note: uses non-streaming mode, so TTFT cannot be distinguished from total
    response time.  Both ``ttft_seconds`` and ``total_seconds`` in the returned
    tuple reflect the full round-trip; ``decode_tokens_per_second`` will equal
    ``tokens_per_second`` for rows produced by this runner.
    """
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        **parameters,
    }
    if seed is not None:
        payload["seed"] = seed
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        body = response.read()
    end = time.perf_counter()

    result = json.loads(body)
    generated_output: str = result["choices"][0]["message"]["content"]

    total_seconds = max(MIN_DURATION_SECONDS, end - start)
    # Non-streaming: first-token latency is indistinguishable from total time.
    ttft_seconds = total_seconds
    return generated_output, ttft_seconds, total_seconds


def evaluate_with_aislop(
    command: list[str], generated_output: str, timeout_seconds: int
) -> dict[str, float]:
    """Score generated output via the configured aislop command.

    Returns a dict of named scores to support multi-dimensional evaluation
    (e.g. ``{"correctness": 0.8, "style": 0.9}``).  Legacy single-score
    output — either ``{"score": <n>}`` JSON or a bare float — is normalized
    to ``{"total": <n>}``.
    """
    completed = subprocess.run(
        command,
        input=generated_output,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"aislop evaluation failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )

    output = completed.stdout.strip()
    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            if set(parsed.keys()) == {"score"}:
                # Legacy single-score shape: {"score": <n>}
                return {"total": float(parsed["score"])}
            # Multi-dimensional: {"correctness": 0.8, "style": 0.9, …}
            return {k: float(v) for k, v in parsed.items()}
    except json.JSONDecodeError:
        pass

    return {"total": float(output)}


def aggregate_trials(
    case_id: str, task: str, trial_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate N trial result rows for a single case into reliability statistics.

    Computes pass rate, mean, and stddev for timing and quality score fields.
    Only successful trials (no ``"error"`` key) contribute to mean/stddev.
    """

    def _mean_stdev(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0.0
        return mean, stdev

    successful = [r for r in trial_results if not r.get("error")]
    pass_count = len(successful)
    trial_count = len(trial_results)

    agg: dict[str, Any] = {
        "row_type": "aggregate",
        "benchmark_name": trial_results[0]["benchmark_name"] if trial_results else "",
        "case_id": case_id,
        "task": task,
        "trial_count": trial_count,
        "pass_count": pass_count,
        "pass_rate": pass_count / trial_count if trial_count else 0.0,
    }

    ttft_values = [r["ttft_seconds"] for r in successful]
    tps_values = [r["tokens_per_second"] for r in successful]
    decode_tps_candidates = [r["decode_tokens_per_second"] for r in successful]
    decode_tps_values = [v for v in decode_tps_candidates if v is not None]

    agg["mean_ttft_seconds"], agg["stddev_ttft_seconds"] = _mean_stdev(ttft_values)
    agg["mean_tokens_per_second"], agg["stddev_tokens_per_second"] = _mean_stdev(tps_values)
    agg["mean_decode_tokens_per_second"], agg["stddev_decode_tokens_per_second"] = _mean_stdev(decode_tps_values)

    # Aggregate per-key quality scores
    all_score_keys: set[str] = set()
    for r in successful:
        all_score_keys.update(r.get("quality_scores", {}).keys())

    quality_agg: dict[str, Any] = {}
    for key in sorted(all_score_keys):
        score_values = [
            r["quality_scores"][key]
            for r in successful
            if key in r.get("quality_scores", {})
        ]
        mean_s, stdev_s = _mean_stdev(score_values)
        quality_agg[key] = {"mean": mean_s, "stddev": stdev_s}
    agg["quality_scores_aggregate"] = quality_agg

    return agg


def benchmark_case(
    config: BenchmarkConfig, case: BenchmarkCase, trial_index: int = 0
) -> dict[str, Any]:
    if config.runner.type == "http":
        generated_output, ttft_seconds, total_seconds = run_prompt_http(
            url=config.runner.url,
            model_id=config.model.id,
            prompt=case.prompt,
            parameters=config.model.parameters,
            seed=config.model.seed,
            max_tokens=config.model.max_tokens,
            timeout_seconds=config.runner.timeout_seconds,
        )
    else:
        generated_output, ttft_seconds, total_seconds = run_prompt(
            command=config.runner.command,
            prompt=case.prompt,
            timeout_seconds=config.runner.timeout_seconds,
        )

    quality_scores = evaluate_with_aislop(
        command=config.aislop.command,
        generated_output=generated_output,
        timeout_seconds=config.aislop.timeout_seconds,
    )
    estimated_tokens = estimate_token_count(generated_output)
    tps = estimated_tokens / max(total_seconds, MIN_DURATION_SECONDS)

    # decode_tokens_per_second excludes prefill (TTFT) from the denominator,
    # giving a decode-phase-only throughput figure.  For HTTP non-streaming
    # runners TTFT equals total_seconds so this cannot be computed meaningfully;
    # those rows carry None to avoid misleading values.
    decode_time = total_seconds - ttft_seconds
    decode_tps: float | None = (
        estimated_tokens / max(decode_time, MIN_DURATION_SECONDS)
        if decode_time > MIN_DURATION_SECONDS
        else None
    )

    return {
        "row_type": "trial",
        "trial_index": trial_index,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": config.name,
        "case_id": case.id,
        "task": case.task,
        # Engine + hardware identity
        "engine_name": config.model.engine.name,
        "engine_version": config.model.engine.version,
        "engine_accelerator": config.model.engine.accelerator,
        "hardware_gpu": config.model.hardware.gpu,
        "hardware_vram_gb": config.model.hardware.vram_gb,
        "hardware_cpu": config.model.hardware.cpu,
        "hardware_os": config.model.hardware.os,
        # Model identity
        "model_id": config.model.id,
        "model_quantization": config.model.quantization,
        "model_parameter_count": config.model.parameter_count,
        "model_parameters": config.model.parameters,
        "context_size": config.model.context_size,
        "seed": config.model.seed,
        "max_tokens": config.model.max_tokens,
        # Prompt + output
        "prompt": case.prompt,
        "generated_output": generated_output,
        # Quality scores dict supports multi-dimensional scoring
        "quality_scores": quality_scores,
        # Timing
        "ttft_seconds": ttft_seconds,
        "total_seconds": total_seconds,
        # TPS: tokens_per_second uses total_seconds (includes prefill/TTFT).
        # decode_tokens_per_second uses decode time only; None when indistinguishable.
        "estimated_tokens": estimated_tokens,
        "tokens_per_second": tps,
        "decode_tokens_per_second": decode_tps,
    }


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for case in config.cases:
        trial_results: list[dict[str, Any]] = []
        for trial_index in range(max(1, case.trials)):
            row = benchmark_case(config, case, trial_index=trial_index)
            append_jsonl(config.output.path, row)
            trial_results.append(row)
            all_rows.append(row)
        if len(trial_results) > 1:
            agg = aggregate_trials(case.id, case.task, trial_results)
            append_jsonl(config.output.path, agg)
            all_rows.append(agg)
    return all_rows
