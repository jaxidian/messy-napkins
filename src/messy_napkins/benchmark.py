from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Callable

from .config import BenchmarkCase, BenchmarkConfig

MIN_DURATION_SECONDS = 0.0001
CHUNK_SIZE_BYTES = 4096
STREAM_JOIN_TIMEOUT_SECONDS = 1.0
VRAM_SAMPLE_INTERVAL_SECONDS = 0.2
SCHEMA_VERSION = 2  # bumped when the JSONL schema has breaking changes


# ---------------------------------------------------------------------------
# VRAM sampling
# ---------------------------------------------------------------------------

def _sample_vram_nvidia_smi(nvidia_smi_path: str) -> float | None:
    try:
        result = subprocess.run(
            [nvidia_smi_path, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    values = [float(line) for line in result.stdout.strip().splitlines() if line.strip()]
    return max(values) if values else None


def _sample_vram_rocm_smi(rocm_smi_path: str) -> float | None:
    try:
        result = subprocess.run(
            [rocm_smi_path, "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    used_bytes = [
        float(gpu["VRAM Total Used Memory (B)"])
        for gpu in data.values()
        if isinstance(gpu, dict) and "VRAM Total Used Memory (B)" in gpu
    ]
    return max(used_bytes) / (1024 * 1024) if used_bytes else None


def detect_vram_sample_fn() -> tuple[Callable[[], float | None] | None, str]:
    """Return a (callable, source_label) pair for sampling VRAM in MB.

    ``source_label`` is one of ``"nvidia-smi"``, ``"rocm-smi"``, or
    ``"unavailable"``.  The callable returns MB used or ``None`` on any error.
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path:
        return lambda: _sample_vram_nvidia_smi(nvidia_smi_path), "nvidia-smi"

    rocm_smi_path = shutil.which("rocm-smi")
    if rocm_smi_path:
        return lambda: _sample_vram_rocm_smi(rocm_smi_path), "rocm-smi"

    return None, "unavailable"


class VramSampler:
    """Polls VRAM usage on a background thread while a prompt is in flight.

    Captures a baseline reading in ``start()`` before polling begins so that
    ``stop()`` can report deltas (``vram_peak_delta_mb``, ``vram_avg_delta_mb``)
    that attribute only the memory consumed by the inference call, excluding
    other processes already using the GPU.

    ``stop()`` returns a dict ready to spread into a result row.  All values are
    ``None`` when no supported GPU tool is found on PATH.  Pass a custom
    ``sample_fn`` (returning MB used, or None) to override auto-detection for
    tests; supply ``source`` to override the source label in that case.
    """

    def __init__(
        self,
        sample_fn: Callable[[], float | None] | None = None,
        interval_seconds: float = VRAM_SAMPLE_INTERVAL_SECONDS,
        source: str | None = None,
    ) -> None:
        if sample_fn is not None:
            # Caller-supplied function (e.g. in tests); source defaults to "test"
            self._sample_fn: Callable[[], float | None] | None = sample_fn
            self._source = source or "test"
        else:
            self._sample_fn, self._source = detect_vram_sample_fn()
        self._interval_seconds = interval_seconds
        self._baseline: float | None = None
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return self._sample_fn is not None

    def start(self) -> None:
        if not self.available:
            return
        assert self._sample_fn is not None
        # Capture a baseline before polling starts so we can compute deltas.
        self._baseline = self._sample_fn()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._sample_fn is not None
        while not self._stop_event.is_set():
            value = self._sample_fn()
            if value is not None:
                self._samples.append(value)
            self._stop_event.wait(self._interval_seconds)

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return a VRAM telemetry dict."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2)

        null_result: dict[str, Any] = {
            "vram_source": self._source,
            "vram_baseline_mb": None,
            "vram_peak_mb": None,
            "vram_peak_delta_mb": None,
            "vram_avg_delta_mb": None,
        }

        if not self._samples:
            return null_result

        peak = max(self._samples)
        avg = statistics.mean(self._samples)
        peak_delta = (peak - self._baseline) if self._baseline is not None else None
        avg_delta = (avg - self._baseline) if self._baseline is not None else None

        return {
            "vram_source": self._source,
            "vram_baseline_mb": self._baseline,
            "vram_peak_mb": peak,
            "vram_peak_delta_mb": peak_delta,
            "vram_avg_delta_mb": avg_delta,
        }


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def estimate_token_count(text: str) -> int:
    """Estimate token count via whitespace chunks.

    Returns 0 for empty or whitespace-only input so that downstream code can
    recognise "no output" and avoid reporting misleading positive TPS values.
    This is a cross-model approximation only; rows record the source via
    ``output_token_source`` (``"api"``, ``"estimated"``, or ``"unavailable"``).
    """
    return len(re.findall(r"\S+", text))


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------

def run_prompt(command: list[str], prompt: str, timeout_seconds: int) -> tuple[str, float, float]:
    """Run a prompt via a subprocess command and return (output, ttft, total_seconds).

    Uses thread-per-stream blocking readers so it works identically on
    Windows/macOS/Linux (no ``selectors``/``os.set_blocking`` dependency).
    """
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
    system_prompt: str | None = None,
) -> tuple[str, float, float, int | None, int | None]:
    """Run a prompt via an OpenAI-compatible /v1/chat/completions endpoint (non-streaming).

    Unlike the subprocess runner, all parameters in this payload are provably
    the exact configuration that executed.

    Note: uses non-streaming mode, so ``ttft_seconds`` equals ``total_seconds``
    (first-token latency is indistinguishable from full round-trip).  Prefer
    ``run_prompt_http_streaming`` when real TTFT metrics are required.

    Returns ``(output, ttft_seconds, total_seconds, completion_tokens, prompt_tokens)``.
    The last two values are ``None`` if the backend didn't report ``usage``.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
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
    usage = result.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    api_completion_tokens = int(completion_tokens) if completion_tokens is not None else None
    prompt_tokens = usage.get("prompt_tokens")
    api_prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None

    total_seconds = max(MIN_DURATION_SECONDS, end - start)
    # Non-streaming: first-token latency is indistinguishable from total time.
    ttft_seconds = total_seconds
    return generated_output, ttft_seconds, total_seconds, api_completion_tokens, api_prompt_tokens


def run_prompt_http_streaming(
    url: str,
    model_id: str,
    prompt: str,
    parameters: dict[str, Any],
    seed: int | None,
    max_tokens: int | None,
    timeout_seconds: int,
    system_prompt: str | None = None,
) -> tuple[str, float, float, int | None, int | None]:
    """Run a prompt via an OpenAI-compatible /v1/chat/completions endpoint with SSE streaming.

    Streams the response token by token so that ``ttft_seconds`` reflects the
    arrival of the *first content token* rather than the full round-trip time.
    This gives a valid TTFT figure for Lemonade, Ollama, LM Studio, and other
    OpenAI-compatible backends.

    Sends ``stream_options: {include_usage: true}`` to request token counts in
    the final SSE chunk; falls back gracefully if the backend doesn't support it.

    Returns ``(output, ttft_seconds, total_seconds, completion_tokens, prompt_tokens)``.
    The last two values are ``None`` if the backend didn't report usage.
    """
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
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
    output_parts: list[str] = []
    first_token_at: float | None = None
    api_completion_tokens: int | None = None
    api_prompt_tokens: int | None = None

    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        while True:
            raw_line = response.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            # Content delta
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output_parts.append(content)

            # Usage (present in final chunk when stream_options.include_usage=true)
            usage = chunk.get("usage")
            if usage:
                ct = usage.get("completion_tokens")
                pt = usage.get("prompt_tokens")
                if ct is not None:
                    api_completion_tokens = int(ct)
                if pt is not None:
                    api_prompt_tokens = int(pt)

    end = time.perf_counter()
    total_seconds = max(MIN_DURATION_SECONDS, end - start)
    ttft_seconds = (
        max(MIN_DURATION_SECONDS, first_token_at - start)
        if first_token_at is not None
        else total_seconds  # no content received; fall back to total round-trip
    )
    output = "".join(output_parts).strip()
    return output, ttft_seconds, total_seconds, api_completion_tokens, api_prompt_tokens


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_with_aislop(
    command: list[str],
    evaluation_payload: dict[str, Any],
    timeout_seconds: int,
) -> tuple[dict[str, float], bool | None, str]:
    """Score a generation via the configured aislop command.

    Sends a structured JSON evaluation payload via stdin so the evaluator has
    full task context (case ID, prompt, system prompt, generated output,
    expected answer, pass condition).  This replaces the previous raw-text
    stdin protocol.

    Payload keys: ``case_id``, ``task``, ``prompt``, ``system_prompt``,
    ``generated_output``, ``expected_answer``, ``pass_condition``,
    ``pass_threshold``.

    Returns ``(quality_scores, task_passed, raw_evaluator_output)``.

    ``quality_scores`` is a ``dict[str, float]`` supporting multi-dimensional
    scoring (e.g. ``{"correctness": 0.8, "style": 0.9}``).  Legacy single-score
    output — either ``{"score": <n>}`` JSON or a bare float — is normalised to
    ``{"total": <n>}``.

    ``task_passed`` is the boolean from the evaluator's ``task_passed`` key if
    present, otherwise ``None`` (caller computes it from ``pass_condition``).
    """
    input_json = json.dumps(evaluation_payload)
    completed = subprocess.run(
        command,
        input=input_json,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"aislop evaluation failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )

    raw_output = completed.stdout.strip()
    evaluator_task_passed: bool | None = None

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, dict):
            if "task_passed" in parsed:
                evaluator_task_passed = bool(parsed["task_passed"])
            scores = {k: float(v) for k, v in parsed.items() if k != "task_passed"}
            if set(scores.keys()) == {"score"}:
                return {"total": scores["score"]}, evaluator_task_passed, raw_output
            return scores, evaluator_task_passed, raw_output
    except json.JSONDecodeError:
        pass

    return {"total": float(raw_output)}, None, raw_output


def _compute_task_passed(
    case: BenchmarkCase,
    generated_output: str,
    quality_scores: dict[str, float],
    evaluator_task_passed: bool | None,
) -> bool | None:
    """Derive ``task_passed`` from pass_condition or evaluator result.

    Priority:
    1. Deterministic pass conditions (``exact_match``, ``contains``,
       ``score_threshold``) are computed directly from the output and scores.
    2. If no deterministic condition, use the evaluator's own ``task_passed``
       value if it provided one.
    3. Otherwise ``None`` (no pass judgement available).
    """
    if case.pass_condition == "exact_match":
        if case.expected_answer is not None:
            return generated_output.strip() == case.expected_answer.strip()
        return None

    if case.pass_condition == "contains":
        if case.expected_answer is not None:
            return case.expected_answer in generated_output
        return None

    if case.pass_condition == "score_threshold":
        threshold = case.pass_threshold if case.pass_threshold is not None else 0.5
        total_score = quality_scores.get("total")
        if total_score is not None:
            return total_score >= threshold
        return None

    # No deterministic condition; use the evaluator's own judgment if provided.
    return evaluator_task_passed


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_trials(
    case_id: str, task: str, trial_results: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate N trial result rows for a single case into reliability statistics.

    Distinguishes execution success (no error) from task pass (explicit pass
    criteria satisfied).  Reports median, p95, min, max alongside mean/stddev
    for latency and throughput so a small number of outliers don't dominate.
    """

    def _stats(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "stddev": None, "median": None, "p95": None, "min": None, "max": None}
        mean = statistics.mean(values)
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        median = statistics.median(values)
        p95 = statistics.quantiles(values, n=100)[94] if len(values) >= 2 else values[0]
        return {
            "mean": mean,
            "stddev": stddev,
            "median": median,
            "p95": p95,
            "min": min(values),
            "max": max(values),
        }

    successful = [r for r in trial_results if not r.get("error")]
    trial_count = len(trial_results)
    execution_success_count = len(successful)

    # task_pass_rate counts trials with an explicit boolean task_passed result.
    task_judged = [r for r in trial_results if r.get("task_passed") is not None]
    task_passed_count = sum(1 for r in task_judged if r.get("task_passed") is True)
    task_pass_rate = task_passed_count / len(task_judged) if task_judged else None

    agg: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "aggregate",
        "run_id": trial_results[0].get("run_id") if trial_results else None,
        "benchmark_name": trial_results[0]["benchmark_name"] if trial_results else "",
        "case_id": case_id,
        "task": task,
        "trial_count": trial_count,
        "execution_success_count": execution_success_count,
        "execution_success_rate": execution_success_count / trial_count if trial_count else 0.0,
        "task_passed_count": task_passed_count,
        "task_pass_rate": task_pass_rate,
    }

    ttft_values = [r["ttft_seconds"] for r in successful if r.get("ttft_seconds") is not None]
    tps_values = [r["tokens_per_second"] for r in successful if r.get("tokens_per_second") is not None]
    decode_tps_values = [r["decode_tokens_per_second"] for r in successful if r.get("decode_tokens_per_second") is not None]

    ttft_stats = _stats(ttft_values)
    agg["mean_ttft_seconds"] = ttft_stats["mean"]
    agg["stddev_ttft_seconds"] = ttft_stats["stddev"]
    agg["median_ttft_seconds"] = ttft_stats["median"]
    agg["p95_ttft_seconds"] = ttft_stats["p95"]
    agg["min_ttft_seconds"] = ttft_stats["min"]
    agg["max_ttft_seconds"] = ttft_stats["max"]

    tps_stats = _stats(tps_values)
    agg["mean_tokens_per_second"] = tps_stats["mean"]
    agg["stddev_tokens_per_second"] = tps_stats["stddev"]
    agg["median_tokens_per_second"] = tps_stats["median"]
    agg["p95_tokens_per_second"] = tps_stats["p95"]
    agg["min_tokens_per_second"] = tps_stats["min"]
    agg["max_tokens_per_second"] = tps_stats["max"]

    decode_stats = _stats(decode_tps_values)
    agg["mean_decode_tokens_per_second"] = decode_stats["mean"]
    agg["stddev_decode_tokens_per_second"] = decode_stats["stddev"]
    agg["median_decode_tokens_per_second"] = decode_stats["median"]
    agg["p95_decode_tokens_per_second"] = decode_stats["p95"]

    # Per-key quality score aggregates
    all_score_keys: set[str] = set()
    for r in successful:
        all_score_keys.update((r.get("quality_scores") or {}).keys())

    quality_agg: dict[str, Any] = {}
    for key in sorted(all_score_keys):
        score_values = [
            r["quality_scores"][key]
            for r in successful
            if key in (r.get("quality_scores") or {})
        ]
        quality_agg[key] = _stats(score_values)
    agg["quality_scores_aggregate"] = quality_agg

    return agg


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------

def _config_hash(config: BenchmarkConfig) -> str:
    config_dict = dataclasses.asdict(config)
    config_json = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_json.encode()).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def build_run_manifest(config: BenchmarkConfig, run_id: str) -> dict[str, Any]:
    """Build a run manifest row capturing all provenance for this benchmark run.

    Written as the first row of the JSONL output so consumers can always find
    the configuration and environment context for any trial in the same file.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "row_type": "run_manifest",
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": config.name,
        "config_hash": _config_hash(config),
        "git_commit": _git_commit(),
        # Model identity
        "model_id": config.model.id,
        "model_quantization": config.model.quantization,
        "model_parameter_count": config.model.parameter_count,
        "model_context_size": config.model.context_size,
        "model_seed": config.model.seed,
        "model_max_tokens": config.model.max_tokens,
        # Engine and hardware
        "engine_name": config.model.engine.name,
        "engine_version": config.model.engine.version,
        "engine_accelerator": config.model.engine.accelerator,
        "hardware_gpu": config.model.hardware.gpu,
        "hardware_vram_gb": config.model.hardware.vram_gb,
        "hardware_cpu": config.model.hardware.cpu,
        "hardware_os": config.model.hardware.os,
        # Runner configuration
        "runner_type": config.runner.type,
        # Benchmark structure
        "total_cases": len(config.cases),
        "case_ids": [c.id for c in config.cases],
        # Environment
        "python_version": platform.python_version(),
        "platform_info": platform.platform(),
        "cpu_count": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# Core benchmark execution
# ---------------------------------------------------------------------------

def benchmark_case(
    config: BenchmarkConfig,
    case: BenchmarkCase,
    trial_index: int = 0,
    run_id: str | None = None,
    is_warmup: bool = False,
) -> dict[str, Any]:
    """Execute one trial for a benchmark case.

    Always returns a row dict; errors are recorded in-band (``error: True``)
    with ``error_type``, ``error_message``, ``timed_out``, and ``stage`` so
    downstream aggregation can distinguish inference failures from evaluation
    failures without losing the trial record.
    """
    row_type = "warmup" if is_warmup else "trial"
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "row_type": row_type,
        "run_id": run_id,
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
        "context_size": config.model.context_size,
        "seed": config.model.seed,
        "max_tokens": config.model.max_tokens,
        "model_parameters": config.model.parameters,
        "prompt": case.prompt,
        "system_prompt": case.system_prompt,
    }

    # Stage 1: inference
    vram_sampler = VramSampler()
    vram_sampler.start()
    generated_output: str = ""
    ttft_seconds: float | None = None
    total_seconds: float | None = None
    api_completion_tokens: int | None = None
    api_prompt_tokens: int | None = None
    effective_request: dict[str, Any] | None = None
    effective_command: list[str] | None = None
    inference_error: Exception | None = None

    try:
        if config.runner.type == "http":
            messages: list[dict[str, str]] = []
            if case.system_prompt:
                messages.append({"role": "system", "content": case.system_prompt})
            messages.append({"role": "user", "content": case.prompt})
            effective_request = {
                "model": config.model.id,
                "messages": messages,
                "stream": True,
                **config.model.parameters,
            }
            if config.model.seed is not None:
                effective_request["seed"] = config.model.seed
            if config.model.max_tokens is not None:
                effective_request["max_tokens"] = config.model.max_tokens

            (
                generated_output,
                ttft_seconds,
                total_seconds,
                api_completion_tokens,
                api_prompt_tokens,
            ) = run_prompt_http_streaming(
                url=config.runner.url,
                model_id=config.model.id,
                prompt=case.prompt,
                parameters=config.model.parameters,
                seed=config.model.seed,
                max_tokens=config.model.max_tokens,
                timeout_seconds=config.runner.timeout_seconds,
                system_prompt=case.system_prompt,
            )
        else:
            effective_command = [*config.runner.command, case.prompt]
            generated_output, ttft_seconds, total_seconds = run_prompt(
                command=config.runner.command,
                prompt=case.prompt,
                timeout_seconds=config.runner.timeout_seconds,
            )
    except Exception as exc:
        inference_error = exc
    finally:
        vram_telemetry = vram_sampler.stop()

    if inference_error is not None:
        exc = inference_error
        return {
            **base,
            "error": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "timed_out": "timed out" in str(exc).lower(),
            "stage": "inference",
            "generated_output": None,
            "effective_command": effective_command,
            "effective_request": effective_request,
            "output_tokens": None,
            "output_token_source": None,
            "prompt_tokens": None,
            "prompt_token_source": None,
            "ttft_seconds": None,
            "total_seconds": None,
            "tokens_per_second": None,
            "decode_tokens_per_second": None,
            "quality_scores": None,
            "task_passed": None,
            "evaluator_raw_output": None,
            **vram_telemetry,
        }

    # Token counts — prefer API-reported over whitespace estimate.
    if api_completion_tokens is not None:
        output_tokens: int = api_completion_tokens
        output_token_source = "api"
    else:
        estimated = estimate_token_count(generated_output)
        if estimated > 0:
            output_tokens = estimated
            output_token_source = "estimated"
        else:
            output_tokens = 0
            output_token_source = "unavailable"

    if api_prompt_tokens is not None:
        prompt_tokens: int | None = api_prompt_tokens
        prompt_token_source = "api"
    else:
        prompt_tokens = None
        prompt_token_source = "unavailable"

    # TPS — null when token count is unavailable to avoid misleading values.
    if output_tokens > 0 and total_seconds is not None:
        tps: float | None = output_tokens / max(total_seconds, MIN_DURATION_SECONDS)
    else:
        tps = None

    decode_time = (total_seconds - ttft_seconds) if (total_seconds is not None and ttft_seconds is not None) else None
    if output_tokens > 0 and decode_time is not None and decode_time > MIN_DURATION_SECONDS:
        decode_tps: float | None = output_tokens / decode_time
    else:
        decode_tps = None

    # Stage 2: evaluation
    evaluation_payload: dict[str, Any] = {
        "case_id": case.id,
        "task": case.task,
        "prompt": case.prompt,
        "system_prompt": case.system_prompt,
        "generated_output": generated_output,
        "expected_answer": case.expected_answer,
        "pass_condition": case.pass_condition,
        "pass_threshold": case.pass_threshold,
    }
    try:
        quality_scores, evaluator_task_passed, evaluator_raw = evaluate_with_aislop(
            command=config.aislop.command,
            evaluation_payload=evaluation_payload,
            timeout_seconds=config.aislop.timeout_seconds,
        )
    except Exception as exc:
        return {
            **base,
            "error": True,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "timed_out": "timed out" in str(exc).lower(),
            "stage": "evaluation",
            "generated_output": generated_output,
            "effective_command": effective_command,
            "effective_request": effective_request,
            "output_tokens": output_tokens,
            "output_token_source": output_token_source,
            "prompt_tokens": prompt_tokens,
            "prompt_token_source": prompt_token_source,
            "ttft_seconds": ttft_seconds,
            "total_seconds": total_seconds,
            "tokens_per_second": tps,
            "decode_tokens_per_second": decode_tps,
            "quality_scores": None,
            "task_passed": None,
            "evaluator_raw_output": None,
            **vram_telemetry,
        }

    task_passed = _compute_task_passed(
        case=case,
        generated_output=generated_output,
        quality_scores=quality_scores,
        evaluator_task_passed=evaluator_task_passed,
    )

    return {
        **base,
        "error": False,
        "generated_output": generated_output,
        "effective_command": effective_command,
        "effective_request": effective_request,
        # Token counts
        "output_tokens": output_tokens,
        "output_token_source": output_token_source,
        "prompt_tokens": prompt_tokens,
        "prompt_token_source": prompt_token_source,
        # Timing
        "ttft_seconds": ttft_seconds,
        "total_seconds": total_seconds,
        "tokens_per_second": tps,
        "decode_tokens_per_second": decode_tps,
        # Quality
        "quality_scores": quality_scores,
        "task_passed": task_passed,
        "evaluator_raw_output": evaluator_raw,
        **vram_telemetry,
    }


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    """Execute all benchmark cases and return every row written to the JSONL log.

    Row order: ``run_manifest``, then for each case: ``warmup`` rows (if any),
    ``trial`` rows, and (when ``trials > 1``) one ``aggregate`` row.
    """
    run_id = str(uuid.uuid4())
    manifest = build_run_manifest(config, run_id)
    append_jsonl(config.output.path, manifest)
    all_rows: list[dict[str, Any]] = [manifest]

    for case in config.cases:
        # Warmup trials — run but keep separately; not counted in aggregates.
        for warmup_index in range(max(0, case.warmup_trials)):
            try:
                warmup_row = benchmark_case(
                    config, case, trial_index=warmup_index, run_id=run_id, is_warmup=True
                )
            except Exception as exc:
                warmup_row = {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "warmup",
                    "run_id": run_id,
                    "case_id": case.id,
                    "error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            append_jsonl(config.output.path, warmup_row)
            all_rows.append(warmup_row)

        # Recorded trials.
        trial_results: list[dict[str, Any]] = []
        for trial_index in range(max(1, case.trials)):
            try:
                row = benchmark_case(config, case, trial_index=trial_index, run_id=run_id)
            except Exception as exc:
                # Last-resort catch for truly unexpected failures in benchmark_case itself.
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "row_type": "trial",
                    "run_id": run_id,
                    "trial_index": trial_index,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "benchmark_name": config.name,
                    "case_id": case.id,
                    "task": case.task,
                    "error": True,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "timed_out": "timed out" in str(exc).lower(),
                    "stage": "unknown",
                }
            append_jsonl(config.output.path, row)
            trial_results.append(row)
            all_rows.append(row)

        if len(trial_results) > 1:
            agg = aggregate_trials(case.id, case.task, trial_results)
            append_jsonl(config.output.path, agg)
            all_rows.append(agg)

    return all_rows
