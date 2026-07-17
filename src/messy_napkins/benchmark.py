from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BenchmarkCase, BenchmarkConfig


def estimate_token_count(text: str) -> int:
    """Estimate token count via whitespace chunks.

    This is a fast approximation for cross-model comparison only and will not
    match exact tokenizer counts for subword/tokenizer-specific schemes.
    """

    return max(1, len(re.findall(r"\S+", text)))


def run_prompt(command: list[str], prompt: str, timeout_seconds: int) -> tuple[str, float, float]:
    start = time.perf_counter()
    with subprocess.Popen(
        [*command, prompt], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    ) as process:
        stdout_chunks: list[str] = []
        first_output_at: float | None = None
        while True:
            if time.perf_counter() - start > timeout_seconds:
                process.kill()
                raise RuntimeError(f"Prompt command timed out after {timeout_seconds} seconds.")

            line = process.stdout.readline() if process.stdout else ""
            if line:
                stdout_chunks.append(line)
                if first_output_at is None and line.strip():
                    first_output_at = time.perf_counter()
                continue

            if process.poll() is not None:
                break

            time.sleep(0.01)

        stderr_output = process.stderr.read() if process.stderr else ""
        return_code = process.wait()
        end = time.perf_counter()

    if return_code != 0:
        raise RuntimeError(
            f"Prompt command failed with exit code {return_code}: {stderr_output.strip()}"
        )

    total_seconds = max(0.0001, end - start)
    ttft_seconds = total_seconds if first_output_at is None else max(0.0001, first_output_at - start)
    return "".join(stdout_chunks).strip(), ttft_seconds, total_seconds


def evaluate_with_aislop(command: list[str], generated_output: str, timeout_seconds: int) -> float:
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
        if isinstance(parsed, dict) and "score" in parsed:
            return float(parsed["score"])
    except json.JSONDecodeError:
        pass

    return float(output)


def benchmark_case(config: BenchmarkConfig, case: BenchmarkCase) -> dict[str, Any]:
    generated_output, ttft_seconds, total_seconds = run_prompt(
        command=config.runner.command,
        prompt=case.prompt,
        timeout_seconds=config.runner.timeout_seconds,
    )
    quality_score = evaluate_with_aislop(
        command=config.aislop.command,
        generated_output=generated_output,
        timeout_seconds=config.aislop.timeout_seconds,
    )
    generated_tokens = estimate_token_count(generated_output)
    tps = generated_tokens / total_seconds

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": config.name,
        "case_id": case.id,
        "task": case.task,
        "backend": config.model.backend,
        "model_id": config.model.id,
        "model_parameters": config.model.parameters,
        "context_size": config.model.context_size,
        "prompt": case.prompt,
        "generated_output": generated_output,
        "quality_score": quality_score,
        "ttft_seconds": ttft_seconds,
        "total_seconds": total_seconds,
        "tokens_per_second": tps,
    }


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def run_benchmark(config: BenchmarkConfig) -> list[dict[str, Any]]:
    results = []
    for case in config.cases:
        row = benchmark_case(config, case)
        append_jsonl(config.output.path, row)
        results.append(row)
    return results
