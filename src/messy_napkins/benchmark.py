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
    return max(1, len(re.findall(r"\S+", text)))


def run_prompt(command: list[str], prompt: str, timeout_seconds: int) -> tuple[str, float, float]:
    start = time.perf_counter()
    completed = subprocess.run(
        [*command, prompt],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    end = time.perf_counter()

    if completed.returncode != 0:
        raise RuntimeError(
            f"Prompt command failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )

    total_seconds = max(0.0001, end - start)
    ttft_seconds = total_seconds
    return completed.stdout.strip(), ttft_seconds, total_seconds


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
