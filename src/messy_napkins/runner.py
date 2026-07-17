from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark
from .config import BenchmarkConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run messy-napkins benchmarks.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to benchmark configuration JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = BenchmarkConfig.from_file(args.config)
    results = run_benchmark(config)
    print(f"Completed {len(results)} benchmark case(s). Results: {config.output.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
