"""messy-napkins benchmarking package."""

from .benchmark import run_benchmark
from .config import BenchmarkConfig

__all__ = ["BenchmarkConfig", "run_benchmark"]
