from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    id: str
    backend: str
    context_size: int
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunnerConfig:
    command: list[str]
    timeout_seconds: int = 120


@dataclass
class AislopConfig:
    command: list[str]
    timeout_seconds: int = 60


@dataclass
class OutputConfig:
    path: str = "logs/benchmark-results.jsonl"


@dataclass
class BenchmarkCase:
    id: str
    task: str
    prompt: str


@dataclass
class BenchmarkConfig:
    name: str
    model: ModelConfig
    runner: RunnerConfig
    aislop: AislopConfig
    output: OutputConfig
    cases: list[BenchmarkCase]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        model = ModelConfig(**data["model"])
        runner = RunnerConfig(**data["runner"])
        aislop = AislopConfig(**data["aislop"])
        output = OutputConfig(**data.get("output", {}))
        cases = [BenchmarkCase(**case) for case in data["cases"]]
        return cls(
            name=data["name"],
            model=model,
            runner=runner,
            aislop=aislop,
            output=output,
            cases=cases,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "BenchmarkConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
