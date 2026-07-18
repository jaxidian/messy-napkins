from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EngineConfig:
    """Serving engine identity.

    Separates the serving software (Ollama, Lemonade, LM Studio) from the
    hardware accelerator so both can be tracked independently for comparisons.
    Previously, a flat ``backend`` string conflated these two concepts.
    """

    name: str = ""         # serving software, e.g. "ollama", "lemonade", "lm-studio"
    version: str = ""      # engine version, e.g. "0.1.34"
    accelerator: str = ""  # hardware accelerator, e.g. "rocm", "vulkan", "cuda", "metal"
    startup_flags: list[str] = field(default_factory=list)  # extra CLI flags passed to the engine

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EngineConfig":
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            accelerator=data.get("accelerator", ""),
            startup_flags=data.get("startup_flags", []),
        )


@dataclass
class HardwareConfig:
    """Host hardware identity.

    Required context for interpreting any tok/s number — the same model at the
    same quantization will perform very differently across GPU generations and
    VRAM capacities.
    """

    gpu: str = ""           # GPU model, e.g. "AMD RX 7900 XTX"
    vram_gb: float = 0.0
    cpu: str = ""           # CPU model, e.g. "AMD Ryzen 9 7950X"
    os: str = ""            # OS description, e.g. "Ubuntu 24.04"
    device_count: int = 0   # number of GPU devices used
    driver_version: str = "" # GPU driver version, e.g. "545.23.08"
    runtime_version: str = "" # CUDA/ROCm/Vulkan runtime version, e.g. "CUDA 12.3"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareConfig":
        return cls(
            gpu=data.get("gpu", ""),
            vram_gb=float(data.get("vram_gb", 0.0)),
            cpu=data.get("cpu", ""),
            os=data.get("os", ""),
            device_count=int(data.get("device_count", 0)),
            driver_version=data.get("driver_version", ""),
            runtime_version=data.get("runtime_version", ""),
        )


@dataclass
class ModelConfig:
    """Model and serving configuration.

    ``engine`` captures the serving software and hardware accelerator separately
    so both dimensions can be compared independently.
    ``hardware`` records the host system so results can be interpreted in context.
    ``quantization`` and ``parameter_count`` are needed to compare across quant
    levels (e.g. Q4_K_M vs Q8_0) and model sizes (e.g. 7B vs 13B).
    ``seed`` and ``max_tokens`` affect reproducibility and TPS comparability —
    without a token cap, generation length varies run-to-run and confounds TPS.
    ``source``, ``revision``, ``artifact_filename``, and ``artifact_checksum``
    provide immutable identification of the model artifact so results can be
    attributed to a specific file rather than just a human-readable name.
    """

    id: str
    context_size: int
    engine: EngineConfig = field(default_factory=EngineConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    quantization: str = ""      # quant format, e.g. "Q4_K_M", "Q8_0", "F16"
    parameter_count: str = ""   # model size, e.g. "7B", "13B", "70B"
    seed: int | None = None     # RNG seed for reproducibility; None = not set
    max_tokens: int | None = None  # generation cap; None = backend default
    parameters: dict[str, Any] = field(default_factory=dict)  # temperature, top_p, …
    # Immutable artifact provenance (all optional; fill in what is known)
    source: str = ""            # model source/repository, e.g. "https://huggingface.co/..."
    revision: str = ""          # model revision/commit hash
    artifact_filename: str = "" # artifact filename, e.g. "model-Q4_K_M.gguf"
    artifact_checksum: str = "" # SHA-256 checksum of the model artifact file

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        data = dict(data)
        # Backward compat: plain "backend" string maps to engine.accelerator
        if "backend" in data and "engine" not in data:
            data["engine"] = {"accelerator": data.pop("backend")}
        engine = EngineConfig.from_dict(data.get("engine", {}))
        hardware = HardwareConfig.from_dict(data.get("hardware", {}))
        return cls(
            id=data["id"],
            context_size=data["context_size"],
            engine=engine,
            hardware=hardware,
            quantization=data.get("quantization", ""),
            parameter_count=data.get("parameter_count", ""),
            seed=data.get("seed"),
            max_tokens=data.get("max_tokens"),
            parameters=data.get("parameters", {}),
            source=data.get("source", ""),
            revision=data.get("revision", ""),
            artifact_filename=data.get("artifact_filename", ""),
            artifact_checksum=data.get("artifact_checksum", ""),
        )


@dataclass
class RunnerConfig:
    """Subprocess or HTTP runner configuration.

    When ``type`` is ``"subprocess"``, ``command`` is invoked with the prompt
    appended as a positional argument.  The logged ``model.parameters`` are
    **metadata only** — they are not automatically forwarded to the subprocess;
    the user is responsible for keeping the wrapper script and the config in sync.

    When ``type`` is ``"http"``, the runner POSTs to an OpenAI-compatible
    ``/v1/chat/completions`` endpoint at ``url`` using Server-Sent Events (SSE)
    streaming so that ``ttft_seconds`` measures the arrival of the first content
    token rather than the full round-trip.  In this mode ``model.parameters``
    (temperature, top_p, …) plus ``model.seed`` and ``model.max_tokens`` are
    sent as the actual request payload, so the logged config is provably the
    config that executed.
    """

    command: list[str] = field(default_factory=list)
    timeout_seconds: int = 120
    type: str = "subprocess"  # "subprocess" | "http"
    url: str = ""             # OpenAI-compatible base URL, used when type="http"
    provider: str = "auto"     # metadata adapter: auto | lemonade | ollama | lm-studio


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
    trials: int = 1           # number of recorded runs; use >1 for reliability statistics
    warmup_trials: int = 0    # runs executed before recorded trials, excluded from aggregates
    system_prompt: str | None = None   # optional system prompt forwarded to HTTP runner
    expected_answer: str | None = None # reference answer for deterministic pass checks
    pass_condition: str | None = None  # "exact_match" | "contains" | "score_threshold" | None
    pass_threshold: float | None = None  # score gate for "score_threshold" condition
    evaluation: dict[str, Any] = field(default_factory=dict)


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
        model = ModelConfig.from_dict(data["model"])
        runner_data = data["runner"]
        runner = RunnerConfig(
            command=runner_data.get("command", []),
            timeout_seconds=runner_data.get("timeout_seconds", 120),
            type=runner_data.get("type", "subprocess"),
            url=runner_data.get("url", ""),
            provider=runner_data.get("provider", "auto"),
        )
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
