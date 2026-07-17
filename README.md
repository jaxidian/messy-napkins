# messy-napkins

A multi-dimensional benchmarking framework for local LLMs. Log, score, and compare model configurations across AI slop generation, speed, hardware efficiency, and task-specific quality (Code Gen, Docs, Design) to find the ideal local setup.

## Project Structure

```text
messy-napkins/
├── configs/
│   └── example-benchmark.json
├── logs/
│   └── .gitkeep
├── src/
│   └── messy_napkins/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── config.py
│       └── runner.py
├── tests/
│   └── test_benchmark.py
├── pyproject.toml
├── LICENSE
└── README.md
```

## What This Boilerplate Includes

- **Modular benchmark runner** for multiple cases (`code_gen`, `docs`, `system_prompt_following`, etc.).
- **Repeated trials** per case with automatic aggregation (pass rate, mean/stddev of quality scores and TPS).
- **Two runner modes**: subprocess (for any local CLI wrapper) and HTTP (OpenAI-compatible `/v1/chat/completions` — parameters are the actual request payload, not just metadata).
- **Rich config schema**: engine name/version/accelerator (separate from hardware accelerator), hardware identity (GPU, VRAM, CPU, OS), quantization, parameter count, seed, max tokens.
- **Telemetry capture**: TTFT, total duration, `tokens_per_second` (total time), and `decode_tokens_per_second` (decode phase only, excluding prefill).
- **Multi-dimensional quality scoring**: `quality_scores` is a `dict[str, float]` so `aislop` (or any rubric) can return sub-scores (correctness, style, hallucination, …) alongside a single total.
- **Structured result logging** to JSON Lines (`.jsonl`) — one row per trial plus one aggregate row per case when `trials > 1`.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync
```

For local development without installing:

```bash
export PYTHONPATH=src
```

## Usage

Run the benchmark CLI with the example config:

```bash
uv run messy-napkins --config configs/example-benchmark.json
```

Or without uv:

```bash
PYTHONPATH=src python -m messy_napkins.runner --config configs/example-benchmark.json
```

This will:
1. Execute each configured benchmark case prompt through the configured model runner, `trials` times each.
2. Measure latency values used for **TTFT** and **TPS** (both total and decode-only).
3. Send generated output to the configured `aislop` command for quality scoring.
4. Append each trial result as one JSON object per line in `logs/benchmark-results.jsonl`, followed by an aggregate summary row when `trials > 1`.

> The commands in `configs/example-benchmark.json` are intentionally stubbed so
> the scaffold runs anywhere. Replace them with your real local model inference
> command and actual `aislop` invocation for production benchmarking.

## Configuration Notes

### Model

| Field | Description |
|---|---|
| `engine.name` | Serving software (e.g., `"ollama"`, `"lemonade"`, `"lm-studio"`) |
| `engine.version` | Engine version string |
| `engine.accelerator` | Hardware accelerator (e.g., `"rocm"`, `"vulkan"`, `"cuda"`, `"metal"`) |
| `hardware.gpu` | GPU model name |
| `hardware.vram_gb` | VRAM in GB |
| `hardware.cpu` | CPU model name |
| `hardware.os` | OS description |
| `quantization` | Quant format (e.g., `"Q4_K_M"`, `"Q8_0"`, `"F16"`) |
| `parameter_count` | Model size (e.g., `"7B"`, `"70B"`) |
| `seed` | RNG seed for reproducibility (`null` = not set) |
| `max_tokens` | Generation cap (`null` = backend default) |
| `parameters` | Remaining sampler params forwarded to HTTP runner (`temperature`, `top_p`, …) |

### Runner

| Field | Description |
|---|---|
| `type` | `"subprocess"` (default) or `"http"` |
| `command` | Command array for subprocess runner — prompt appended as final argument |
| `url` | OpenAI-compatible endpoint URL for HTTP runner (e.g., `"http://localhost:11434/v1/chat/completions"`) |
| `timeout_seconds` | Per-prompt timeout |

> **Config fidelity note:** with `type: "subprocess"`, `model.parameters` are
> logged as metadata only — they are not automatically forwarded to the
> subprocess command.  With `type: "http"`, `model.parameters`, `seed`, and
> `max_tokens` are sent as the literal request payload, so the logged config is
> provably what executed.

### Cases

| Field | Description |
|---|---|
| `trials` | Number of repeated runs per case (default `1`); set to `≥ 3` to measure reliability |

### Output

- `output.path`: JSONL file destination for benchmark records.

## Tests

Run focused unit tests:

```bash
uv run python -m unittest -v tests/test_benchmark.py
```

Or without uv:

```bash
PYTHONPATH=src python -m unittest -v tests/test_benchmark.py
```

