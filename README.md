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
├── LICENSE
└── README.md
```

## What This Boilerplate Includes

- **Modular benchmark runner** for multiple cases (`code_gen`, `docs`, `system_prompt_following`, etc.).
- **Telemetry capture** for backend, model params, context size, TTFT, total duration, and estimated TPS.
- **`aislop` integration point** via configurable CLI command that scores generated output.
- **Structured result logging** to JSON Lines (`.jsonl`) for easy comparison and downstream visualization.

## Installation

> Placeholder: add packaging/publish instructions (pip, uv, or Docker) once dependency strategy is finalized.

For local development from source:

```bash
export PYTHONPATH=src
```

## Usage

Run the benchmark CLI with the example config:

```bash
PYTHONPATH=src python -m messy_napkins.runner --config configs/example-benchmark.json
```

This will:
1. Execute each configured benchmark case prompt through the configured model runner command.
2. Measure latency values used for **TTFT** and **TPS**.
3. Send generated output to the configured `aislop` command for deterministic quality scoring.
4. Append each benchmark result as one JSON object per line in `logs/benchmark-results.jsonl`.

> The commands in `configs/example-benchmark.json` are intentionally stubbed so
> the scaffold runs anywhere. Replace them with your real local model inference
> command and actual `aislop` invocation for production benchmarking.

## Configuration Notes

- `model.backend`: hardware backend label (e.g., `rocm`, `vulkan`, `cuda`, `metal`).
- `model.context_size`: context window for the model run (e.g., `32768`, `65536`).
- `runner.command`: shell command array used to generate output from a prompt.
- `aislop.command`: shell command array used to score generated output.
- `output.path`: JSONL file destination for benchmark records.

## Tests

Run focused unit tests:

```bash
PYTHONPATH=src python -m unittest -v tests/test_benchmark.py
```
