# messy-napkins

A multi-dimensional benchmarking framework for local LLMs. Log, score, and compare model configurations across reliability, speed, hardware efficiency, and task-specific quality (Code Gen, Docs, Design) to find the ideal local setup.

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

## What This Framework Includes

- **Two runner modes** — subprocess (any local CLI wrapper) and HTTP (OpenAI-compatible `/v1/chat/completions` with SSE streaming for real TTFT metrics).
- **Repeated trials** per case with configurable warmup, and full aggregation: mean, stddev, median, p95, min, max per metric.
- **Separate reliability definitions** — `execution_success_rate` (no crash/timeout) and `task_pass_rate` (explicit pass criteria satisfied) are tracked independently.
- **Persisted failure rows** — a timeout or error on one trial produces a failure row (with `error_type`, `error_message`, `timed_out`, `stage`) rather than aborting the run; the benchmark always completes.
- **Real TTFT from SSE streaming** — the HTTP runner sends `stream: true` and records first-token arrival so `ttft_seconds` is the genuine time-to-first-token, not a full round-trip proxy.
- **Effective config capture** — every trial row includes `effective_command` (subprocess) or `effective_request` (HTTP) so the logged config is provably what executed.
- **Source-qualified token counts** — `output_tokens`/`prompt_tokens` record whether counts came from `"api"` (exact, via `usage.completion_tokens`), `"estimated"` (whitespace proxy), or `"unavailable"` (empty output). Empty output sets TPS to `null` rather than producing a misleading positive value.
- **VRAM telemetry with baseline** — `vram_baseline_mb`, `vram_peak_mb`, `vram_peak_delta_mb`, `vram_avg_delta_mb` sampled from `nvidia-smi`/`rocm-smi`; a pre-inference baseline is captured so deltas attribute only inference memory, not pre-existing GPU load.
- **Structured evaluation payload** — the aislop evaluator receives a full JSON payload (case ID, prompt, system prompt, generated output, expected answer, pass condition) so it can make context-aware judgments.
- **Deterministic pass conditions** — cases can declare `pass_condition: "exact_match"`, `"contains"`, or `"score_threshold"` to produce a boolean `task_passed` without LLM-as-judge.
- **Run manifest** — every JSONL output begins with a `row_type: "run_manifest"` row capturing run ID, config hash, git commit, Python version, platform, and model/engine/hardware identity so results are self-describing.
- **Rich config schema** — engine name/version/accelerator (separate from hardware), hardware identity, quantization, parameter count, seed, max_tokens, system prompt, expected answer, warmup trials.
- **Multi-dimensional quality scores** — `quality_scores` is `dict[str, float]` so evaluators can return sub-scores (correctness, style, hallucination, …).

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
1. Write a `run_manifest` row to the JSONL output with full provenance.
2. For each case: run `warmup_trials` (excluded from aggregates), then `trials` recorded runs.
3. Persist every trial row — including failures — to JSONL.
4. Measure real TTFT (HTTP streaming) or first-output timestamp (subprocess), total duration, `tokens_per_second` (total time), and `decode_tokens_per_second` (decode phase, excluding prefill).
5. Send a structured JSON evaluation payload to the configured `aislop` command for quality scoring.
6. Write an aggregate row per case when `trials > 1`.

> **Important:** the runner and aislop commands in `configs/example-benchmark.json` are stubs. Replace them with your real local model inference command and actual `aislop` invocation.

## Configuration Reference

### Model

| Field | Description |
|---|---|
| `engine.name` | Serving software (e.g., `"ollama"`, `"lemonade"`, `"lm-studio"`) |
| `engine.version` | Engine version string |
| `engine.accelerator` | Hardware accelerator (e.g., `"rocm"`, `"vulkan"`, `"cuda"`, `"metal"`) |
| `hardware.gpu` | GPU model name |
| `hardware.vram_gb` | Declared VRAM in GB |
| `hardware.cpu` | CPU model name |
| `hardware.os` | OS description |
| `quantization` | Quant format (e.g., `"Q4_K_M"`, `"Q8_0"`, `"F16"`) |
| `parameter_count` | Model size (e.g., `"7B"`, `"70B"`) |
| `seed` | RNG seed for reproducibility (`null` = not set) |
| `max_tokens` | Generation cap (`null` = backend default) |
| `parameters` | Sampler params forwarded to HTTP runner (`temperature`, `top_p`, …) |

### Runner

| Field | Description |
|---|---|
| `type` | `"subprocess"` (default) or `"http"` |
| `command` | Command array for subprocess runner — prompt appended as final argument |
| `url` | OpenAI-compatible endpoint URL for HTTP runner (e.g., `"http://localhost:11434/v1/chat/completions"`) |
| `timeout_seconds` | Per-prompt timeout |

> **Config fidelity:** with `type: "subprocess"`, `model.parameters` are logged as metadata only — they are not automatically forwarded to the subprocess command. With `type: "http"`, `model.parameters`, `seed`, and `max_tokens` are sent as the literal request payload and recorded in `effective_request`, so the logged config is provably what executed.

> **TTFT:** with `type: "http"`, the runner sends `stream: true` and records `ttft_seconds` from the first content token arrival. With `type: "subprocess"`, TTFT is the first non-whitespace byte from stdout.

### Cases

| Field | Description |
|---|---|
| `id` | Unique case identifier |
| `task` | Task category (e.g., `"code_gen"`, `"docs"`) |
| `prompt` | Prompt sent to the model |
| `system_prompt` | Optional system prompt forwarded to HTTP runners |
| `trials` | Number of recorded runs (default `1`); set ≥ 3 for reliability statistics |
| `warmup_trials` | Runs before recorded trials, excluded from aggregates (default `0`) |
| `expected_answer` | Reference answer for deterministic pass checks |
| `pass_condition` | `"exact_match"` \| `"contains"` \| `"score_threshold"` \| `null` |
| `pass_threshold` | Score gate for `"score_threshold"` (default `0.5`) |

### Evaluator (aislop)

The `aislop` command receives a structured **JSON payload** on stdin:

```json
{
  "case_id": "code-gen-hello-world",
  "task": "code_gen",
  "prompt": "Write a Python hello world function.",
  "system_prompt": null,
  "generated_output": "def hello():\n    print('Hello, world!')",
  "expected_answer": null,
  "pass_condition": null,
  "pass_threshold": null
}
```

It should write to stdout one of:
- A JSON object with named scores: `{"correctness": 0.9, "style": 0.8}` — multi-dimensional
- A JSON object `{"score": 0.85}` — normalised to `{"total": 0.85}`
- A bare float: `0.85` — normalised to `{"total": 0.85}`
- Optionally include `"task_passed": true/false` to report a boolean pass result alongside scores

### Output schema

Each trial row includes (among other fields):

| Field | Description |
|---|---|
| `schema_version` | JSONL schema version (currently `2`) |
| `row_type` | `"run_manifest"` \| `"trial"` \| `"warmup"` \| `"aggregate"` |
| `run_id` | UUID linking all rows from the same benchmark run |
| `error` | `true` when the trial failed (inference or evaluation stage) |
| `error_type` / `error_message` / `timed_out` / `stage` | Failure details when `error: true` |
| `effective_command` | Actual subprocess command executed (subprocess runner) |
| `effective_request` | Actual HTTP payload sent, including all model parameters (HTTP runner) |
| `output_tokens` / `output_token_source` | Token count and source: `"api"` \| `"estimated"` \| `"unavailable"` |
| `prompt_tokens` / `prompt_token_source` | Input token count and source |
| `ttft_seconds` | Time to first token (first SSE content event for HTTP; first stdout byte for subprocess) |
| `total_seconds` | Full generation time |
| `tokens_per_second` | Throughput using total time; `null` when token count unavailable |
| `decode_tokens_per_second` | Decode-phase throughput (post-first-token time); `null` when indistinguishable |
| `quality_scores` | Dict of named scores from the evaluator |
| `task_passed` | `true`/`false` from deterministic pass condition or evaluator; `null` if not determined |
| `vram_baseline_mb` | VRAM before inference started (nvidia-smi/rocm-smi); `null` if unavailable |
| `vram_peak_mb` | Peak VRAM during inference |
| `vram_peak_delta_mb` | `vram_peak_mb - vram_baseline_mb` (memory attributed to the inference call) |
| `vram_avg_delta_mb` | Average VRAM delta during inference |

Aggregate rows (`row_type: "aggregate"`) add: `execution_success_rate`, `task_pass_rate`, and per-metric statistics (`mean`, `stddev`, `median`, `p95`, `min`, `max`).

## Tests

Run the full test suite (32 tests):

```bash
uv run python -m unittest -v tests/test_benchmark.py
```

Or without uv:

```bash
PYTHONPATH=src python -m unittest -v tests/test_benchmark.py
```
