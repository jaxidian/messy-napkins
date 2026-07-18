# messy-napkins

A multi-dimensional benchmarking framework for local LLMs. Log, score, and compare model configurations across reliability, speed, hardware efficiency, and task-specific quality (Code Gen, Docs, Design) to find the ideal local setup.

## Project Structure

```text
messy-napkins/
├── configs/
│   ├── examples/
│   │   ├── README.md
│   │   ├── aislop_evaluator.py
│   │   └── qwen3-14b-gguf.json
│   ├── local/
│   │   ├── .gitkeep
│   │   └── SANITIZATION.md
│   └── example-benchmark.json
├── docs/
│   └── hosting/
│       ├── README.md
│       ├── lemonade.md
│       └── qwen3-14b-gguf.md
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
- **TTFT provenance** — `ttft_source` records how TTFT was measured: `"streaming_api"` (HTTP runner: genuine first-token latency from SSE event stream) or `"first_chunk_approx"` (subprocess runner: first non-whitespace stdout chunk, subject to process/stdout buffering).
- **Effective config capture** — every trial row includes `effective_command` (subprocess) or `effective_request` (HTTP) so the logged config is provably what the client executed. The `effective_request` is built once from a shared helper and sent verbatim, so it exactly matches the payload on the wire (including `stream_options`). A `sampler_settings_source` field records `"sent_not_backend_verified"` for HTTP parameters and `"unverified_metadata"` for subprocess parameters.
- **Source-qualified token counts and TPS** — `output_tokens` and TPS (`tokens_per_second`, `decode_tokens_per_second`) are only non-null when the backend reports token counts via `usage.completion_tokens` (HTTP runner). A separate `output_tokens_whitespace_approx` field stores the whitespace-chunk count as a diagnostic-only value; it is explicitly not used for throughput metrics because word counts are not model tokens and differ across tokenizer families. Empty output sets TPS to `null` rather than producing a misleading positive value.
- **VRAM telemetry with baseline** — `vram_baseline_mb`, `vram_peak_mb`, `vram_peak_delta_mb`, `vram_avg_delta_mb` sampled from `nvidia-smi`/`rocm-smi`; a pre-inference baseline is captured so deltas attribute only inference memory, not pre-existing GPU load.
- **Structured evaluation payload** — the aislop evaluator receives a full JSON payload (case ID, prompt, system prompt, generated output, expected answer, pass condition) so it can make context-aware judgments.
- **Deterministic pass conditions** — cases can declare `pass_condition: "exact_match"`, `"contains"`, or `"score_threshold"` to produce a boolean `task_passed` without LLM-as-judge.
- **Run manifest** — every JSONL output begins with a `row_type: "run_manifest"` row capturing run ID, config hash, git commit, Python version, platform, model/engine/hardware identity, an evaluator command hash (`aislop_command_hash`), and per-case content hashes (`case_content_hashes`) so results are self-describing and case definitions can be verified across runs.
- **Rich provenance config schema** — engine name/version/accelerator/startup_flags (separate from hardware), hardware identity with `device_count`/`driver_version`/`runtime_version`, model artifact provenance (`source`, `revision`, `artifact_filename`, `artifact_checksum`), quantization, parameter count, seed, max_tokens, system prompt, expected answer, warmup trials. All provenance fields are optional; populate what you know.
- **Multi-dimensional quality scores** — `quality_scores` is `dict[str, float]` so evaluators can return sub-scores (correctness, style, hallucination, …).
- **Case-level evaluation contracts** — cases can run deterministic `python_tests`, `json_schema`, or documentation `rubric` checks in addition to the configured `aislop` score. Functional and instruction-following pass rates are therefore separate from static code quality.

### Evaluation Contracts

Use `evaluation.kind` to declare what a case must satisfy:

| Kind | Purpose | Result fields |
|---|---|---|
| `python_tests` | Extract Python from the response, execute it in a temporary isolated interpreter, and run configured assertions | `code_executed`, `tests_passed` |
| `json_schema` | Parse the response as JSON and validate required fields and primitive types | `schema_valid` |
| `rubric` | Check required and forbidden phrases for a deterministic task-specific coverage score | `rubric_coverage` |

The code check is a bounded subprocess convenience for benchmark isolation, not a security boundary for hostile code. Keep benchmark prompts and test contracts trusted, use a container for untrusted generations, and set a short per-case `timeout_seconds`. The `aislop` command remains an independent static-analysis signal and is not treated as a general answer-quality judge.

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

For machine-specific or private configurations, copy the example into
`configs/local/` and edit the copy. That directory is ignored by Git, so the
configuration stays local while remaining easy to organize alongside the
project:

```powershell
New-Item -ItemType Directory -Force configs/local | Out-Null
Copy-Item configs/example-benchmark.json configs/local/my-ollama.json
uv run messy-napkins --config configs/local/my-ollama.json
```

Reference configurations for specific hardware and engines are available in
`configs/examples/`. They are illustrative and may require substantial edits;
create real machine-specific configurations in `configs/local/`. Review
`configs/local/SANITIZATION.md` before publishing a local configuration as an
example.

Verified engine, model, and hardware findings are maintained in
`docs/hosting/`. Use those profiles as source material when creating a local
configuration, and preserve their distinction between requested, advertised,
and effective settings.

You can keep multiple private configurations there, for example one per model
or server. The runner loads the single file passed to `--config`; it does not
merge a local file with `example-benchmark.json`.

Or without uv:

```bash
PYTHONPATH=src python -m messy_napkins.runner --config configs/example-benchmark.json
```

This will:
1. Write a `run_manifest` row to the JSONL output with full provenance.
2. For each case: run `warmup_trials` (excluded from aggregates), then `trials` recorded runs.
3. Persist every trial row — including failures — to JSONL.
4. Measure real TTFT (HTTP streaming, `ttft_source: "streaming_api"`) or first-output timestamp (subprocess, `ttft_source: "first_chunk_approx"`), total duration, `tokens_per_second` and `decode_tokens_per_second` (only non-null when backend reports API token counts).
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
| `engine.startup_flags` | Extra CLI flags passed to the engine (array; for provenance logging) |
| `hardware.gpu` | GPU model name |
| `hardware.vram_gb` | Declared VRAM in GB |
| `hardware.cpu` | CPU model name |
| `hardware.os` | OS description |
| `hardware.device_count` | Number of GPU devices used |
| `hardware.driver_version` | GPU driver version string |
| `hardware.runtime_version` | CUDA/ROCm/Vulkan runtime version string |
| `quantization` | Quant format (e.g., `"Q4_K_M"`, `"Q8_0"`, `"F16"`) |
| `parameter_count` | Model size (e.g., `"7B"`, `"70B"`) |
| `seed` | RNG seed for reproducibility (`null` = not set) |
| `max_tokens` | Generation cap (`null` = backend default) |
| `parameters` | Sampler params forwarded to HTTP runner (`temperature`, `top_p`, …) |
| `source` | Model source/repository URL (optional) |
| `revision` | Model revision or commit hash (optional) |
| `artifact_filename` | Artifact filename, e.g. `"model-Q4_K_M.gguf"` (optional) |
| `artifact_checksum` | SHA-256 checksum of the model artifact file (optional) |

### Runner

| Field | Description |
|---|---|
| `type` | `"subprocess"` (default) or `"http"` |
| `command` | Command array for subprocess runner — prompt appended as final argument |
| `url` | OpenAI-compatible endpoint URL for HTTP runner (e.g., `"http://localhost:11434/v1/chat/completions"`) |
| `timeout_seconds` | Per-prompt timeout |
| `provider` | Metadata adapter: `"auto"` (default), `"lemonade"`, `"ollama"`, or `"lm-studio"` |

> **Config fidelity:** with `type: "http"`, `model.parameters`, `seed`, and `max_tokens` are sent as the literal API request payload and recorded in `effective_request`. This proves what the client sent, but not necessarily what the server applied. The run manifest performs a provider metadata preflight and records `provider_metadata`, `observed_*` fields, and `provider_field_verification`. Sampler settings remain `"sent_not_backend_verified"` in the evidence model unless the provider explicitly reports them. With `type: "subprocess"`, `model.parameters` are logged as metadata only and are **not** forwarded to the subprocess command. Prefer `type: "http"` whenever possible for reproducible, comparable benchmark results.

The Lemonade adapter queries `/api/v1/models` and `/api/v1/health` before an HTTP run. It records the raw responses and normalizes model identity, checkpoint, artifact format, quantization, parameter count, context size, recipe, accelerator, server version, readiness, and the llama.cpp sampler defaults reported in `recipe_options.llamacpp_args`. These are loaded-model/server defaults, not proof of the values applied to an individual completion request; request-level values remain in `effective_request`. Ollama uses `/api/tags`; LM Studio uses the OpenAI-compatible `/v1/models` endpoint. A failed metadata query is recorded as `provider_metadata_status: "unavailable"` and does not abort the benchmark.

> **TTFT:** with `type: "http"`, the runner sends `stream: true` and records `ttft_seconds` from the first SSE content token (`ttft_source: "streaming_api"`). With `type: "subprocess"`, TTFT is measured from the first non-whitespace stdout chunk (`ttft_source: "first_chunk_approx"`) and may be skewed by process/stdout buffering.

> **TPS:** `tokens_per_second` and `decode_tokens_per_second` are only non-null when the backend reports token counts via `usage.completion_tokens` (HTTP runner). The whitespace-chunk estimate is stored in `output_tokens_whitespace_approx` as a diagnostic field only; it is not used for TPS because word counts are not model tokens and differ across tokenizer families.

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
| `schema_version` | JSONL schema version (currently `3`) |
| `row_type` | `"run_manifest"` \| `"trial"` \| `"warmup"` \| `"aggregate"` |
| `run_id` | UUID linking all rows from the same benchmark run |
| `case_content_hash` | SHA-256 prefix of the case definition (detects if a case changed between runs) |
| `error` | `true` when the trial failed (inference or evaluation stage) |
| `error_type` / `error_message` / `timed_out` / `stage` | Failure details when `error: true` |
| `sampler_settings_source` | `"sent_not_backend_verified"` (HTTP: params provably sent in API request) \| `"unverified_metadata"` (subprocess: params logged but not forwarded) |
| `provider_metadata_status` | `"verified"`, `"unavailable"`, or `"not_requested"` for the provider preflight |
| `provider_metadata` | Raw model/config response returned by the provider metadata endpoint |
| `observed_model_id` / `observed_checkpoint` | Model identity reported by the provider |
| `observed_model_format` / `observed_quantization` | Artifact facts reported or unambiguously encoded in the provider model identity |
| `observed_parameter_count` / `observed_context_size` | Parameter count and context limit reported or normalized from provider metadata |
| `observed_engine` / `observed_accelerator` | Serving recipe and hardware backend reported by the provider |
| `observed_server_version` / `observed_model_status` | Provider version and loaded-model readiness state |
| `observed_server_llamacpp_args` | Raw llama.cpp defaults reported by Lemonade |
| `observed_server_sampler_defaults` | Parsed server defaults such as `temperature`, `top_p`, `top_k`, and `repeat_penalty` |
| `provider_field_verification` | Per-field status: `verified_match`, `mismatch`, `server_reported`, or `unverified` |
| `effective_command` | Actual subprocess command executed (subprocess runner) |
| `effective_request` | Actual HTTP payload sent verbatim (HTTP runner), including `stream_options` |
| `output_tokens` / `output_token_source` | Token count and source: `"api"` (backend-reported) \| `"unavailable"` (not reported) |
| `output_tokens_whitespace_approx` | Whitespace-chunk token estimate — diagnostic only, not used for TPS |
| `prompt_tokens` / `prompt_token_source` | Input token count and source |
| `ttft_seconds` | Time to first token |
| `ttft_source` | `"streaming_api"` (valid first-token latency from SSE) \| `"first_chunk_approx"` (subprocess, subject to buffering) |
| `total_seconds` | Full generation time |
| `tokens_per_second` | Throughput using total time; `null` when API token count unavailable |
| `decode_tokens_per_second` | Decode-phase throughput (post-first-token time); `null` when indistinguishable or tokens unavailable |
| `quality_scores` | Dict of named scores from the evaluator |
| `task_passed` | `true`/`false` from deterministic pass condition or evaluator; `null` if not determined |
| `vram_baseline_mb` | VRAM before inference started (nvidia-smi/rocm-smi); `null` if unavailable |
| `vram_peak_mb` | Peak VRAM during inference |
| `vram_peak_delta_mb` | `vram_peak_mb - vram_baseline_mb` (memory attributed to the inference call) |
| `vram_avg_delta_mb` | Average VRAM delta during inference |

Aggregate rows (`row_type: "aggregate"`) add: `execution_success_rate`, `task_pass_rate`, and per-metric statistics (`mean`, `stddev`, `median`, `p95`, `min`, `max`).

The run manifest (`row_type: "run_manifest"`) includes all the above model/engine/hardware identity fields plus `aislop_command_hash` (hash of the evaluator command), `case_content_hashes` (per-case content hashes), `model_source`, `model_revision`, `model_artifact_filename`, `model_artifact_checksum`, `engine_startup_flags`, `hardware_device_count`, `hardware_driver_version`, and `hardware_runtime_version`.

## Tests

Run the full test suite (34 tests):

```bash
uv run python -m unittest -v tests/test_benchmark.py
```

Or without uv:

```bash
PYTHONPATH=src python -m unittest -v tests/test_benchmark.py
```
