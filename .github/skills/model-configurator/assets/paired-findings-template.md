# Findings: <config-name>.json

Married 1:1 to [`<config-name>.json`](../../../../configs/local/<config-name>.json).
This report is instance-specific and records the exact hosting hypothesis,
live evidence, resulting config decisions, recommended commands, and the
benchmark history derived from JSONL results.

For portable model or engine facts, see [`docs/hosting/`](../../../../docs/hosting/README.md).

## Target

- Model: `<model-id>`
- Engine: `<engine> <version>`
- Endpoint: `<endpoint>`
- Host/hardware: `<host summary>`

## <YYYY-MM-DD> - Configuration verification

### Reported commands

Record the user's commands and environment variables verbatim.

```powershell
<reported commands>
```

### Live evidence

- **Verified:** `<metadata, logs, or boundary evidence>`
- **Observed:** `<machine-specific observations>`
- **Unknown:** `<values that could not be probed>`

### Resulting configuration

Explain the important resulting fields and why they have these values. Keep
requested, advertised, training, and effective context sizes distinct.

- `context_size`: `<effective value and evidence>`
- `engine.startup_flags`: `<recorded provenance and evidence status>`
- Hardware: `<observed hardware and evidence status>`
- Quantization/source/revision: `<values and evidence status>`

## Recommended commands

Provide the ideal or least-confusing commands for reproducing this
configuration. Explain every difference from the reported commands. A command
that was not successfully executed must be labeled **Recommended/Untested**;
do not call it Verified.

```powershell
<recommended commands>
```

- **Verified:** `<commands successfully executed>`
- **Recommended:** `<reason for changes>`
- **Untested assumptions:** `<provider precedence, optional flags, etc.>`

## Benchmark history dashboard

The JSONL result file is the source of truth. This section is a curated visual
index only. Cite the full JSONL run IDs/source path, and compare only runs with
compatible config hashes, model settings, and host conditions. Assign a
chronological artificial `Run #` (`1`, `2`, `3`, ...) for the table and Mermaid
X axes; never use GUIDs as chart labels.

Source: [`logs/<results-file>.jsonl`](../../../../logs/<results-file>.jsonl)

### Run history

| Run # | Started | Cases | Passed | Mean tok/s | Mean TTFT | Source |
|---|---|---:|---:|---:|---:|---|
| 1 | `<timestamp>` | `<count>` | `<passed>` | `<value>` | `<value>` | JSONL run `<full-run-id>` |

### Throughput trend

```mermaid
xychart-beta
    title "Mean generation throughput by run"
    x-axis [1]
    y-axis "tokens/sec" 0 --> <max>
    line [<value>]
```

### Time-to-first-token trend

```mermaid
xychart-beta
    title "Mean TTFT by run"
    x-axis [1]
    y-axis "milliseconds" 0 --> <max>
    line [<value>]
```

### Dashboard notes

- `<anomalies, incomparable runs, or missing metrics>`
- The dashboard is derived from `run_manifest` and `aggregate` rows; do not
  copy raw trial rows into this report.
