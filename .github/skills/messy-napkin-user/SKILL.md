---
name: messy-napkin-user
description: 'Benchmark-runner workflow: confirm which messy-napkins config matches a live-hosted model (or that a specific config is still accurate), then run the benchmark suite against it and save results. Use when a user gives minimal connection info (e.g. "Qwen3-14B-GGUF at http://localhost:13305/") and asks which config to use, wants to re-check a config still matches reality, or wants to actually run a benchmark and get results. Never creates, edits, or deletes a config file or a docs/hosting/ entry — deeply technical config changes belong to model-configurator. May run the existing messy-napkins CLI, which is expected to write result logs under logs/.'
---

# messy-napkin-user (Verify & Run)

## When to Use

Three related jobs, all without touching config content:

- **Identification**: "Which config should I use for the model at this
  endpoint?" — given only connection info + model identity, find the
  best-matching existing config.
- **Verification**: "Is this config still right for this target?" (or "is
  this target still hosted the way this config expects?") — given connection
  info + model identity (and usually a specific config to check), confirm
  whether they still agree.
- **Run**: "Run the benchmark and save the results" — once the right config
  is identified or verified, execute it with the existing CLI and report
  where the results landed.

## Hard Constraint

**This skill never creates, edits, or deletes a config file under
`configs/`, or a doc under `docs/hosting/`.** Deeply technical config changes
(context size, engine startup flags, hardware, sampler defaults) belong to
[model-configurator](../model-configurator/SKILL.md) — hand off there and
stop, rather than fixing it yourself, even if the fix looks trivial.

Running the benchmark suite via the existing CLI (`messy-napkins --config
<existing-config>`) is expected and allowed even though it writes result
files under `logs/` — that is a normal benchmarking side effect, not a config
change, and is the whole point of the Run job.

## Required Inputs

- How to reach the target: URL/IP/port/domain, and API key if required.
- Model identity as reported by the user (e.g. "Qwen3-14B-GGUF").
- For Verification or Run mode: which config to check/run, if the user has
  one in mind (otherwise treat it as Identification first).

If ambiguous or insufficient, probe the live endpoint before asking the user
anything. Only ask the user a question if probing still leaves a genuine
ambiguity (e.g. two configs equally match the observed metadata).

## Procedure

1. **Query live metadata**: hit the model/health endpoints appropriate to the
   detected provider (Lemonade `/api/v1/models` + `/api/v1/health`, Ollama
   `/api/tags` + `/api/show`, OpenAI-compatible `/v1/models`) to learn model
   id, quantization, advertised context, engine/accelerator/version.
2. **Search existing configs**: look through `configs/local/*.json`,
   `configs/examples/*.json`, and `docs/hosting/*.md` profiles for a match on
   model id, checkpoint, or engine.
3. **Identification mode**: rank candidate configs by how well their
   recorded fields (model id, quantization, engine, hardware) match the live
   metadata. Report the best match and any discrepancies. If none match
   well, say so explicitly rather than forcing a pick.
4. **Verification mode**: compare the specific config's recorded fields
   (`context_size`, `engine.*`, `quantization`, `parameters`) against the
   live metadata field by field. If a deeper check is warranted (e.g.
   confirming effective context), send a small number of direct completion
   requests straight to the endpoint to observe API-reported
   `prompt_tokens`/`output_tokens` at a couple of sizes — do this with a
   direct HTTP call (e.g. a terminal `Invoke-RestMethod`/`curl` command),
   never by creating a config file under `configs/`.
5. **Run mode**: once a config is confirmed accurate (via step 3 or 4, or
   because the user already knows which one to use), run it with
   `messy-napkins --config <config path>` and report the results file path
   (`logs/*.jsonl`) plus a brief summary of what ran.
6. **Report findings**: state what matched, what didn't, and what's Unknown
   (couldn't be probed), using Verified / Observed / Unknown labels. Do not
   silently fix a mismatched config.
7. **If the user wants a mismatch fixed**: hand off to
   [model-configurator](../model-configurator/SKILL.md) — do not edit the
   config yourself, even to apply an "obvious" fix.

## Related

- [model-configurator](../model-configurator/SKILL.md) — the write-capable
  counterpart for creating or correcting configs and hosting docs.
- [docs/hosting/README.md](../../../docs/hosting/README.md) — evidence-label
  conventions.
