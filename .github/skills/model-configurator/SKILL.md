---
name: model-configurator
description: 'Manage messy-napkins hosting configs: probe a hosted LLM (Lemonade, Ollama, OpenAI-compatible, etc.) to generate a targeted benchmark configuration, or correct an existing config''s deeply technical fields (context size, engine startup flags, hardware, sampler defaults). Use when a user provides hosting commands, connection details (URL/IP/domain/API key), and freeform notes and wants a config produced or fixed. Probes live metadata and boundary limits to fill in as much as possible, then asks the user only for what cannot be discovered automatically. Produces/updates a configs/local/*.json plus a corresponding dated markdown findings entry — this skill writes files. This is the only role that should change configs/ or docs/hosting/ content; benchmark runners should be pointed here instead of editing configs themselves.'
---

# Model Configurator

## When to Use

- The user gives you: (a) the command(s)/env vars they used to host a model,
  (b) how to reach it (URL, IP, port, domain, API key if any), and (c) any
  extra unstructured notes (hardware, quantization, intent) — and wants a
  benchmark-ready config generated or an existing one corrected.
- Trigger phrases: "probe this model", "generate a config for...", "I hosted
  it with these commands, make me a config", "fix/update this config's
  context size/engine flags".
- A `messy-napkin-user` session found a config that no longer matches its
  target and the user wants it corrected — that fix belongs here, not there.

## Required Inputs

Ask for only what's missing, after attempting to probe for it:

1. Hosting command(s) / environment variables used to launch the server
   (verbatim — do not paraphrase).
2. Connection info: base URL or IP:port, API key if required, provider hint
   if known (lemonade / ollama / openai-compatible).
3. Any additional context the user wants to provide (hardware, intended
   quantization, use case) — optional, freeform.

Do not block on details you can discover by probing. Only ask the user for
things that genuinely cannot be determined externally (e.g. a friendly config
name, specific sampler parameters they want benchmarked, an artifact checksum
that isn't computable from what's exposed).

## Procedure

1. **Capture the hypothesis** verbatim: commands, env vars, connection info.
2. **Probe live metadata**: query the server's model/health endpoints (e.g.
   Lemonade `/api/v1/models`, `/api/v1/health`; Ollama `/api/tags`,
   `/api/show`; OpenAI-compatible `/v1/models`) to discover model id,
   quantization, advertised context, engine/accelerator, and backend version.
3. **Probe effective limits**: generate a boundary-test config with
   [create-context-boundary-config.py](../../../configs/examples/create-context-boundary-config.py)
   (`--targets` bracketing the advertised or claimed context) and run it with
   `messy-napkins --config <generated-config>` to get API-reported
   `prompt_tokens` evidence of the true effective context, not just the
   advertised one.
4. **Cross-check server logs if the user can supply them.** Metadata
   endpoints may not reveal custom launch flags (GPU layers, KV cache type,
   flash attention, mmap, etc.) — only the literal startup log proves those
   took effect. Label as Unknown if unavailable.
5. **Fill remaining gaps** by asking the user only for what step 1-4 could
   not resolve.
6. **Produce or update the config**: write `configs/local/<name>.json` with
   the verified `context_size`, `engine.startup_flags`, hardware,
   quantization, and source/revision where discoverable. If correcting an
   existing config, edit it in place rather than creating a duplicate.
7. **Produce the markdown findings**: if a profile for this model/engine
   already exists under `docs/hosting/`, append a dated "Verification log"
   entry there. Otherwise create a new profile following the evidence-label
   conventions in [docs/hosting/README.md](../../../docs/hosting/README.md)
   (Verified / Observed / Recommended / Unknown).

## Output Contract

This skill **writes files**: a config under `configs/local/` and a markdown
entry under `docs/hosting/` (new profile or appended log entry). Confirm the
destination paths with the user if ambiguous. This is the only skill that
should modify configs or `docs/hosting/` content.

## Evidence Discipline

- A successful HTTP response is not proof of a claimed limit — check
  API-reported token counts and, when possible, server logs.
- Distinguish requested / advertised / training / effective values explicitly.
- Never assume env-var vs. CLI-flag precedence without a probe if both are
  set to different values at the same time.

## Related

- [messy-napkin-user](../messy-napkin-user/SKILL.md) — read-only
  identification/verification and benchmark-running counterpart; it hands
  off config corrections here instead of making them itself.
- [configs/local/SANITIZATION.md](../../../configs/local/SANITIZATION.md) —
  checklist before sharing any generated config or log excerpt.
