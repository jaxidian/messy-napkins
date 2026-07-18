# Model Hosting Findings

This directory records verified hosting behavior for engines, models, and
hardware combinations. The documents are intended to be useful both to people
and as structured source material when creating files under `configs/local/`
or `configs/examples/`.

## Evidence labels

- **Verified:** Confirmed by server metadata, server logs, or a boundary test.
- **Observed:** True for the recorded environment, but not established as a
  portable engine or model property.
- **Recommended:** A configuration choice derived from verified evidence.
- **Unknown:** A value that must be collected before claiming reproducibility.

Keep engine behavior separate from model limits. A server can accept a setting
that the model runtime later caps, and a model can support a capability that a
particular backend or hardware configuration cannot provide.

## Profiles

- [Lemonade](lemonade.md): OpenAI-compatible endpoint, metadata, startup
  behavior, and Windows/Vulkan observations.
- [Qwen3-14B-GGUF](qwen3-14b-gguf.md): artifact identity, context-window
  evidence, sampler settings, and a config-ready profile.

## Config generation order

1. Start with the model profile for identity, quantization, context limits,
   and model-specific request parameters.
2. Apply the engine profile for endpoint, provider, timeout, and startup
   settings.
3. Fill in the actual host hardware, driver, runtime, and engine version.
4. Verify the loaded model through provider metadata and server logs.
5. Keep requested, model-supported, and effective context sizes distinct.
6. Save private or machine-specific output under `configs/local/`.

Do not copy an observed value into a config merely because it is concrete.
First check whether it belongs to the model, engine, accelerator, operating
system, or one specific machine.