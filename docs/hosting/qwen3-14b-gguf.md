# Qwen3-14B-GGUF

## Scope and identity

This profile describes the Q4_0 artifact of `unsloth/Qwen3-14B-GGUF` hosted by
Lemonade 11.0.0 through llama.cpp and Vulkan on Windows 11.

| Property | Value | Status |
|---|---|---|
| Model ID | `Qwen3-14B-GGUF` | Verified by Lemonade metadata |
| Checkpoint | `unsloth/Qwen3-14B-GGUF:Q4_0` | Verified by Lemonade metadata |
| Artifact | `Qwen3-14B-Q4_0.gguf` | Verified by startup log |
| Artifact revision | `a04a82c4739b3ef5fa6da7d10261db2c67dd1985` | Observed cache snapshot |
| Format | GGUF | Verified by artifact and metadata |
| Quantization | Q4_0 | Verified by checkpoint and artifact |
| Parameter count | 14B | Verified by model identity |
| Training context | 40,960 tokens | Verified by llama.cpp `n_ctx_train` |
| Effective context | 40,960 tokens | Verified by slot logs and boundary test |
| Artifact checksum | Unknown | Compute before claiming immutable provenance |

The revision above identifies the observed local Hugging Face snapshot. Confirm
that it is a stable upstream revision before using it as a permanent source
revision, and add a SHA-256 checksum for the GGUF artifact when exact
reproducibility is required.

## Context-window findings

Lemonade correctly passed `--ctx-size 65536` to llama.cpp. The runtime logged:

```text
n_ctx_seq (65536) > n_ctx_train (40960) -- possible training context overflow
the slot context (65536) exceeds the training context of the model (40960) - capping
new slot, n_ctx = 40960
```

This is model/runtime behavior, not a Windows, Vulkan, or RX 9070 XT memory
limit. Different hardware may change whether a large KV cache fits or how fast
it runs, but the observed cap was explicitly based on the model's training
context.

Boundary tests provided matching behavioral evidence:

| API prompt tokens | Result |
|---:|---|
| 16,024 | Response and usage returned |
| 32,024 | Response and usage returned |
| 39,024 | Response and usage returned |
| 42,024 | Server rejected the request as exceeding 40,960 |
| Approximately 60K | Empty client response; beyond effective limit |
| Approximately 68K | Empty client response; beyond effective limit |

The client observed an empty streaming response for the 42K request, while the
Lemonade log contained the explicit context error. When probing limits, inspect
both client results and server logs.

### Prompt budget

The context budget includes chat-template tokens and requested completion
tokens, not just visible prompt text:

```text
prompt tokens + maximum completion tokens <= effective context
```

With `max_tokens: 2048`, the theoretical prompt ceiling is 38,912 tokens before
chat-template overhead. Use a lower operational ceiling. The boundary prompt
added approximately 24 template tokens in the observed setup.

## Recommended messy-napkins model profile

```json
{
  "id": "Qwen3-14B-GGUF",
  "engine": {
    "name": "lemonade",
    "version": "11.0.0",
    "accelerator": "vulkan",
    "startup_flags": ["--ctx-size", "40960"]
  },
  "hardware": {
    "gpu": "AMD Radeon RX 9070 XT",
    "vram_gb": 16.0,
    "cpu": "AMD Ryzen 9 9900X 12-Core Processor",
    "os": "Windows 11",
    "device_count": 1,
    "driver_version": "",
    "runtime_version": ""
  },
  "quantization": "Q4_0",
  "parameter_count": "14B",
  "seed": 42,
  "max_tokens": 2048,
  "context_size": 40960,
  "parameters": {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "chat_template_kwargs": {
      "enable_thinking": false
    }
  },
  "source": "https://huggingface.co/unsloth/Qwen3-14B-GGUF",
  "revision": "a04a82c4739b3ef5fa6da7d10261db2c67dd1985",
  "artifact_filename": "Qwen3-14B-Q4_0.gguf",
  "artifact_checksum": ""
}
```

The hardware block above is an observed host profile, not a requirement for the
model. Replace it when generating a config for another machine. The RX 9070 XT
name comes directly from the Vulkan device log; earlier RX 5070 XT and RX 3070
XT references were not supported by the captured evidence.

## Sampler and thinking behavior

The observed Lemonade server defaults were temperature `0.6`, top-p `0.85`,
top-k `20`, min-p `0.0`, and repeat penalty `1.0`. The benchmark intentionally
sent temperature `0.2` and top-p `0.95`, along with the other values shown
above. Request values and server defaults should be recorded separately.

`chat_template_kwargs.enable_thinking: false` was sent by the benchmark. This
is model/template-specific request behavior, not a Windows, Vulkan, or GPU
setting.

## Known benchmark behavior

Across two comparable five-trial runs, throughput changed by about one percent,
which was consistent for the observed host. Functional results were stable for
most cases. The arithmetic and complex TTL-cache cases failed repeatedly, while
slugify varied between runs. These are benchmark observations, not universal
claims about all Qwen3-14B prompts or quantizations.

Do not compare these numbers directly with another engine, quantization,
context allocation, thinking mode, or hardware profile without preserving the
full run manifest.