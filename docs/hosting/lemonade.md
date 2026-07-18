# Hosting with Lemonade

## Scope

This profile is based on Lemonade Server 11.0.0 running on Windows 11 with the
llama.cpp recipe and Vulkan backend. The HTTP and metadata behavior is
Lemonade-specific. Windows paths and process details are Windows-specific.
Device enumeration and memory behavior are backend- and hardware-specific.

## Config-ready engine profile

```json
{
  "model": {
    "engine": {
      "name": "lemonade",
      "version": "11.0.0",
      "accelerator": "vulkan",
      "startup_flags": []
    }
  },
  "runner": {
    "type": "http",
    "url": "http://localhost:13305/v1/chat/completions",
    "provider": "lemonade",
    "timeout_seconds": 120
  }
}
```

Set `startup_flags` to the arguments actually used to load the model. These
flags are provenance in messy-napkins; the benchmark does not start Lemonade
or forward them to the server.

## Endpoints and evidence

Lemonade exposes an OpenAI-compatible completion endpoint at
`/v1/chat/completions`. messy-napkins also queries:

- `/api/v1/models` for registered model identity and advertised limits.
- `/api/v1/health` for the loaded backend, recipe options, device, and status.

Use both metadata responses and the backend startup log. A recipe value records
what was requested, while the llama.cpp log can reveal what became effective.

### Context-size distinction

The command below was accepted and passed through correctly:

```powershell
lemonade run Qwen3-14B-GGUF --ctx-size 65536
```

The corresponding llama-server command contained `--ctx-size 65536`.
Therefore, Lemonade did not ignore the requested size. llama.cpp subsequently
capped it because the model reported a smaller training context.

Keep these values separate:

| Value | Meaning | Evidence source |
|---|---|---|
| Requested context | Value passed to `lemonade run --ctx-size` | CLI and Lemonade recipe options |
| Advertised model context | Model's `max_context_window` | `/api/v1/models` or loaded-model metadata |
| Training context | Model's `n_ctx_train` | llama.cpp startup log |
| Effective context | Context assigned to serving slots | llama.cpp `new slot, n_ctx` log and boundary tests |

The current messy-napkins `model.context_size` field is metadata. It is not
included in the OpenAI-compatible completion request and cannot resize an
already loaded model. Set it to the effective context for honest run records.

### Metadata caveat

In the observed Lemonade response, `/api/v1/health` temporarily reported
`recipe_options.ctx_size: 65536`, while the model record still reported
`max_context_window: 40960`. The current metadata adapter merges these records
and may normalize the requested recipe size as `observed_context_size`.

Until requested and effective context receive separate manifest fields, inspect
the raw `provider_metadata`, model startup log, and boundary-test behavior when
the values disagree.

## Windows-specific observations

- Lemonade launched `llama-server.exe` from its user cache.
- The model artifact was stored under the user's Hugging Face cache.
- Use paths reported by Lemonade only for diagnosis; do not publish absolute
  cache paths containing a Windows username.
- The executable and cache layout are not portable to Linux or macOS.
- `--no-mmap` appeared in the generated llama-server command for this run. Do
  not treat that as a universal Lemonade requirement without testing another
  operating system and backend.

## Vulkan-specific observations

- Lemonade selected `llamacpp_backend=vulkan` and llama.cpp enumerated both the
  discrete AMD GPU and integrated AMD graphics.
- The observed host did not specify a Vulkan device explicitly. Record a
  device-selection argument if one is later required for reproducibility.
- The startup log identified the discrete device as AMD Radeon RX 9070 XT with
  16,304 MiB visible memory. This is a host observation, not a Lemonade limit.
- messy-napkins VRAM telemetry currently samples `nvidia-smi` or `rocm-smi`.
  A Windows Vulkan run may therefore lack process VRAM telemetry even though
  llama.cpp uses the GPU.

## Validation checklist

1. Load the model with the intended Lemonade arguments.
2. Confirm the exact llama-server command in the Lemonade log.
3. Confirm `backend_health=ready` and the intended checkpoint in
   `/api/v1/health`.
4. Compare recipe `ctx_size`, model `max_context_window`, `n_ctx_train`, and
   serving-slot `n_ctx`.
5. Run a prompt boundary test below and above the claimed effective context.
6. Record the Lemonade version, backend, driver/runtime, and device identity.