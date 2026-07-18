from __future__ import annotations

import json
import re
import shlex
import urllib.parse
import urllib.request
from typing import Any


class ProviderMetadataAdapter:
    """Fetch and normalize server/model metadata before a benchmark run."""

    name = "unknown"

    def fetch(self, runner_url: str, timeout_seconds: int) -> dict[str, Any]:
        raise NotImplementedError


def _server_root(runner_url: str) -> str:
    parsed = urllib.parse.urlparse(runner_url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "", "", "", ""))


def _get_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError(f"Metadata endpoint returned {type(value).__name__}, expected an object")
    return value


def _first_model(data: dict[str, Any]) -> dict[str, Any]:
    models = data.get("data")
    if isinstance(models, list) and models and isinstance(models[0], dict):
        return models[0]
    models = data.get("models")
    if isinstance(models, list) and models and isinstance(models[0], dict):
        return models[0]
    return {}


def _model_facts(model: dict[str, Any]) -> dict[str, Any]:
    """Normalize facts without presenting inferred values as server assertions."""
    checkpoint = str(model.get("checkpoint") or model.get("name") or model.get("model") or model.get("id") or "")
    recipe_options = model.get("recipe_options") if isinstance(model.get("recipe_options"), dict) else {}
    details = model.get("details") if isinstance(model.get("details"), dict) else {}
    identity = " ".join(str(value) for value in (checkpoint, model.get("id", "")))
    quantization_match = re.search(r"(?i)(Q[0-9](?:_[A-Z0-9]+)?)", identity)
    parameter_match = re.search(r"(?i)([0-9]+(?:\.[0-9]+)?[BM])(?:[-_:]|$)", identity)
    if parameter_match is None:
        parameter_match = re.search(r"(?i)\b([0-9]+(?:\.[0-9]+)?[BM])\b", identity)

    quantization = model.get("quantization") or details.get("quantization_level")
    if not quantization and quantization_match:
        quantization = quantization_match.group(1).upper()
    parameter_count = model.get("parameter_count") or details.get("parameter_size")
    if not parameter_count and parameter_match:
        parameter_count = parameter_match.group(1).upper()

    artifact_format = model.get("format") or details.get("format")
    if not artifact_format and re.search(r"(?i)(?:\.gguf|[-_:]gguf)(?:$|[:])", checkpoint):
        artifact_format = "GGUF"
    return {
        "model_id": model.get("id") or model.get("name") or model.get("model"),
        "checkpoint": checkpoint or None,
        "model_format": artifact_format,
        "quantization": quantization,
        "parameter_count": parameter_count,
        "context_size": recipe_options.get("ctx_size") or model.get("max_context_window"),
        "engine": model.get("recipe"),
        "accelerator": recipe_options.get("llamacpp_backend"),
    }


def _parse_llamacpp_args(args: str | None) -> dict[str, Any]:
    """Parse Lemonade's reported llama.cpp defaults without guessing request overrides."""
    if not args:
        return {}
    tokens = shlex.split(args)
    values: dict[str, Any] = {}
    aliases = {
        "--temp": "temperature",
        "--top-p": "top_p",
        "--top-k": "top_k",
        "--min-p": "min_p",
        "--repeat-penalty": "repeat_penalty",
    }
    index = 0
    while index < len(tokens):
        key = aliases.get(tokens[index])
        if key and index + 1 < len(tokens):
            raw_value = tokens[index + 1]
            try:
                value: Any = float(raw_value)
                if value.is_integer():
                    value = int(value)
            except ValueError:
                value = raw_value
            values[key] = value
            index += 2
            continue
        index += 1
    return values


class LemonadeMetadataAdapter(ProviderMetadataAdapter):
    name = "lemonade"

    def fetch(self, runner_url: str, timeout_seconds: int) -> dict[str, Any]:
        root = _server_root(runner_url)
        models_endpoint = f"{root}/api/v1/models"
        health_endpoint = f"{root}/api/v1/health"
        raw_models = _get_json(models_endpoint, timeout_seconds)
        raw_health = _get_json(health_endpoint, timeout_seconds)
        model = _first_model(raw_models)
        loaded_models = raw_health.get("all_models_loaded")
        loaded_model = loaded_models[0] if isinstance(loaded_models, list) and loaded_models else {}
        health_options = loaded_model.get("recipe_options") if isinstance(loaded_model.get("recipe_options"), dict) else {}
        facts = _model_facts({**model, **loaded_model, "recipe_options": {**(model.get("recipe_options") or {}), **health_options}})
        facts.update({
            "server_version": raw_health.get("version"),
            "model_status": loaded_model.get("status"),
            "backend_health": loaded_model.get("backend_health"),
            "device": loaded_model.get("device"),
            "server_llamacpp_args": health_options.get("llamacpp_args"),
            "server_sampler_defaults": _parse_llamacpp_args(health_options.get("llamacpp_args")),
        })
        return {
            "provider": self.name,
            "endpoint": [models_endpoint, health_endpoint],
            "raw": {"models": raw_models, "health": raw_health},
            "facts": facts,
        }


class OllamaMetadataAdapter(ProviderMetadataAdapter):
    name = "ollama"

    def fetch(self, runner_url: str, timeout_seconds: int) -> dict[str, Any]:
        endpoint = f"{_server_root(runner_url)}/api/tags"
        raw = _get_json(endpoint, timeout_seconds)
        model = _first_model(raw)
        return {"provider": self.name, "endpoint": endpoint, "raw": raw, "facts": _model_facts(model)}


class OpenAIModelsMetadataAdapter(ProviderMetadataAdapter):
    name = "openai-compatible"

    def fetch(self, runner_url: str, timeout_seconds: int) -> dict[str, Any]:
        endpoint = f"{_server_root(runner_url)}/v1/models"
        raw = _get_json(endpoint, timeout_seconds)
        model = _first_model(raw)
        return {"provider": self.name, "endpoint": endpoint, "raw": raw, "facts": _model_facts(model)}


def get_metadata_adapter(provider: str, runner_url: str) -> ProviderMetadataAdapter:
    normalized = (provider or "auto").lower()
    if normalized == "auto":
        parsed = urllib.parse.urlparse(runner_url)
        if parsed.port == 13305:
            normalized = "lemonade"
        elif parsed.port == 11434:
            normalized = "ollama"
        else:
            normalized = "openai-compatible"
    adapters: dict[str, ProviderMetadataAdapter] = {
        "lemonade": LemonadeMetadataAdapter(),
        "ollama": OllamaMetadataAdapter(),
        "lm-studio": OpenAIModelsMetadataAdapter(),
        "openai-compatible": OpenAIModelsMetadataAdapter(),
    }
    try:
        return adapters[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported metadata provider: {provider}") from exc


def collect_provider_metadata(
    provider: str,
    runner_url: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    adapter = get_metadata_adapter(provider, runner_url)
    try:
        result = adapter.fetch(runner_url, timeout_seconds)
        result["status"] = "verified"
        return result
    except Exception as exc:
        return {
            "provider": adapter.name,
            "status": "unavailable",
            "endpoint": None,
            "raw": None,
            "facts": {},
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
