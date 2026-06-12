"""Provider utilities for handling user-supplied API keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from anthropic import Anthropic
from openai import OpenAI

from config import (
    ANTHROPIC_API_KEY,
    COHERE_API_KEY,
    DEFAULT_PROVIDER,
    DEFAULT_COHERE_KEY,
    DEFAULT_NEMOTRON_MODEL,
    GEMINI_API_KEY,
    DEFAULT_QWEN_MODEL,
    HF_TOKEN,
    NEMOTRON_API_KEY,
    NEMOTRON_BASE_URL,
    OPENAI_API_KEY,
    PROVIDER_METADATA,
    ProviderConfig,
    ProviderLiteral,
)


@dataclass
class ProviderClient:
    """Container for a provider client and default model."""

    name: ProviderLiteral
    client: Any
    default_model: str
    api_key: str


_PROVIDER_MODEL_MAP: Dict[ProviderLiteral, str] = {
    "nemotron": DEFAULT_NEMOTRON_MODEL,
    "qwen": DEFAULT_QWEN_MODEL,
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.5-flash",
    "cohere": "command-a-reasoning-08-2025",
}


_DEFAULT_PROVIDER_KEYS: Dict[ProviderLiteral, Optional[str]] = {
    "nemotron": NEMOTRON_API_KEY or HF_TOKEN,
    "qwen": HF_TOKEN,
    "openai": OPENAI_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "gemini": GEMINI_API_KEY,
    "cohere": COHERE_API_KEY or DEFAULT_COHERE_KEY,
}


def get_default_api_key(provider: ProviderLiteral) -> Optional[str]:
    return _DEFAULT_PROVIDER_KEYS.get(provider)


def get_provider_config(name: ProviderLiteral) -> ProviderConfig:
    for config in PROVIDER_METADATA:
        if config.name == name:
            return config
    raise ValueError(f"Unsupported provider: {name}")


def validate_api_key(provider: ProviderLiteral, api_key: str | None) -> tuple[bool, Optional[str]]:
    resolved_key = api_key or get_default_api_key(provider)
    if not resolved_key:
        return False, "API key is required."

    config = get_provider_config(provider)
    if config.key_prefix:
        if not resolved_key.startswith(config.key_prefix):
            return False, f"Expected key to start with '{config.key_prefix}'."

    if len(resolved_key.strip()) < 10:
        return False, "API key looks too short."

    return True, None


def instantiate_client(provider: ProviderLiteral, api_key: str) -> ProviderClient:
    if provider == "nemotron":
        client = OpenAI(api_key=api_key, base_url=NEMOTRON_BASE_URL)
    elif provider == "qwen":
        client = OpenAI(api_key=api_key, base_url="https://router.huggingface.co/v1")
    elif provider == "openai":
        client = OpenAI(api_key=api_key)
    elif provider == "anthropic":
        client = Anthropic(api_key=api_key)
    elif provider == "gemini":
        try:
            import google.generativeai as genai
        except ImportError as exc:  # pragma: no cover - defensive path
            raise RuntimeError(
                "google-generativeai must be installed to use the Gemini provider"
            ) from exc

        genai.configure(api_key=api_key)
        client = genai
    elif provider == "cohere":
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - defensive path
            raise RuntimeError("cohere must be installed to use the Cohere provider") from exc

        client = cohere.Client(api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    default_model = _PROVIDER_MODEL_MAP[provider]
    return ProviderClient(name=provider, client=client, default_model=default_model, api_key=api_key)


def list_providers() -> list[ProviderConfig]:
    return sorted(PROVIDER_METADATA, key=lambda config: config.name != DEFAULT_PROVIDER)
