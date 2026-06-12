"""Configuration constants for the NITA bill Gradio app."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
while _PROJECT_ROOT.name and _PROJECT_ROOT.name not in {"", "."}:
    if (_PROJECT_ROOT / "pyproject.toml").exists():
        break
    if _PROJECT_ROOT.parent == _PROJECT_ROOT:
        break
    _PROJECT_ROOT = _PROJECT_ROOT.parent

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)

# Hugging Face tokenizers can emit fork/parallelism warnings in Gradio dev
# servers. Default this off unless the environment explicitly overrides it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Suppress a known huggingface_hub deprecation warning emitted during first-time
# model/tokenizer downloads. It is noisy but not actionable for app users.
warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated and will be removed in version 1\.0\.0\.",
    category=FutureWarning,
)

SUPPORTED_PROVIDERS = ["qwen", "openai", "anthropic", "gemini", "cohere"]
DEFAULT_PROVIDER: str = "qwen"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-14B:cheapest"
DEFAULT_CHUNK_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_REASONING_EFFORT = "medium"
ANTHROPIC_THINKING_BUDGET = 2048
DEFAULT_CHUNK_SIZE = 350
DEFAULT_CHUNK_OVERLAP = 60
SCAN_CHUNK_SIZE = 1200
SCAN_CHUNK_OVERLAP = 150
SCAN_MAX_WINDOWS = 40
SCAN_TOP_K = 5
SCAN_BATCH_SIZE = 6
TOP_K_RETRIEVAL = 5
MAX_UPLOAD_SIZE_MB = 25
TIMEOUT_SECONDS = 30

ProviderLiteral = Literal["qwen", "openai", "anthropic", "gemini", "cohere"]

# Conservative full-document QA input budgets derived from provider/model
# context-window docs, with headroom reserved for prompts and outputs.
PROVIDER_FULL_DOCUMENT_QA_TOKEN_BUDGETS: dict[ProviderLiteral, int] = {
    "qwen": 24_000,
    "openai": 900_000,
    "anthropic": 900_000,
    "gemini": 900_000,
    "cohere": 220_000,
}


@dataclass(frozen=True)
class ProviderConfig:
    name: ProviderLiteral
    key_prefix: Optional[str]
    display_name: str
    instructions: str


def _read_env_key(var_name: str) -> Optional[str]:
    value = os.getenv(var_name)
    if value is None:
        return None
    sanitized = value.strip().strip('"').strip("'")
    return sanitized or None


OPENAI_API_KEY: Optional[str] = _read_env_key("OPENAI_API_KEY")
ANTHROPIC_API_KEY: Optional[str] = _read_env_key("ANTHROPIC_API_KEY")
GEMINI_API_KEY: Optional[str] = _read_env_key("GEMINI_API_KEY")
COHERE_API_KEY: Optional[str] = _read_env_key("COHERE_API_KEY")
DEFAULT_COHERE_KEY: Optional[str] = _read_env_key("DEFAULT_COHERE_KEY")
HF_TOKEN: Optional[str] = _read_env_key("HF_TOKEN")


PROVIDER_METADATA: list[ProviderConfig] = [
    ProviderConfig(
        name="qwen",
        key_prefix=None,
        display_name="Qwen3 14B",
        instructions=(
            "Use your Hugging Face token for the router-backed Qwen model. Leave blank to use HF_TOKEN from .env if configured."
        ),
    ),
    ProviderConfig(
        name="openai",
        key_prefix="sk-",
        display_name="OpenAI GPT-5.5",
        instructions=(
            "Enter your OpenAI API key. Leave blank to use OPENAI_API_KEY from .env if configured."
        ),
    ),
    ProviderConfig(
        name="anthropic",
        key_prefix="sk-ant-",
        display_name="Anthropic Claude Sonnet 4.6",
        instructions=(
            "Provide your Anthropic API key. Leave blank to use ANTHROPIC_API_KEY from .env if configured."
        ),
    ),
    ProviderConfig(
        name="gemini",
        key_prefix=None,
        display_name="Google Gemini 2.5 Flash",
        instructions=(
            "Use your Gemini API key. Leave blank to use the built-in GEMINI_API_KEY if configured."
        ),
    ),
    ProviderConfig(
        name="cohere",
        key_prefix=None,
        display_name="Cohere Command A Reasoning",
        instructions=(
            "Use your Cohere API key with Command R access. Leave blank to use COHERE_API_KEY or "
            "DEFAULT_COHERE_KEY if configured."
        ),
    ),
]


APP_TITLE = "Legislation Explainer"
APP_DESCRIPTION = (
    "A Gradio policy assistant for public-interest legislation. "
    "Upload or link to a bill, generate a structured review, and ask grounded follow-up questions."
)
