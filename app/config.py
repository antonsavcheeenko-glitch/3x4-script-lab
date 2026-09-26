"""Конфігурація застосунку: шляхи та налаштування LLM з env-змінних."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = Path(os.getenv("APP_DATA_DIR", ROOT_DIR / "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLES_DIR = ROOT_DIR / "data" / "samples"
DB_PATH = DATA_DIR / "app.db"

PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4.1-mini",
    "ollama": "llama3.1",
}


@dataclass
class LLMSettings:
    provider: str
    model: str
    anthropic_api_key: str
    openai_api_key: str
    openai_base_url: str
    ollama_base_url: str
    timeout: int


def provider_model(provider: str) -> str:
    env_specific = os.getenv(f"{provider.upper()}_MODEL", "").strip()
    return env_specific or PROVIDER_DEFAULT_MODELS.get(provider, "")


def load_llm_settings() -> LLMSettings:
    provider = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic").strip().lower() or "anthropic"
    model = os.getenv("DEFAULT_LLM_MODEL", "").strip() or provider_model(provider)
    return LLMSettings(
        provider=provider,
        model=model,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "").strip(),
        timeout=int(os.getenv("LLM_TIMEOUT", "180") or 180),
    )


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
