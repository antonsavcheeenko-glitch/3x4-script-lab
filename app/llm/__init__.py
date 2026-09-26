"""Фабрика провайдерів + сервіс виклику LLM з логуванням у БД."""
from __future__ import annotations

from app.config import LLMSettings, load_llm_settings, provider_model
from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import LLMError, LLMNotConfigured, LLMProvider, LLMResponse, extract_json
from app.llm.ollama_provider import OllamaProvider
from app.llm.openai_provider import OpenAIProvider

PROVIDERS = ["anthropic", "openai", "ollama"]

__all__ = [
    "LLMError", "LLMNotConfigured", "LLMProvider", "LLMResponse", "extract_json",
    "PROVIDERS", "build_provider", "LLMService",
]


def build_provider(name: str, model: str | None = None, settings: LLMSettings | None = None) -> LLMProvider | None:
    s = settings or load_llm_settings()
    model = model or (s.model if name == s.provider else provider_model(name))
    if name == "anthropic":
        return AnthropicProvider(s.anthropic_api_key, model, s.timeout)
    if name == "openai":
        return OpenAIProvider(s.openai_api_key, model, s.openai_base_url, s.timeout)
    if name == "ollama":
        return OllamaProvider(s.ollama_base_url, model, max(s.timeout, 300))
    return None


class LLMService:
    """Обгортка над провайдером: логування викликів та зручні методи."""

    def __init__(self, provider: LLMProvider | None, session=None, project_id: int | None = None):
        self.provider = provider
        self.session = session
        self.project_id = project_id

    @property
    def available(self) -> bool:
        return self.provider is not None and self.provider.is_configured()

    @property
    def label(self) -> str:
        if not self.provider:
            return "ШІ вимкнено"
        return f"{self.provider.name} · {self.provider.model}"

    def unavailable_reason(self) -> str:
        if self.provider is None:
            return "Провайдер ШІ не обрано (DEFAULT_LLM_PROVIDER=none)."
        return self.provider.config_hint()

    def complete(self, system: str, prompt: str, purpose: str, max_tokens: int = 2000, temperature: float = 0.5) -> str:
        if not self.available:
            raise LLMNotConfigured(self.unavailable_reason())
        err = ""
        resp: LLMResponse | None = None
        try:
            resp = self.provider.complete(system, prompt, max_tokens=max_tokens, temperature=temperature)
            return resp.text
        except LLMError as e:
            err = str(e)
            raise
        finally:
            self._log(purpose, len(system) + len(prompt), resp, err)

    def complete_json(self, system: str, prompt: str, purpose: str, max_tokens: int = 3000, temperature: float = 0.2):
        text = self.complete(system, prompt, purpose, max_tokens=max_tokens, temperature=temperature)
        return extract_json(text)

    def _log(self, purpose: str, prompt_chars: int, resp: LLMResponse | None, err: str) -> None:
        if self.session is None:
            return
        from app.db.models import LLMRunLog

        try:
            self.session.add(
                LLMRunLog(
                    project_id=self.project_id,
                    provider=self.provider.name if self.provider else "-",
                    model=self.provider.model if self.provider else "-",
                    purpose=purpose,
                    prompt_chars=prompt_chars,
                    response_chars=len(resp.text) if resp else 0,
                    duration_ms=resp.duration_ms if resp else 0,
                    success=not err,
                    error=err[:2000],
                )
            )
            self.session.commit()
        except Exception:  # noqa: BLE001 — лог не повинен ламати основну дію
            self.session.rollback()
