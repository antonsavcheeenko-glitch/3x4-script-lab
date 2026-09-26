from __future__ import annotations

import time

import requests

from app.llm.base import LLMError, LLMNotConfigured, LLMProvider, LLMResponse


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str, timeout: int = 180):
        super().__init__(model, timeout)
        self.api_key = api_key

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def config_hint(self) -> str:
        return "Вкажіть ANTHROPIC_API_KEY у файлі .env"

    def complete(self, system, prompt, max_tokens=2000, temperature=0.5):
        if not self.is_configured():
            raise LLMNotConfigured(self.config_hint())
        t0 = time.time()
        try:
            r = requests.post(
                self.URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise LLMError(f"Anthropic: помилка мережі: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Anthropic: HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return LLMResponse(text=text, provider=self.name, model=self.model, duration_ms=int((time.time() - t0) * 1000))
