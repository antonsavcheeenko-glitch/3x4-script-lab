from __future__ import annotations

import time

import requests

from app.llm.base import LLMError, LLMNotConfigured, LLMProvider, LLMResponse


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", timeout: int = 180):
        super().__init__(model, timeout)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def config_hint(self) -> str:
        return "Вкажіть OPENAI_API_KEY у файлі .env"

    def complete(self, system, prompt, max_tokens=2000, temperature=0.5):
        if not self.is_configured():
            raise LLMNotConfigured(self.config_hint())
        t0 = time.time()
        payload = {
            "model": self.model,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        r = self._post(payload)
        # частина моделей (reasoning) не приймає temperature — повторюємо без неї
        if r.status_code == 400 and "temperature" in r.text:
            payload.pop("temperature")
            r = self._post(payload)
        if r.status_code != 200:
            raise LLMError(f"OpenAI: HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        text = data["choices"][0]["message"].get("content") or ""
        return LLMResponse(text=text, provider=self.name, model=self.model, duration_ms=int((time.time() - t0) * 1000))

    def _post(self, payload: dict) -> requests.Response:
        try:
            return requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise LLMError(f"OpenAI: помилка мережі: {e}") from e
