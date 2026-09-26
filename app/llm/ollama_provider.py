from __future__ import annotations

import time

import requests

from app.llm.base import LLMError, LLMNotConfigured, LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout: int = 300):
        super().__init__(model, timeout)
        self.base_url = (base_url or "").rstrip("/")

    def is_configured(self) -> bool:
        return bool(self.base_url)

    def config_hint(self) -> str:
        return "Вкажіть OLLAMA_BASE_URL (напр. http://localhost:11434) і запустіть `ollama serve`"

    def ping(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def complete(self, system, prompt, max_tokens=2000, temperature=0.5):
        if not self.is_configured():
            raise LLMNotConfigured(self.config_hint())
        t0 = time.time()
        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise LLMError(f"Ollama недоступна за адресою {self.base_url}: {e}") from e
        if r.status_code != 200:
            raise LLMError(f"Ollama: HTTP {r.status_code}: {r.text[:500]}")
        text = r.json().get("message", {}).get("content", "")
        return LLMResponse(text=text, provider=self.name, model=self.model, duration_ms=int((time.time() - t0) * 1000))
