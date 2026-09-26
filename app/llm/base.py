"""Інтерфейс LLM-провайдера."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(Exception):
    pass


class LLMNotConfigured(LLMError):
    pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    duration_ms: int = 0


class LLMProvider(ABC):
    name: str = "base"

    def __init__(self, model: str, timeout: int = 180):
        self.model = model
        self.timeout = timeout

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def config_hint(self) -> str:
        """Що треба налаштувати, якщо провайдер недоступний."""

    @abstractmethod
    def complete(self, system: str, prompt: str, max_tokens: int = 2000, temperature: float = 0.5) -> LLMResponse: ...


def extract_json(text: str):
    """Дістає JSON з відповіді моделі (у тому числі з ```json блоків)."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    candidates = [m.group(1)] if m else []
    candidates.append(text)
    for cand in candidates:
        cand = cand.strip()
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            s, e = cand.find(opener), cand.rfind(closer)
            if s != -1 and e > s:
                try:
                    return json.loads(cand[s : e + 1])
                except json.JSONDecodeError:
                    continue
    raise LLMError("Модель повернула відповідь, яку не вдалося розібрати як JSON.")
