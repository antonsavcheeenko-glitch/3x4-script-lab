from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.database import make_engine  # noqa: E402
from app.llm import LLMService  # noqa: E402
from app.llm.base import LLMProvider, LLMResponse  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@pytest.fixture()
def session(tmp_path, monkeypatch):
    monkeypatch.setattr("app.services.research.UPLOAD_DIR", tmp_path / "uploads")
    engine = make_engine("sqlite://")
    s = sessionmaker(bind=engine, expire_on_commit=False)()
    yield s
    s.close()


class FakeProvider(LLMProvider):
    """Провайдер-заглушка: повертає відповідь залежно від purpose (за системним промптом)."""

    name = "fake"

    def __init__(self, responder):
        super().__init__("fake-model")
        self.responder = responder
        self.calls: list[tuple[str, str]] = []

    def is_configured(self):
        return True

    def config_hint(self):
        return ""

    def complete(self, system, prompt, max_tokens=2000, temperature=0.5):
        self.calls.append((system, prompt))
        out = self.responder(system, prompt)
        if not isinstance(out, str):
            out = json.dumps(out, ensure_ascii=False)
        return LLMResponse(text=out, provider="fake", model="fake-model")


@pytest.fixture()
def fake_llm(session):
    def make(responder):
        return LLMService(FakeProvider(responder), session=session)

    return make
