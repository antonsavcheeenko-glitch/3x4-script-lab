"""CRUD профілю стилю + (опційне) якісне збагачення через LLM."""
from __future__ import annotations

import json
import random

from sqlalchemy.orm import Session

from app.db.models import StyleCorpusDocument, StyleProfile
from app.llm import LLMService
from app.parsers import parse_bytes
from app.services.style_analyzer import build_profile, profile_to_prompt
from app.utils import text as T


def create_profile(session: Session, name: str) -> StyleProfile:
    p = StyleProfile(name=name.strip() or "Мій стиль", profile={}, summary="")
    session.add(p)
    session.commit()
    return p


def add_corpus_file(session: Session, profile: StyleProfile, filename: str, data: bytes) -> StyleCorpusDocument:
    parsed = parse_bytes(filename, data)
    return add_corpus_text(session, profile, filename, parsed.text)


def add_corpus_text(session: Session, profile: StyleProfile, filename: str, text: str) -> StyleCorpusDocument:
    d = StyleCorpusDocument(profile_id=profile.id, filename=filename, text=text, word_count=len(T.words(text)))
    session.add(d)
    session.commit()
    session.refresh(profile)
    return d


def rebuild_profile(session: Session, profile: StyleProfile) -> StyleProfile:
    texts = [d.text for d in profile.corpus]
    old_notes = (profile.profile or {}).get("llm_notes", {})
    data = build_profile(texts, name=profile.name)
    data["llm_notes"] = old_notes
    profile.profile = data
    profile.summary = data["summary"]
    session.commit()
    return profile


ENRICH_SYSTEM = (
    "Ти — редактор-стиліст, який аналізує авторський голос українського документального YouTube-автора. "
    "Будь конкретним: називай конструкції, прийоми, ритм. Не вигадуй того, чого немає в прикладах. "
    "Відповідай ЛИШЕ валідним JSON."
)

ENRICH_PROMPT = """Ось статистика корпусу автора:
{stats}

Ось фрагменти його сценаріїв:
<<<
{samples}
>>>

Поверни JSON:
{{"voice": "2–4 речення: як звучить автор (ритм, дистанція, іронія, гонзо-елементи)",
  "do": ["5–8 конкретних прийомів, які автор реально використовує"],
  "avoid": ["5–8 речей, яких автор уникає або які йому не властиві"],
  "intro": "як автор зазвичай починає",
  "transitions": "як автор переходить між блоками",
  "ending": "як автор завершує",
  "sample_sentences": ["3–5 дослівних характерних речень з фрагментів"]}}"""


def enrich_with_llm(session: Session, profile: StyleProfile, llm: LLMService) -> StyleProfile:
    if not profile.profile:
        rebuild_profile(session, profile)
    paragraphs: list[str] = []
    for d in profile.corpus:
        ps = T.split_paragraphs(d.text)
        paragraphs.extend(ps[:2] + ps[-1:])  # вступи й фінали особливо показові
        mid = ps[2:-1]
        random.Random(d.id).shuffle(mid)
        paragraphs.extend(mid[:4])
    samples = "\n\n".join(paragraphs)[:9000]
    stats = profile.summary
    data = llm.complete_json(ENRICH_SYSTEM, ENRICH_PROMPT.format(stats=stats, samples=samples), purpose="style_enrich")
    if not isinstance(data, dict):
        raise ValueError("Неочікувана відповідь моделі")
    prof = dict(profile.profile)
    prof["llm_notes"] = {k: data.get(k) for k in ("voice", "do", "avoid", "intro", "transitions", "ending", "sample_sentences") if data.get(k)}
    profile.profile = prof
    session.commit()
    return profile


def profile_json(profile: StyleProfile) -> str:
    return json.dumps(profile.profile, ensure_ascii=False, indent=2)


def profile_prompt(profile: StyleProfile | None) -> str:
    return profile_to_prompt(profile.profile if profile else {})
