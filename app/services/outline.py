"""Структура сценарію: блоки, шаблон, підбір фактів, генерація структури з дослідження."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import BLOCK_TYPES, ExtractedFact, OutlineBlock, Project
from app.llm import LLMService
from app.services.retrieval import BM25

DEFAULT_TEMPLATE = [
    ("hook", "Хук", "Сильний конкретний вхід: подія, цифра або сцена. Без «у цьому відео».", 120),
    ("context", "Контекст", "Що відбувається і чому це важливо зараз.", 250),
    ("key_question", "Ключове питання", "Одне питання, на яке відповідає відео.", 80),
    ("mechanism", "Механізм", "Як саме це працює: крок за кроком.", 350),
    ("actors", "Актори", "Хто заробляє, хто регулює, хто втрачає.", 300),
    ("ukrainian_angle", "Український кут", "Як це виглядає в Україні: закони, кейси, цифри.", 300),
    ("evidence", "Докази", "Найсильніші дані й документи.", 300),
    ("counterpoint", "Контраргумент", "Найкращий аргумент іншої сторони і його межі.", 200),
    ("conclusion", "Висновок", "Що ми знаємо напевно, що ні, і що з цього випливає.", 180),
]


def add_block(session: Session, project: Project, block_type: str, title: str = "", notes: str = "", target_words: int = 250) -> OutlineBlock:
    pos = max((b.position for b in project.outline), default=-1) + 1
    b = OutlineBlock(
        project_id=project.id,
        position=pos,
        block_type=block_type if block_type in BLOCK_TYPES else "custom",
        title=title.strip() or BLOCK_TYPES.get(block_type, "Блок"),
        notes=notes,
        target_words=target_words,
        fact_ids=[],
    )
    session.add(b)
    session.commit()
    session.refresh(project)
    return b


def apply_template(session: Session, project: Project, replace: bool = False) -> None:
    if replace:
        for b in list(project.outline):
            session.delete(b)
        session.commit()
        session.refresh(project)
    for btype, title, notes, words in DEFAULT_TEMPLATE:
        add_block(session, project, btype, title, notes, words)
    auto_assign_facts(session, project)


def normalize_positions(session: Session, project: Project) -> None:
    for i, b in enumerate(sorted(project.outline, key=lambda x: x.position)):
        b.position = i
    session.commit()


def move_block(session: Session, project: Project, block: OutlineBlock, delta: int) -> None:
    blocks = sorted(project.outline, key=lambda x: x.position)
    i = blocks.index(block)
    j = i + delta
    if 0 <= j < len(blocks):
        blocks[i], blocks[j] = blocks[j], blocks[i]
        for k, b in enumerate(blocks):
            b.position = k
        session.commit()


def delete_block(session: Session, project: Project, block: OutlineBlock) -> None:
    session.delete(block)
    session.commit()
    session.refresh(project)
    normalize_positions(session, project)


def block_query(project: Project, block: OutlineBlock, with_topic: bool = True) -> str:
    q = f"{block.title} {block.notes} {BLOCK_TYPES.get(block.block_type, '')}"
    return f"{q} {project.topic}" if with_topic else q


def relevant_facts(project: Project, block: OutlineBlock, k: int = 8) -> list[ExtractedFact]:
    """Закріплені за блоком факти + найрелевантніші за BM25."""
    facts = list(project.facts)
    if not facts:
        return []
    by_id = {f.id: f for f in facts}
    chosen = [by_id[i] for i in (block.fact_ids or []) if i in by_id]
    if len(chosen) >= k:
        return chosen[:k]
    bm = BM25([f"{f.statement} {f.snippet}" for f in facts])
    for idx, _score in bm.top(block_query(project, block), k=k * 2, min_score=0.1):
        f = facts[idx]
        if f not in chosen:
            chosen.append(f)
        if len(chosen) >= k:
            break
    return chosen


# Спорідненість категорії факту з типом блоку (бонус до BM25 при автопідборі)
CATEGORY_AFFINITY = {
    "hook": {"quote": 1.5, "number": 1.0, "event": 1.0},
    "context": {"event": 1.5, "context": 1.5, "number": 0.5},
    "key_question": {"context": 0.5},
    "mechanism": {"mechanism": 2.0, "number": 0.8, "context": 0.5},
    "actors": {"actor": 2.0, "quote": 1.0},
    "ukrainian_angle": {"legal": 2.0, "event": 0.5},
    "evidence": {"number": 1.5, "quote": 1.0, "legal": 0.5},
    "counterpoint": {"context": 0.5, "legal": 0.5},
    "conclusion": {},
    "custom": {},
}


def auto_assign_facts(session: Session, project: Project, per_block: int = 4) -> None:
    """Розподіляє факти по блоках без закріплених фактів.

    Оцінка = BM25(назва+нотатки блоку) + бонус за спорідненість категорії факту з типом блоку.
    Кожен факт потрапляє максимум в один блок.
    """
    facts = list(project.facts)
    if not facts or not project.outline:
        return
    bm = BM25([f"{f.statement} {f.snippet}" for f in facts])
    used: set[int] = set()
    for b in project.outline:
        used.update(b.fact_ids or [])
    for b in sorted(project.outline, key=lambda x: x.position):
        if b.fact_ids or b.block_type in ("key_question", "conclusion"):
            continue
        aff = CATEGORY_AFFINITY.get(b.block_type, {})
        scores = bm.score(block_query(project, b, with_topic=False))
        ranked = sorted(
            ((scores[i] + aff.get(f.category, 0) + f.confidence * 0.5, f) for i, f in enumerate(facts) if f.id not in used),
            key=lambda x: -x[0],
        )
        picks = [f.id for sc, f in ranked if sc >= 1.0][:per_block]
        used.update(picks)
        b.fact_ids = picks
    session.commit()


OUTLINE_SYSTEM = (
    "Ти — шеф-редактор українського документального YouTube-каналу (розслідування, аналітика). "
    "Будуєш структуру відео на основі фактів. Не вигадуй фактів. Відповідай ЛИШЕ валідним JSON."
)

OUTLINE_PROMPT = """Тема: {topic}
Опис: {description}

Факти з дослідження (id: твердження):
{facts}

Допустимі типи блоків: {types}.

Склади структуру з 7–11 блоків. Поверни JSON-масив:
[{{"block_type": "один з типів", "title": "конкретна назва блоку (не загальна)",
   "notes": "1–3 речення: що саме розповідаємо і який ефект", "target_words": 150-450,
   "fact_ids": [id фактів, що мають увійти]}}]"""


def generate_outline(session: Session, project: Project, llm: LLMService, replace: bool = True) -> list[OutlineBlock]:
    facts = list(project.facts)[:120]
    fact_lines = "\n".join(f"{f.id}: {f.statement}" for f in facts) or "(фактів ще немає)"
    data = llm.complete_json(
        OUTLINE_SYSTEM,
        OUTLINE_PROMPT.format(
            topic=project.topic or project.title,
            description=project.description or "-",
            facts=fact_lines,
            types=", ".join(k for k in BLOCK_TYPES),
        ),
        purpose="generate_outline",
    )
    if isinstance(data, dict):
        data = data.get("blocks", [])
    if not isinstance(data, list) or not data:
        raise ValueError("Модель не повернула структуру")
    if replace:
        for b in list(project.outline):
            session.delete(b)
        session.commit()
        session.refresh(project)
    valid_ids = {f.id for f in facts}
    created = []
    for item in data:
        if not isinstance(item, dict):
            continue
        b = add_block(
            session,
            project,
            str(item.get("block_type", "custom")),
            str(item.get("title", "")),
            str(item.get("notes", "")),
            int(item.get("target_words") or 250) if str(item.get("target_words", "")).isdigit() else 250,
        )
        b.fact_ids = [int(i) for i in item.get("fact_ids", []) if str(i).isdigit() and int(i) in valid_ids]
        created.append(b)
    session.commit()
    return created
