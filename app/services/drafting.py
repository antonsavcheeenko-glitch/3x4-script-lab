"""Генерація тексту сценарію з прив'язкою тверджень до фактів ([F12] → ExtractedFact.id=12)."""
from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.db.models import BLOCK_TYPES, SOURCE_TYPES, DraftBlock, ExtractedFact, OutlineBlock, Project, ScriptDraft
from app.llm import LLMService
from app.services.outline import relevant_facts
from app.services.style_service import profile_prompt
from app.utils import text as T

CITE_ANY_RE = re.compile(r"\[F\d+(?:\s*,\s*F?\d+)*\]")


def cited_ids(text: str) -> list[int]:
    ids: list[int] = []
    for m in CITE_ANY_RE.finditer(text):
        for n in re.findall(r"\d+", m.group(0)):
            if int(n) not in ids:
                ids.append(int(n))
    return ids


def strip_citations(text: str) -> str:
    return re.sub(r"\s*" + CITE_ANY_RE.pattern, "", text)


def fact_line(f: ExtractedFact) -> str:
    doc = f.document
    src = f"{doc.title or doc.filename} ({SOURCE_TYPES.get(doc.source_type, doc.source_type)})" if doc else "без джерела"
    return f"[F{f.id}] {f.statement} — джерело: {src}; впевненість джерела: {f.confidence:.1f}"


DRAFT_SYSTEM = """Ти — співавтор сценаріїв для українського документально-розслідувального YouTube-каналу.
Пишеш українською, від імені автора, у його стилі (профіль нижче).

Жорсткі правила:
1. Використовуй ЛИШЕ надані факти. Не вигадуй цифр, дат, імен, цитат.
2. Після кожного речення, що спирається на факт, став маркер джерела: [F12] (або [F3, F7]).
3. Не роби твердження сильнішим, ніж у факті. Якщо впевненість джерела < 0.5 — формулюй обережно («за даними…», «ймовірно»).
4. Інтерпретацію відокремлюй від факту словами-маркерами («це означає», «я бачу тут», «схоже»).
5. Жодних «у цьому відео», «варто зазначити», «давайте розберемося», «у сучасному світі».
6. Щільність фактів висока, речення короткі й середні, емоційність стримана.
7. Пиши тільки текст для озвучення: без заголовків, списків, ремарок у дужках.

{style}"""

BLOCK_PROMPT = """Тема відео: {topic}
Опис: {description}

Структура всього відео: {outline}

ЗАРАЗ ПИШЕМО БЛОК №{num}: «{title}» (тип: {btype}).
Завдання блоку: {notes}
Обсяг: ≈{words} слів.
{prev}
Доступні факти:
{facts}
{extra}
Напиши текст блоку."""


def _outline_str(project: Project) -> str:
    return " → ".join(b.title for b in sorted(project.outline, key=lambda x: x.position))


def _prev_context(draft: ScriptDraft | None, block: OutlineBlock) -> str:
    if draft is None:
        return ""
    prev = [
        db for db in draft.blocks if db.is_selected and db.position < block.position and db.text.strip()
    ]
    if not prev:
        return "Це перший блок — починай з сильного конкретного входу.\n"
    last = sorted(prev, key=lambda x: x.position)[-1]
    tail = T.split_paragraphs(strip_citations(last.text))[-1:]
    return "Попередній блок закінчився так (зроби плавний перехід, не повторюй):\n«" + T.truncate(" ".join(tail), 600) + "»\n"


def template_block_text(block: OutlineBlock, facts: list[ExtractedFact]) -> str:
    """Офлайн-чернетка без LLM: каркас блоку з фактами і маркерами джерел."""
    lines = [f"[ЧЕРНЕТКА БЕЗ ШІ — перепишіть власними словами] {block.notes}".strip()]
    if not facts:
        lines.append("Фактів для цього блоку не знайдено. Додайте факти в розділі «Дослідження» або закріпіть їх за блоком.")
    for f in facts:
        s = f.statement.rstrip(".") + "."
        if f.confidence < 0.5:
            s = "За наявними даними, " + s[0].lower() + s[1:]
        lines.append(f"{s} [F{f.id}]")
    return "\n\n".join(lines)


def get_or_create_draft(session: Session, project: Project, name: str | None = None) -> ScriptDraft:
    if project.drafts and name is None:
        return sorted(project.drafts, key=lambda d: d.created_at)[-1]
    d = ScriptDraft(
        project_id=project.id,
        name=name or f"Чернетка {len(project.drafts) + 1}",
        style_profile_id=project.style_profile_id,
    )
    session.add(d)
    session.commit()
    session.refresh(project)
    return d


def generate_block(
    session: Session,
    project: Project,
    draft: ScriptDraft,
    block: OutlineBlock,
    llm: LLMService | None,
    alternative: bool = False,
    extra_instruction: str = "",
) -> DraftBlock:
    facts = relevant_facts(project, block, k=8)
    variants = [db for db in draft.blocks if db.outline_block_id == block.id]
    if llm is not None and llm.available:
        style = profile_prompt(project.style_profile)
        extra = ""
        if alternative and variants:
            current = next((v for v in variants if v.is_selected), variants[-1])
            extra = (
                "\nЦе АЛЬТЕРНАТИВНА версія. Попередня версія була такою — зроби інакше (інший вхід, інша послідовність, інший ритм):\n«"
                + T.truncate(strip_citations(current.text), 1500)
                + "»\n"
            )
        if extra_instruction:
            extra += f"\nДодаткова вказівка автора: {extra_instruction}\n"
        prompt = BLOCK_PROMPT.format(
            topic=project.topic or project.title,
            description=project.description or "-",
            outline=_outline_str(project),
            num=block.position + 1,
            title=block.title,
            btype=BLOCK_TYPES.get(block.block_type, block.block_type),
            notes=block.notes or "-",
            words=block.target_words,
            prev=_prev_context(draft, block),
            facts="\n".join(fact_line(f) for f in facts) or "(фактів немає — пиши обережно, без конкретних цифр)",
            extra=extra,
        )
        text = llm.complete(DRAFT_SYSTEM.format(style=style), prompt, purpose="draft_block", max_tokens=2500, temperature=0.7 if alternative else 0.5)
        method = "llm"
    else:
        text = template_block_text(block, facts)
        method = "template"

    valid = {f.id for f in project.facts}
    used = [i for i in cited_ids(text) if i in valid]
    for v in variants:
        v.is_selected = False
    db = DraftBlock(
        draft_id=draft.id,
        outline_block_id=block.id,
        position=block.position,
        title=block.title,
        text=text.strip(),
        used_fact_ids=used,
        variant=len(variants) + 1,
        is_selected=True,
        method=method,
    )
    draft.blocks.append(db)
    session.flush()
    mark_used_facts(session, project)
    session.commit()
    session.refresh(draft)
    return db


def generate_full(session: Session, project: Project, draft: ScriptDraft, llm: LLMService | None, progress=None) -> list[DraftBlock]:
    out = []
    blocks = sorted(project.outline, key=lambda x: x.position)
    for i, b in enumerate(blocks):
        if progress:
            progress(i, len(blocks), b.title)
        out.append(generate_block(session, project, draft, b, llm))
    return out


def select_variant(session: Session, draft: ScriptDraft, db: DraftBlock) -> None:
    for v in draft.blocks:
        if v.outline_block_id == db.outline_block_id:
            v.is_selected = v.id == db.id
    session.commit()


def update_block_text(session: Session, project: Project, db: DraftBlock, text: str) -> None:
    db.text = text
    valid = {f.id for f in project.facts}
    db.used_fact_ids = [i for i in cited_ids(text) if i in valid]
    mark_used_facts(session, project)
    session.commit()


def mark_used_facts(session: Session, project: Project) -> None:
    """Синхронізує ExtractedFact.used_in_script з активними варіантами блоків усіх чернеток."""
    used: set[int] = set()
    for d in project.drafts:
        for db in d.blocks:
            if db.is_selected:
                used.update(db.used_fact_ids or [])
    for f in project.facts:
        f.used_in_script = f.id in used


def selected_blocks(draft: ScriptDraft) -> list[DraftBlock]:
    return sorted((b for b in draft.blocks if b.is_selected), key=lambda b: b.position)


def export_markdown(project: Project, draft: ScriptDraft, with_citations: bool = True) -> str:
    facts = {f.id: f for f in project.facts}
    lines = [f"# {project.title}", "", f"_{draft.name}_", ""]
    all_ids: list[int] = []
    for b in selected_blocks(draft):
        lines += [f"## {b.title}", "", b.text if with_citations else strip_citations(b.text), ""]
        all_ids += [i for i in b.used_fact_ids if i not in all_ids]
    if with_citations and all_ids:
        lines += ["---", "## Джерела тверджень", ""]
        for i in all_ids:
            f = facts.get(i)
            if f:
                src = f.document.title or f.document.filename if f.document else "—"
                lines.append(f"- **[F{i}]** {f.statement} — _{src}_  \n  > {T.truncate(f.snippet, 300)}")
    return "\n".join(lines)
