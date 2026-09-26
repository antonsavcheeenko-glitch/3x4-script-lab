"""Спільні UI-хелпери."""
from __future__ import annotations

import html
from datetime import datetime

import streamlit as st
from sqlalchemy.orm import Session

from app.config import load_llm_settings
from app.db.models import FACT_CATEGORIES, SOURCE_TYPES, ExtractedFact, Project
from app.llm import LLMService, build_provider
from app.services.research import confidence_label
from app.utils.text import truncate

NAV = [
    ("projects", "📁 Проєкти"),
    ("research", "🔎 Дослідження"),
    ("structure", "🧱 Структура"),
    ("script", "📝 Сценарій"),
    ("edit", "✂️ Редагування"),
    ("style", "🎙️ Профіль стилю"),
    ("settings", "⚙️ Налаштування"),
]
NAV_LABELS = dict(NAV)


def goto(page: str) -> None:
    st.session_state["_goto"] = page
    st.rerun()


def flash(msg: str, kind: str = "success") -> None:
    """Повідомлення, яке переживе st.rerun()."""
    st.session_state.setdefault("_flash", []).append((kind, msg))


def show_flash() -> None:
    for kind, msg in st.session_state.pop("_flash", []):
        getattr(st, kind, st.info)(msg)


def current_project(session: Session) -> Project | None:
    pid = st.session_state.get("project_id")
    if pid is None:
        return None
    p = session.get(Project, pid)
    if p is None:
        st.session_state.pop("project_id", None)
    return p


def require_project(session: Session) -> Project | None:
    p = current_project(session)
    if p is None:
        st.info("Спочатку оберіть або створіть проєкт.")
        if st.button("Перейти до проєктів", type="primary"):
            goto("projects")
    return p


def get_llm(session: Session, project: Project | None = None) -> LLMService:
    s = load_llm_settings()
    provider_name = st.session_state.get("llm_provider", s.provider)
    model = st.session_state.get("llm_model") or None
    provider = build_provider(provider_name, model, s) if provider_name != "none" else None
    return LLMService(provider, session=session, project_id=project.id if project else None)


def llm_notice(llm: LLMService, feature: str, fallback: str | None = None) -> None:
    if llm.available:
        return
    msg = f"**{feature}** найкраще працює з ШІ-провайдером. Зараз він недоступний: {llm.unavailable_reason()}"
    if fallback:
        msg += f"\n\nБез ШІ: {fallback}"
    st.warning(msg, icon="🤖")


def fmt_dt(dt: datetime | None) -> str:
    return dt.strftime("%d.%m.%Y %H:%M") if dt else "—"


def source_label(key: str) -> str:
    return SOURCE_TYPES.get(key, key)


def category_label(key: str) -> str:
    return FACT_CATEGORIES.get(key, key)


def fact_source_line(f: ExtractedFact) -> str:
    if not f.document:
        return "ручний запис"
    d = f.document
    loc = ""
    if f.chunk is not None:
        loc = f", фрагмент #{f.chunk.idx + 1}"
        if f.chunk.page:
            loc += f", стор. {f.chunk.page}"
    return f"{d.title or d.filename} · {source_label(d.source_type)}{loc}"


def render_fact(f: ExtractedFact, expanded: bool = False, prefix: str = "") -> None:
    """Картка факту з джерелом і сніпетом (для панелі трасування)."""
    conf = f"{confidence_label(f.confidence)} ({f.confidence:.2f})"
    title = f"{prefix}[F{f.id}] {truncate(f.statement, 110)}"
    with st.expander(title, expanded=expanded):
        st.markdown(f"**Твердження:** {html.escape(f.statement)}")
        st.caption(f"Джерело: {fact_source_line(f)} · категорія: {category_label(f.category)} · впевненість: {conf} · метод: {f.method}")
        if f.snippet:
            st.markdown(
                f"<div class='snippet'>{html.escape(f.snippet)}</div>",
                unsafe_allow_html=True,
            )
        if f.document and f.document.url:
            st.caption(f"URL: {f.document.url}")


CSS = """
<style>
.snippet {border-left: 3px solid #E0A526; padding: .4rem .8rem; margin: .3rem 0 .6rem;
          background: rgba(224,165,38,.07); font-size: .92rem; white-space: pre-wrap;}
.badge {display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: .78rem;
        margin-right: 6px; color: #fff;}
.sent {padding: .35rem .6rem; margin: .25rem 0; border-radius: 6px; border-left: 4px solid;}
.cite {color: #E0A526; font-weight: 600; font-size: .85em;}
.muted {opacity: .7; font-size: .9rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""
