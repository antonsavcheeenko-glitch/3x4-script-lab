from __future__ import annotations

import streamlit as st
from sqlalchemy.orm import Session

from app.db.models import BLOCK_TYPES
from app.llm import LLMError
from app.services import drafting
from app.services import outline as svc
from app.ui.common import flash, get_llm, goto, llm_notice, render_fact, require_project
from app.utils.text import truncate


def render(session: Session) -> None:
    project = require_project(session)
    if not project:
        return
    llm = get_llm(session, project)
    st.title("Структура")
    st.caption(f"Проєкт: **{project.title}** · блоків: {len(project.outline)}")

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("📋 Шаблон з 9 блоків", help="Хук → Контекст → Питання → Механізм → Актори → Український кут → Докази → Контраргумент → Висновок"):
            svc.apply_template(session, project, replace=False)
            flash("Додано шаблонні блоки й автоматично підібрано факти.")
            st.rerun()
        if c2.button("🤖 Згенерувати з дослідження", disabled=not llm.available, help=None if llm.available else llm.unavailable_reason()):
            with st.spinner("Модель будує структуру…"):
                try:
                    blocks = svc.generate_outline(session, project, llm, replace=True)
                    flash(f"Структуру згенеровано: {len(blocks)} блоків.")
                    st.rerun()
                except (LLMError, ValueError) as e:
                    st.error(str(e))
        if c3.button("🔗 Підібрати факти до блоків", help="Для блоків без закріплених фактів: BM25 за назвою/нотатками + категорія факту"):
            svc.auto_assign_facts(session, project)
            flash("Факти розподілено.")
            st.rerun()
        with c4.popover("🗑 Очистити структуру"):
            st.write("Видалити всі блоки структури? Чернетки залишаться.")
            if st.button("Так, видалити", key="clear_outline"):
                for b in list(project.outline):
                    session.delete(b)
                session.commit()
                st.rerun()
        llm_notice(llm, "Генерація структури", "використовуйте шаблон і редагуйте блоки вручну.")

    with st.expander("➕ Додати блок", expanded=not project.outline):
        with st.form("add_block", clear_on_submit=True):
            c1, c2 = st.columns([1, 2])
            btype = c1.selectbox("Тип", list(BLOCK_TYPES), format_func=BLOCK_TYPES.get)
            title = c2.text_input("Назва блоку", placeholder="Порожньо = назва типу")
            notes = st.text_area("Нотатки: що має бути в блоці", height=70)
            words = st.number_input("Цільовий обсяг, слів", 50, 1500, 250, 50)
            if st.form_submit_button("Додати блок", type="primary"):
                svc.add_block(session, project, btype, title, notes, int(words))
                st.rerun()

    facts = {f.id: f for f in project.facts}
    blocks = sorted(project.outline, key=lambda b: b.position)
    total = sum(b.target_words for b in blocks)
    if blocks:
        st.caption(f"Сумарний цільовий обсяг: ≈{total} слів (≈{total / 140:.0f} хв озвучки при 140 слів/хв).")

    for i, b in enumerate(blocks):
        with st.container(border=True):
            h1, h2, h3, h4, h5 = st.columns([6, 1, 1, 1, 2])
            h1.markdown(f"#### {i + 1}. {b.title}  \n<span class='muted'>{BLOCK_TYPES.get(b.block_type, b.block_type)} · ≈{b.target_words} слів · фактів: {len(b.fact_ids or [])}</span>", unsafe_allow_html=True)
            if h2.button("↑", key=f"up_{b.id}", disabled=i == 0):
                svc.move_block(session, project, b, -1)
                st.rerun()
            if h3.button("↓", key=f"dn_{b.id}", disabled=i == len(blocks) - 1):
                svc.move_block(session, project, b, 1)
                st.rerun()
            if h4.button("🗑", key=f"rm_{b.id}"):
                svc.delete_block(session, project, b)
                st.rerun()
            if h5.button("✍️ Чернетка блоку", key=f"gen_{b.id}", help="Згенерувати текст цього блоку в поточну чернетку"):
                draft = drafting.get_or_create_draft(session, project)
                with st.spinner("Пишу блок…"):
                    try:
                        drafting.generate_block(session, project, draft, b, llm)
                        st.session_state["draft_id"] = draft.id
                        flash(f"Блок «{b.title}» згенеровано{' (шаблонна чернетка без ШІ)' if not llm.available else ''}.")
                        goto("script")
                    except LLMError as e:
                        st.error(str(e))

            with st.expander("Редагувати блок"):
                with st.form(f"blk_{b.id}"):
                    c1, c2, c3 = st.columns([1, 2, 1])
                    types = list(BLOCK_TYPES)
                    btype = c1.selectbox("Тип", types, index=types.index(b.block_type) if b.block_type in types else len(types) - 1, format_func=BLOCK_TYPES.get)
                    title = c2.text_input("Назва", b.title)
                    words = c3.number_input("Слів", 50, 1500, int(b.target_words), 50)
                    notes = st.text_area("Нотатки", b.notes, height=80)
                    fact_opts = list(facts)
                    chosen = st.multiselect(
                        "Закріплені факти (пріоритетні для генерації)",
                        fact_opts,
                        default=[x for x in (b.fact_ids or []) if x in facts],
                        format_func=lambda x: f"F{x}: {truncate(facts[x].statement, 90)}",
                    )
                    if st.form_submit_button("Зберегти"):
                        b.block_type, b.title, b.target_words, b.notes, b.fact_ids = btype, title.strip() or b.title, int(words), notes, chosen
                        session.commit()
                        st.rerun()
            if b.notes:
                st.markdown(f"<div class='muted'>{b.notes}</div>", unsafe_allow_html=True)
            if b.fact_ids:
                with st.expander(f"Факти блоку ({len(b.fact_ids)})"):
                    for fid in b.fact_ids:
                        if fid in facts:
                            render_fact(facts[fid])
