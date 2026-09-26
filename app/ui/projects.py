from __future__ import annotations

import streamlit as st
from sqlalchemy.orm import Session

from app.services import projects as svc
from app.ui.common import current_project, flash, fmt_dt, goto


def _profile_options(session: Session) -> dict[int | None, str]:
    opts: dict[int | None, str] = {None: "— без профілю —"}
    for p in svc.list_profiles(session):
        opts[p.id] = p.name
    return opts


def render(session: Session) -> None:
    st.title("Проєкти")
    st.caption("Кожен проєкт — одне відео: джерела, факти, структура, чернетки, історія правок.")

    left, right = st.columns([3, 2], gap="large")

    with right:
        with st.container(border=True):
            st.subheader("Новий проєкт")
            with st.form("new_project", clear_on_submit=True):
                title = st.text_input("Назва *", placeholder="Напр.: Як працюють онлайн-казино")
                topic = st.text_area("Тема / ключове питання", height=80)
                desc = st.text_area("Короткий опис", height=80)
                opts = _profile_options(session)
                prof = st.selectbox("Профіль стилю", list(opts), format_func=opts.get)
                if st.form_submit_button("Створити", type="primary"):
                    try:
                        p = svc.create_project(session, title, topic, desc, prof)
                        st.session_state["project_id"] = p.id
                        flash(f"Проєкт «{p.title}» створено.")
                        goto("research")
                    except ValueError as e:
                        st.error(str(e))
        with st.container(border=True):
            st.subheader("Демо")
            st.caption(
                "Створює демо-профіль стилю (3 вигадані сценарії) і демо-проєкт з 3 джерелами, "
                "витягнутими фактами і шаблоном структури. Дані вигадані."
            )
            if st.button("Завантажити демо-проєкт"):
                with st.spinner("Імпортую демо…"):
                    p = svc.load_demo(session)
                st.session_state["project_id"] = p.id
                flash("Демо-проєкт створено. Перегляньте дослідження, структуру і згенеруйте чернетку.")
                goto("research")

    with left:
        items = svc.list_projects(session)
        if not items:
            st.info("Проєктів ще немає. Створіть новий або завантажте демо.")
        cur = current_project(session)
        for p in items:
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    mark = "✅ " if cur and cur.id == p.id else ""
                    st.markdown(f"### {mark}{p.title}")
                    if p.topic:
                        st.markdown(p.topic)
                    n_used = sum(1 for f in p.facts if f.used_in_script)
                    st.caption(
                        f"Джерел: {len(p.documents)} · фактів: {len(p.facts)} (у сценарії: {n_used}) · "
                        f"блоків: {len(p.outline)} · чернеток: {len(p.drafts)} · оновлено {fmt_dt(p.updated_at)}"
                    )
                with c2:
                    if st.button("Відкрити", key=f"open_{p.id}", type="primary" if not cur or cur.id != p.id else "secondary"):
                        st.session_state["project_id"] = p.id
                        goto("research")

    if cur:
        st.divider()
        st.subheader(f"Налаштування проєкту «{cur.title}»")
        with st.form("edit_project"):
            title = st.text_input("Назва", cur.title)
            topic = st.text_area("Тема / ключове питання", cur.topic, height=80)
            desc = st.text_area("Опис", cur.description, height=80)
            opts = _profile_options(session)
            keys = list(opts)
            prof = st.selectbox(
                "Профіль стилю для генерації",
                keys,
                index=keys.index(cur.style_profile_id) if cur.style_profile_id in keys else 0,
                format_func=opts.get,
            )
            if st.form_submit_button("Зберегти"):
                svc.update_project(session, cur, title=title.strip() or cur.title, topic=topic, description=desc, style_profile_id=prof)
                flash("Збережено.")
                st.rerun()
        with st.expander("🗑 Видалити проєкт"):
            st.warning("Будуть видалені всі джерела, факти, структура, чернетки та історія правок цього проєкту.")
            confirm = st.text_input("Введіть назву проєкту для підтвердження", key="del_confirm")
            if st.button("Видалити назавжди", disabled=confirm.strip() != cur.title):
                svc.delete_project(session, cur)
                st.session_state.pop("project_id", None)
                flash("Проєкт видалено.", "info")
                st.rerun()
