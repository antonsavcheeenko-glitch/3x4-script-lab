"""3x4 Script Lab — точка входу Streamlit.

Запуск: streamlit run main.py
"""
from __future__ import annotations

import streamlit as st

from app.db.database import get_session
from app.ui import common, edit, projects, research, script, settings, structure, style

st.set_page_config(page_title="3x4 Script Lab", page_icon="🎬", layout="wide")
st.markdown(common.CSS, unsafe_allow_html=True)

PAGES = {
    "projects": projects.render,
    "research": research.render,
    "structure": structure.render,
    "script": script.render,
    "edit": edit.render,
    "style": style.render,
    "settings": settings.render,
}

if "_goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_goto")
st.session_state.setdefault("nav", "projects")

session = get_session()
try:
    with st.sidebar:
        st.markdown("## 🎬 3x4 Script Lab")
        project = common.current_project(session)
        if project:
            st.markdown(f"**Проєкт:** {project.title}")
            prof = project.style_profile
            st.caption(f"Профіль стилю: {prof.name if prof else '— не обрано'}")
        else:
            st.caption("Проєкт не обрано")
        st.radio("Навігація", [k for k, _ in common.NAV], format_func=common.NAV_LABELS.get, key="nav", label_visibility="collapsed")
        st.divider()
        llm = common.get_llm(session)
        if llm.available:
            st.caption(f"🟢 ШІ: {llm.label}")
        else:
            st.caption("⚪ ШІ не налаштовано — працюють офлайн-функції")

    common.show_flash()
    PAGES[st.session_state["nav"]](session)
finally:
    session.close()
