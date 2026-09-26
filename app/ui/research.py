from __future__ import annotations

import streamlit as st
from sqlalchemy.orm import Session

from app.db.models import FACT_CATEGORIES, SOURCE_TYPES, ExtractedFact
from app.llm import LLMError
from app.parsers import ParseError
from app.services import research as svc
from app.ui.common import fact_source_line, flash, fmt_dt, get_llm, llm_notice, render_fact, require_project, source_label
from app.utils.text import truncate

CAT_BY_LABEL = {v: k for k, v in FACT_CATEGORIES.items()}


def render(session: Session) -> None:
    project = require_project(session)
    if not project:
        return
    llm = get_llm(session, project)
    st.title("Дослідження")
    st.caption(f"Проєкт: **{project.title}**")

    tab_docs, tab_facts = st.tabs([f"📄 Джерела ({len(project.documents)})", f"📌 Факти ({len(project.facts)})"])

    # ------------------------------------------------------------------ джерела
    with tab_docs:
        up_col, note_col = st.columns(2, gap="large")
        with up_col, st.container(border=True):
            st.subheader("Завантажити файли")
            files = st.file_uploader(
                "Формати: .md, .txt, .docx, .pdf", type=["md", "markdown", "txt", "docx", "pdf"], accept_multiple_files=True, key=f"up_{st.session_state.get('up_n', 0)}"
            )
            stypes = {"auto": "Визначити автоматично", **SOURCE_TYPES}
            stype = st.selectbox("Тип джерела", list(stypes), format_func=stypes.get)
            url = st.text_input("URL джерела (необов'язково)")
            auto_extract = st.checkbox("Одразу витягнути факти", value=True)
            if st.button("Імпортувати", type="primary", disabled=not files):
                ok = 0
                for f in files:
                    try:
                        doc = svc.ingest_document(session, project, f.name, f.getvalue(), None if stype == "auto" else stype, url)
                        ok += 1
                        if auto_extract:
                            with st.spinner(f"Витягую факти з «{f.name}»…"):
                                try:
                                    svc.extract_facts(session, project, doc, llm if llm.available else None)
                                except LLMError as e:
                                    st.warning(f"ШІ-витяг не вдався ({e}). Використано евристику.")
                                    svc.extract_facts(session, project, doc, None)
                    except ParseError as e:
                        st.error(f"{f.name}: {e}")
                if ok:
                    st.session_state["up_n"] = st.session_state.get("up_n", 0) + 1
                    flash(f"Імпортовано файлів: {ok}.")
                    st.rerun()
        with note_col, st.container(border=True):
            st.subheader("Додати нотатку / вставити текст")
            with st.form("note", clear_on_submit=True):
                ntitle = st.text_input("Заголовок")
                ntext = st.text_area("Текст", height=160)
                ntype = st.selectbox("Тип", list(SOURCE_TYPES), index=list(SOURCE_TYPES).index("note"), format_func=SOURCE_TYPES.get)
                if st.form_submit_button("Додати") and ntext.strip():
                    doc = svc.add_text_note(session, project, ntitle, ntext, ntype)
                    svc.extract_facts(session, project, doc, None)
                    flash("Нотатку додано.")
                    st.rerun()

        if not project.documents:
            st.info("Джерел ще немає.")
        for doc in sorted(project.documents, key=lambda d: d.created_at, reverse=True):
            with st.expander(f"**{doc.title or doc.filename}** · {source_label(doc.source_type)} · {doc.file_type.upper()} · фактів: {len(doc.facts)}"):
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.caption(f"Файл: {doc.filename} · {doc.char_count:,} символів · фрагментів: {len(doc.chunks)} · додано {fmt_dt(doc.created_at)}".replace(",", " "))
                    new_type = st.selectbox("Тип джерела", list(SOURCE_TYPES), index=list(SOURCE_TYPES).index(doc.source_type) if doc.source_type in SOURCE_TYPES else 0, format_func=SOURCE_TYPES.get, key=f"st_{doc.id}")
                    new_title = st.text_input("Назва", doc.title, key=f"ti_{doc.id}")
                    new_url = st.text_input("URL", doc.url, key=f"url_{doc.id}")
                    new_notes = st.text_area("Нотатки до джерела", doc.notes, key=f"nt_{doc.id}", height=70)
                    if st.button("Зберегти", key=f"sv_{doc.id}"):
                        doc.source_type, doc.title, doc.url, doc.notes = new_type, new_title, new_url, new_notes
                        session.commit()
                        flash("Збережено.")
                        st.rerun()
                with c2:
                    st.markdown("**Витяг фактів**")
                    b1, b2 = st.columns(2)
                    if b1.button("Евристика", key=f"xh_{doc.id}", help="Офлайн: речення з цифрами, датами, іменами, атрибуцією"):
                        created, _ = svc.extract_facts(session, project, doc, None)
                        flash(f"Евристика: {len(created)} фактів.")
                        st.rerun()
                    if b2.button("Через ШІ", key=f"xl_{doc.id}", disabled=not llm.available, help=None if llm.available else llm.unavailable_reason()):
                        with st.spinner("Модель читає джерело…"):
                            try:
                                created, _ = svc.extract_facts(session, project, doc, llm)
                                flash(f"ШІ: {len(created)} фактів.")
                                st.rerun()
                            except LLMError as e:
                                st.error(str(e))
                    st.caption("Повторний витяг замінює автоматичні факти, крім позначених ⭐, ручних і вже використаних у сценарії.")
                    st.markdown("---")
                    if st.button("🗑 Видалити джерело", key=f"del_{doc.id}"):
                        session.delete(doc)
                        session.commit()
                        flash("Джерело видалено.", "info")
                        st.rerun()
                st.markdown("**Фрагменти (чанки)**")
                for ch in doc.chunks[:50]:
                    page = f" · стор. {ch.page}" if ch.page else ""
                    st.markdown(f"<div class='muted'>#{ch.idx + 1} · символи {ch.start_char}–{ch.end_char}{page}</div>", unsafe_allow_html=True)
                    st.text(truncate(ch.text, 700))

    # ------------------------------------------------------------------ факти
    with tab_facts:
        llm_notice(llm, "Витяг фактів", "працює евристика (речення з цифрами, датами, іменами та атрибуцією). Перевіряйте й редагуйте результат.")
        facts = list(project.facts)
        f1, f2, f3, f4 = st.columns(4)
        docs = {d.id: (d.title or d.filename) for d in project.documents}
        doc_f = f1.multiselect("Джерело", list(docs), format_func=docs.get)
        cat_f = f2.multiselect("Категорія", list(FACT_CATEGORIES), format_func=FACT_CATEGORIES.get)
        used_f = f3.selectbox("У сценарії", ["усі", "так", "ні"])
        min_conf = f4.slider("Мін. впевненість", 0.0, 1.0, 0.0, 0.05)
        q = st.text_input("Пошук у фактах", placeholder="слово або фраза")
        view = [
            f for f in facts
            if (not doc_f or f.document_id in doc_f)
            and (not cat_f or f.category in cat_f)
            and (used_f == "усі" or (used_f == "так") == f.used_in_script)
            and f.confidence >= min_conf
            and (not q or q.lower() in (f.statement + " " + f.snippet).lower())
        ]
        st.caption(f"Показано {len(view)} з {len(facts)}. Редагуйте прямо в таблиці та натисніть «Зберегти зміни».")
        rows = [
            {
                "id": f.id,
                "⭐": f.starred,
                "Твердження": f.statement,
                "Категорія": FACT_CATEGORIES.get(f.category, f.category),
                "Впевненість": float(f.confidence),
                "У сценарії": f.used_in_script,
                "Джерело": fact_source_line(f),
                "Сніпет": truncate(f.snippet, 250),
                "Метод": f.method,
                "Видалити": False,
            }
            for f in view
        ]
        edited = st.data_editor(
            rows,
            key=f"facts_editor_{project.id}",
            width="stretch",
            hide_index=True,
            disabled=["id", "Джерело", "Сніпет", "Метод", "У сценарії"],
            column_config={
                "id": st.column_config.NumberColumn("F#", width="small"),
                "⭐": st.column_config.CheckboxColumn(width="small", help="Важливий факт — не видалятиметься при повторному витягу"),
                "Твердження": st.column_config.TextColumn(width="large"),
                "Категорія": st.column_config.SelectboxColumn(options=list(FACT_CATEGORIES.values())),
                "Впевненість": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.05, format="%.2f"),
                "У сценарії": st.column_config.CheckboxColumn(help="Оновлюється автоматично за маркерами [F#] у чернетці"),
                "Сніпет": st.column_config.TextColumn(width="medium"),
                "Видалити": st.column_config.CheckboxColumn(width="small"),
            },
        )
        if st.button("Зберегти зміни", type="primary", disabled=not rows):
            by_id = {f.id: f for f in view}
            deleted = 0
            for r in edited:
                f = by_id.get(r["id"])
                if not f:
                    continue
                if r["Видалити"]:
                    session.delete(f)
                    deleted += 1
                    continue
                f.statement = r["Твердження"]
                f.category = CAT_BY_LABEL.get(r["Категорія"], f.category)
                f.confidence = float(r["Впевненість"] or 0)
                f.starred = bool(r["⭐"])
            session.commit()
            flash(f"Збережено. Видалено: {deleted}.")
            st.rerun()

        with st.expander("➕ Додати факт вручну"):
            with st.form("manual_fact", clear_on_submit=True):
                stmt = st.text_area("Твердження", height=70)
                snip = st.text_area("Сніпет / цитата з джерела", height=70)
                dsel = st.selectbox("Джерело", [None, *docs], format_func=lambda x: "— без джерела —" if x is None else docs[x])
                c1, c2 = st.columns(2)
                cat = c1.selectbox("Категорія", list(FACT_CATEGORIES), format_func=FACT_CATEGORIES.get)
                conf = c2.slider("Впевненість", 0.0, 1.0, 0.7, 0.05)
                if st.form_submit_button("Додати") and stmt.strip():
                    session.add(ExtractedFact(project_id=project.id, document_id=dsel, statement=stmt.strip(), snippet=snip.strip(), category=cat, confidence=conf, method="manual"))
                    session.commit()
                    flash("Факт додано.")
                    st.rerun()

        with st.expander("🔍 Переглянути факти зі сніпетами"):
            for f in view[:100]:
                render_fact(f)
