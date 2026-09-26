from __future__ import annotations

import html
import re

import streamlit as st
from sqlalchemy.orm import Session

from app.db.models import DraftBlock, ScriptDraft
from app.llm import LLMError
from app.services import drafting as svc
from app.services.fact_check import classify_sentence
from app.ui.common import flash, get_llm, goto, llm_notice, render_fact, require_project
from app.utils.text import split_sentences, words


def cited_html(text: str, facts: dict) -> str:
    """Текст з підсвіченими маркерами [F#] і підказкою-твердженням при наведенні."""
    out = []
    for para in text.split("\n\n"):
        esc = html.escape(para)

        def repl(m: re.Match) -> str:
            ids = [int(x) for x in re.findall(r"\d+", m.group(0))]
            tip = " | ".join(f"F{i}: {facts[i].statement}" if i in facts else f"F{i}: (факт не знайдено)" for i in ids)
            return f"<span class='cite' title='{html.escape(tip, quote=True)}'>{m.group(0)}</span>"

        esc = re.sub(r"\[F\d+(?:\s*,\s*F?\d+)*\]", repl, esc)
        out.append(f"<p>{esc}</p>")
    return "\n".join(out)


def uncited_factual(text: str) -> list[str]:
    """Речення, що виглядають як фактичні, але не мають маркера джерела."""
    res = []
    for s in split_sentences(text):
        if re.search(r"\[F\d+", s) or s.startswith("[ЧЕРНЕТКА"):
            continue
        cls, _ = classify_sentence(s)
        if cls == "fact":
            res.append(s)
    return res


def _draft_selector(session: Session, project) -> ScriptDraft | None:
    drafts = sorted(project.drafts, key=lambda d: d.created_at)
    c1, c2, c3 = st.columns([3, 1, 1])
    if not drafts:
        c1.info("Чернеток ще немає.")
        draft = None
    else:
        ids = [d.id for d in drafts]
        cur = st.session_state.get("draft_id")
        idx = ids.index(cur) if cur in ids else len(ids) - 1
        names = {d.id: f"{d.name} · блоків: {len(svc.selected_blocks(d))}" for d in drafts}
        did = c1.selectbox("Чернетка", ids, index=idx, format_func=names.get)
        st.session_state["draft_id"] = did
        draft = next(d for d in drafts if d.id == did)
    if c2.button("➕ Нова чернетка"):
        d = svc.get_or_create_draft(session, project, name=f"Чернетка {len(drafts) + 1}")
        st.session_state["draft_id"] = d.id
        st.rerun()
    if draft is not None:
        with c3.popover("🗑 Видалити"):
            st.write(f"Видалити «{draft.name}»?")
            if st.button("Так", key="del_draft"):
                session.delete(draft)
                session.commit()
                st.session_state.pop("draft_id", None)
                svc.mark_used_facts(session, project)
                session.commit()
                st.rerun()
    return draft


def render(session: Session) -> None:
    project = require_project(session)
    if not project:
        return
    llm = get_llm(session, project)
    st.title("Сценарій")
    st.caption(
        f"Проєкт: **{project.title}** · профіль стилю: **{project.style_profile.name if project.style_profile else 'не обрано'}**"
    )
    llm_notice(llm, "Генерація сценарію", "створюється шаблонна чернетка-каркас: факти блоку з маркерами джерел, яку ви переписуєте вручну.")
    if not project.style_profile:
        st.info("Профіль стилю не обрано — генерація буде в загальному документальному стилі. Оберіть профіль у «Проєкти → Налаштування проєкту».")

    if not project.outline:
        st.warning("Спочатку створіть структуру.")
        if st.button("Перейти до структури"):
            goto("structure")
        return

    draft = _draft_selector(session, project)
    facts = {f.id: f for f in project.facts}

    a1, a2, a3, a4 = st.columns(4)
    if a1.button("🚀 Згенерувати повну чернетку", type="primary"):
        if draft is None:
            draft = svc.get_or_create_draft(session, project)
            st.session_state["draft_id"] = draft.id
        bar = st.progress(0.0, "Починаю…")
        try:
            svc.generate_full(session, project, draft, llm, progress=lambda i, n, t: bar.progress(i / n, f"Блок {i + 1}/{n}: {t}"))
            bar.progress(1.0, "Готово")
            flash("Повну чернетку згенеровано.")
            st.rerun()
        except LLMError as e:
            st.error(f"Генерацію перервано: {e}")
    if draft is not None:
        blocks = svc.selected_blocks(draft)
        n_words = sum(len(words(svc.strip_citations(b.text))) for b in blocks)
        a2.metric("Слів у чернетці", n_words, help="Без маркерів джерел")
        a3.metric("≈ хвилин озвучки", f"{n_words / 140:.1f}")
        used = {i for b in blocks for i in b.used_fact_ids}
        a4.metric("Фактів використано", f"{len(used)} / {len(facts)}")
        with st.expander("⬇️ Експорт"):
            with_c = st.checkbox("З маркерами і списком джерел", value=True)
            md = svc.export_markdown(project, draft, with_citations=with_c)
            st.download_button("Завантажити .md", md, file_name=f"{project.title[:40]}_{draft.name}.md", mime="text/markdown")
            st.download_button("Завантажити .txt (чистий текст для озвучки)", svc.export_markdown(project, draft, with_citations=False), file_name=f"{project.title[:40]}.txt")

    st.divider()
    outline = sorted(project.outline, key=lambda b: b.position)
    for ob in outline:
        variants = sorted([v for v in (draft.blocks if draft else []) if v.outline_block_id == ob.id], key=lambda v: v.variant)
        current = next((v for v in variants if v.is_selected), variants[-1] if variants else None)
        with st.container(border=True):
            st.markdown(f"### {ob.position + 1}. {ob.title}")
            left, right = st.columns([3, 2], gap="large")
            with left:
                instr = st.text_input("Додаткова вказівка (необов'язково)", key=f"ins_{ob.id}", placeholder="напр.: почни зі сцени, менше цифр, більше про регулятора")
                b1, b2, b3 = st.columns(3)
                gen = b1.button("✍️ Згенерувати" if not current else "🔄 Перегенерувати", key=f"g_{ob.id}")
                alt = b2.button("🔀 Альтернатива", key=f"a_{ob.id}", disabled=current is None, help="Нова версія поруч зі старою — старі варіанти зберігаються")
                if gen or alt:
                    if draft is None:
                        draft = svc.get_or_create_draft(session, project)
                        st.session_state["draft_id"] = draft.id
                    with st.spinner("Пишу…"):
                        try:
                            svc.generate_block(session, project, draft, ob, llm, alternative=alt, extra_instruction=instr)
                            st.rerun()
                        except LLMError as e:
                            st.error(str(e))
                if current is None:
                    st.caption("Тексту ще немає.")
                    continue
                if len(variants) > 1:
                    vid = b3.selectbox(
                        "Варіант", [v.id for v in variants], index=[v.id for v in variants].index(current.id),
                        format_func=lambda x: f"Варіант {next(v.variant for v in variants if v.id == x)}", key=f"var_{ob.id}",
                    )
                    if vid != current.id:
                        svc.select_variant(session, draft, next(v for v in variants if v.id == vid))
                        svc.mark_used_facts(session, project)
                        session.commit()
                        st.rerun()
                tag = {"llm": "ШІ", "template": "шаблон без ШІ", "manual": "вручну"}.get(current.method, current.method)
                st.caption(f"Варіант {current.variant} · {tag} · {len(words(svc.strip_citations(current.text)))} слів")
                new_text = st.text_area("Текст блоку", current.text, height=280, key=f"txt_{current.id}", label_visibility="collapsed")
                s1, s2 = st.columns(2)
                if s1.button("💾 Зберегти", key=f"sv_{current.id}", disabled=new_text == current.text):
                    svc.update_block_text(session, project, current, new_text)
                    flash("Збережено.")
                    st.rerun()
                if s2.button("✂️ У редактор", key=f"ed_{current.id}"):
                    st.session_state["edit_block_id"] = current.id
                    goto("edit")
                with st.expander("👁 Перегляд з підсвіченими джерелами"):
                    st.markdown(cited_html(current.text, facts), unsafe_allow_html=True)
            with right:
                if current is None:
                    continue
                st.markdown("**🔗 Джерела тверджень блоку**")
                if not current.used_fact_ids:
                    st.caption("У тексті немає маркерів [F#].")
                for fid in current.used_fact_ids:
                    if fid in facts:
                        render_fact(facts[fid])
                    else:
                        st.caption(f"[F{fid}] — факт видалено")
                unc = uncited_factual(current.text)
                if unc:
                    with st.expander(f"⚠️ Фактичні речення без джерела ({len(unc)})"):
                        for s in unc:
                            st.markdown(f"- {html.escape(s)}")
                        st.caption("Додайте маркер [F#] або перевірте твердження у «Редагування → Факт чи інтерпретація».")

    # блоки, чий outline-блок видалено
    if draft is not None:
        ids = {b.id for b in outline}
        orphans: list[DraftBlock] = [b for b in draft.blocks if b.is_selected and b.outline_block_id not in ids]
        if orphans:
            with st.expander(f"Блоки поза поточною структурою ({len(orphans)})"):
                for b in orphans:
                    st.markdown(f"**{b.title}**")
                    st.text(b.text)
