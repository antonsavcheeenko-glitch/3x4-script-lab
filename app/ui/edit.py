from __future__ import annotations

import html

import streamlit as st
from sqlalchemy.orm import Session

from app.db.models import DraftBlock
from app.llm import LLMError
from app.services import drafting
from app.services import editing as ed
from app.services import fact_check as fc
from app.services import style_check as sc
from app.ui.common import flash, fmt_dt, get_llm, llm_notice, render_fact, require_project
from app.utils.text import split_paragraphs, truncate

SEV = {"high": "🔴", "medium": "🟠", "low": "🔵"}


def _source_picker(session: Session, project) -> tuple[str, DraftBlock | None, int | None]:
    """Повертає (текст, блок-джерело, індекс абзацу або None=весь блок)."""
    draft_blocks: list[DraftBlock] = []
    for d in sorted(project.drafts, key=lambda d: d.created_at):
        draft_blocks += drafting.selected_blocks(d)
    mode = st.radio("Що редагуємо", ["Блок чернетки", "Власний текст"], horizontal=True, index=0 if draft_blocks else 1)
    if mode == "Власний текст" or not draft_blocks:
        if not draft_blocks and mode == "Блок чернетки":
            st.info("Чернеток ще немає — вставте текст вручну.")
        txt = st.text_area("Текст", key="free_text", height=220, placeholder="Вставте абзац або блок…")
        return txt, None, None
    labels = {b.id: f"{b.draft.name} → {b.position + 1}. {b.title}" for b in draft_blocks}
    ids = list(labels)
    pre = st.session_state.get("edit_block_id")
    bid = st.selectbox("Блок", ids, index=ids.index(pre) if pre in ids else 0, format_func=labels.get)
    st.session_state["edit_block_id"] = bid
    block = next(b for b in draft_blocks if b.id == bid)
    paras = split_paragraphs(block.text)
    opts = [None, *range(len(paras))]
    pidx = st.selectbox(
        "Фрагмент", opts, format_func=lambda i: "Весь блок" if i is None else f"Абзац {i + 1}: {truncate(paras[i], 80)}", key=f"para_{bid}"
    )
    text = block.text if pidx is None else paras[pidx]
    with st.expander("Обраний текст", expanded=False):
        st.text(text)
    return text, block, pidx


def _apply(session: Session, project, block: DraftBlock, pidx: int | None, new_text: str) -> None:
    if pidx is None:
        full = new_text
    else:
        paras = split_paragraphs(block.text)
        paras[pidx] = new_text
        full = "\n\n".join(paras)
    drafting.update_block_text(session, project, block, full)


def render(session: Session) -> None:
    project = require_project(session)
    if not project:
        return
    llm = get_llm(session, project)
    st.title("Редагування")
    text, block, pidx = _source_picker(session, project)
    profile = project.style_profile.profile if project.style_profile else None

    t_tools, t_style, t_fact, t_hist = st.tabs(["🛠 Інструменти", "🎙 Чи звучить як я?", "⚖️ Факт чи інтерпретація", "🕘 Історія правок"])

    # ------------------------------------------------------------ інструменти
    with t_tools:
        llm_notice(llm, "Інструменти редагування", "доступні лише «Прибрати кліше» і «Менш емоційно» (евристичні заміни).")
        ops = list(ed.OPERATIONS)
        op = st.selectbox("Операція", ops, format_func=lambda k: ed.OPERATIONS[k]["label"])
        st.caption(ed.OPERATIONS[op]["instruction"])
        text_b = ""
        if op == "transition":
            text_b = st.text_area("Початок наступного блоку (Б)", height=120, help="Текст А — обраний вище фрагмент")
        can = bool(text.strip()) and (llm.available or op in ed.OFFLINE)
        if st.button("Виконати", type="primary", disabled=not can):
            with st.spinner("Редагую…"):
                try:
                    out, method, op_id = ed.run_operation(session, project, op, text, llm, block.id if block else None, text_b)
                    st.session_state["edit_result"] = {"op": op, "out": out, "method": method, "src": text, "op_id": op_id}
                except (LLMError, RuntimeError) as e:
                    st.error(str(e))
        res = st.session_state.get("edit_result")
        if res and res["src"] == text:
            c1, c2 = st.columns(2)
            c1.markdown("**Було**")
            c1.text(res["src"] if res["op"] != "transition" else truncate(res["src"], 600))
            c2.markdown(f"**Стало** · {ed.OPERATIONS[res['op']]['label']} · {'ШІ' if res['method'] == 'llm' else 'евристика'}")
            new = c2.text_area("Результат", res["out"], height=260, key="edit_out", label_visibility="collapsed")
            if block is not None:
                if res["op"] == "transition":
                    if st.button("Додати перехід у кінець блоку"):
                        drafting.update_block_text(session, project, block, block.text.rstrip() + "\n\n" + new.strip())
                        ed.mark_applied(session, res["op_id"])
                        st.session_state.pop("edit_result", None)
                        flash("Перехід додано.")
                        st.rerun()
                elif st.button("✅ Застосувати до блоку"):
                    _apply(session, project, block, pidx, new)
                    ed.mark_applied(session, res["op_id"])
                    st.session_state.pop("edit_result", None)
                    flash("Зміни застосовано.")
                    st.rerun()

    # ------------------------------------------------------------ стиль
    with t_style:
        if not profile:
            st.info("Профіль стилю для проєкту не обрано — порівнюю з еталоном «стриманий документальний стиль». Для точності створіть профіль.")
        if st.button("Перевірити", type="primary", disabled=not text.strip(), key="style_check"):
            st.session_state["style_res"] = (text, sc.check_style(text, profile))
        pair = st.session_state.get("style_res")
        if pair and pair[0] == text:
            r: sc.StyleCheckResult = pair[1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Схожість", f"{r.score}/100")
            c2.metric("Сер. речення, слів", r.metrics["sentence_len"]["mean"], delta=round(r.metrics["sentence_len"]["mean"] - r.reference["sentence_len"]["mean"], 1), delta_color="off")
            c3.metric("Абстрактні слова", f"{r.metrics['abstract_ratio'] * 100:.0f}%", delta=f"{(r.metrics['abstract_ratio'] - r.reference['abstract_ratio']) * 100:+.0f} п.п.", delta_color="inverse")
            c4.metric("Цифри / 1000 слів", r.metrics["numbers_per_1k"], delta=round(r.metrics["numbers_per_1k"] - r.reference["numbers_per_1k"], 1))
            st.subheader(r.label)
            st.caption("Оцінка = 100 мінус штрафи за відхилення нижче. Це евристика на метриках тексту, а не «розуміння» стилю.")
            if not r.diagnostics:
                st.success("Суттєвих відхилень від профілю не знайдено.")
            for d in r.diagnostics:
                pen = f" (−{d.penalty})" if d.penalty else ""
                st.markdown(f"{SEV[d.severity]} {d.message}{pen}")
                for ex in d.examples:
                    if ex:
                        st.markdown(f"<div class='snippet'>{html.escape(ex)}</div>", unsafe_allow_html=True)
            st.divider()
            llm_notice(llm, "Переписати ближче до стилю")
            if st.button("🤖 Запропонувати переписаний варіант", disabled=not llm.available):
                with st.spinner("Переписую…"):
                    try:
                        r.rewrite = sc.suggest_rewrite(text, r, profile, llm)
                    except LLMError as e:
                        st.error(str(e))
            if r.rewrite:
                new = st.text_area("Варіант ближче до стилю", r.rewrite, height=240, key="style_rw")
                rc = sc.check_style(new, profile)
                st.caption(f"Оцінка переписаного варіанта: {rc.score}/100 ({rc.label})")
                if block is not None and st.button("✅ Застосувати до блоку", key="apply_rw"):
                    _apply(session, project, block, pidx, new)
                    flash("Застосовано.")
                    st.session_state.pop("style_res", None)
                    st.rerun()

    # ------------------------------------------------------------ факт / інтерпретація
    with t_fact:
        facts = list(project.facts)
        mode = st.radio("Режим", ["Евристика (офлайн)", "ШІ"], horizontal=True, index=1 if llm.available else 0)
        if mode == "ШІ":
            llm_notice(llm, "ШІ-аналіз тверджень")
        run = st.button("Аналізувати", type="primary", disabled=not text.strip() or (mode == "ШІ" and not llm.available), key="fc_run")
        if run:
            with st.spinner("Аналізую речення…"):
                try:
                    items = fc.analyze_with_llm(text, facts, llm) if mode == "ШІ" else fc.analyze_paragraph(text, facts)
                    st.session_state["fc_res"] = (text, items)
                except LLMError as e:
                    st.error(str(e))
        pair = st.session_state.get("fc_res")
        if pair and pair[0] == text:
            items: list[fc.SentenceAnalysis] = pair[1]
            counts = fc.summary_counts(items)
            cols = st.columns(5)
            for col, key in zip(cols, fc.CLASS_LABELS):
                col.metric(fc.CLASS_LABELS[key], counts[key])
            cols[4].metric("⚠️ Надто сильні", counts["too_strong"])
            by_id = {f.id: f for f in facts}
            for i, a in enumerate(items):
                color = fc.CLASS_COLORS[a.cls]
                badge = f"<span class='badge' style='background:{color}'>{fc.CLASS_LABELS[a.cls]}</span>"
                warn = "<span class='badge' style='background:#b91c1c'>надто сильно</span>" if a.too_strong else ""
                st.markdown(f"<div class='sent' style='border-color:{color}'>{badge}{warn}<br>{html.escape(a.sentence)}</div>", unsafe_allow_html=True)
                details = "; ".join(a.reasons)
                if details:
                    st.caption("Чому: " + details)
                if a.strength_issues:
                    st.markdown("**Проблема:** " + "; ".join(a.strength_issues))
                if a.suggestion:
                    st.markdown(f"**Точніше:** {html.escape(a.suggestion)}")
                if a.supporting_fact_ids:
                    with st.expander(f"Підтримка в дослідженні ({len(a.supporting_fact_ids)})"):
                        for fid in a.supporting_fact_ids:
                            if fid in by_id:
                                render_fact(by_id[fid])
            sugg = [a for a in items if a.suggestion]
            if sugg and block is not None:
                if st.button(f"✅ Замінити {len(sugg)} надто сильних речень на точніші", key="fc_apply"):
                    new = text
                    for a in sugg:
                        new = new.replace(drafting.strip_citations(a.sentence), a.suggestion)
                    _apply(session, project, block, pidx, new)
                    st.session_state.pop("fc_res", None)
                    flash("Формулювання уточнено. Перевірте текст — маркери [F#] при заміні речення можуть зникнути.")
                    st.rerun()

    # ------------------------------------------------------------ історія
    with t_hist:
        hist = sorted(project.edits, key=lambda e: e.id, reverse=True)[:50]
        if not hist:
            st.caption("Історія порожня.")
        for e in hist:
            label = ed.OPERATIONS.get(e.operation, {}).get("label", e.operation)
            with st.expander(f"{fmt_dt(e.created_at)} · {label} · {'застосовано' if e.applied else 'не застосовано'}"):
                c1, c2 = st.columns(2)
                c1.markdown("**Було**")
                c1.text(e.input_text)
                c2.markdown("**Стало**")
                c2.text(e.output_text)
