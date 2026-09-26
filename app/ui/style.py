from __future__ import annotations

import streamlit as st
from sqlalchemy.orm import Session

from app.llm import LLMError
from app.parsers import ParseError
from app.services import projects as psvc
from app.services import style_service as svc
from app.ui.common import current_project, flash, fmt_dt, get_llm, llm_notice


def _table(pairs, col1: str, col2: str = "Кількість") -> None:
    if pairs:
        st.dataframe([{col1: a, col2: b} for a, b in pairs], hide_index=True, width="stretch")
    else:
        st.caption("Недостатньо даних.")


def render(session: Session) -> None:
    st.title("Профіль стилю")
    st.caption(
        "Завантажте кілька своїх попередніх сценаріїв. Застосунок порахує метрики (довжина речень, абзаців, переходи, "
        "кліше, риторичні питання, присутність автора, абстрактність) і складе профіль, який використовується для генерації і перевірки."
    )
    llm = get_llm(session)
    profiles = psvc.list_profiles(session)
    project = current_project(session)

    c1, c2 = st.columns([3, 2])
    with c2, st.container(border=True):
        with st.form("new_profile", clear_on_submit=True):
            name = st.text_input("Новий профіль", placeholder="Напр.: Мій стиль 2025")
            if st.form_submit_button("Створити"):
                p = svc.create_profile(session, name)
                st.session_state["profile_id"] = p.id
                st.rerun()
    if not profiles:
        c1.info("Профілів ще немає. Створіть профіль праворуч або завантажте демо на сторінці «Проєкти».")
        return
    ids = [p.id for p in profiles]
    pre = st.session_state.get("profile_id") or (project.style_profile_id if project else None)
    pid = c1.selectbox("Профіль", ids, index=ids.index(pre) if pre in ids else 0, format_func=lambda i: next(p.name for p in profiles if p.id == i))
    st.session_state["profile_id"] = pid
    profile = next(p for p in profiles if p.id == pid)
    if project:
        if project.style_profile_id == profile.id:
            c1.success(f"Використовується в поточному проєкті «{project.title}».")
        elif c1.button(f"Використовувати в проєкті «{project.title}»"):
            project.style_profile_id = profile.id
            session.commit()
            st.rerun()

    t_corpus, t_profile, t_prompt = st.tabs([f"📚 Корпус ({len(profile.corpus)})", "📊 Профіль", "🧩 Промпт / JSON"])

    with t_corpus:
        files = st.file_uploader("Попередні сценарії (.md, .txt, .docx, .pdf)", type=["md", "markdown", "txt", "docx", "pdf"], accept_multiple_files=True, key=f"sup_{st.session_state.get('sup_n', 0)}")
        if st.button("Додати в корпус і перерахувати профіль", type="primary", disabled=not files):
            for f in files:
                try:
                    svc.add_corpus_file(session, profile, f.name, f.getvalue())
                except ParseError as e:
                    st.error(f"{f.name}: {e}")
            session.refresh(profile)
            if profile.corpus:
                svc.rebuild_profile(session, profile)
            st.session_state["sup_n"] = st.session_state.get("sup_n", 0) + 1
            flash("Корпус оновлено, профіль перераховано.")
            st.rerun()
        with st.expander("Або вставити текст сценарію"):
            with st.form("paste_corpus", clear_on_submit=True):
                nm = st.text_input("Назва")
                tx = st.text_area("Текст", height=200)
                if st.form_submit_button("Додати") and tx.strip():
                    svc.add_corpus_text(session, profile, nm or "вставлений текст", tx)
                    svc.rebuild_profile(session, profile)
                    st.rerun()
        total = sum(d.word_count for d in profile.corpus)
        st.caption(f"Усього слів у корпусі: {total}. Рекомендовано ≥ 3 000 слів (3–5 сценаріїв) для стабільних метрик.")
        if 0 < total < 1500:
            st.warning("Корпус малий: метрики можуть бути нестабільними.")
        for d in profile.corpus:
            a, b = st.columns([6, 1])
            a.markdown(f"**{d.filename}** · {d.word_count} слів · {fmt_dt(d.created_at)}")
            if b.button("🗑", key=f"cd_{d.id}"):
                session.delete(d)
                session.commit()
                session.refresh(profile)
                if profile.corpus:
                    svc.rebuild_profile(session, profile)
                st.rerun()

    prof = profile.profile or {}
    with t_profile:
        b1, b2 = st.columns(2)
        if b1.button("🔄 Перерахувати профіль", disabled=not profile.corpus):
            svc.rebuild_profile(session, profile)
            st.rerun()
        if b2.button("🤖 Збагатити якісним описом через ШІ", disabled=not (llm.available and profile.corpus), help="Модель читає фрагменти корпусу і додає опис голосу, прийоми та приклади речень"):
            with st.spinner("Модель аналізує голос…"):
                try:
                    svc.enrich_with_llm(session, profile, llm)
                    flash("Профіль збагачено.")
                    st.rerun()
                except (LLMError, ValueError) as e:
                    st.error(str(e))
        llm_notice(llm, "Якісний опис голосу", "профіль будується лише з метрик — це працює і так.")
        if not prof:
            st.info("Профіль ще не побудовано: додайте тексти в корпус.")
        else:
            m = prof["metrics"]
            with st.container(border=True):
                st.subheader("Підсумок")
                st.text(prof.get("summary", ""))
            k = st.columns(6)
            k[0].metric("Сер. речення", f"{m['sentence_len']['mean']} сл.")
            k[1].metric("Діапазон (p25–p75)", f"{m['sentence_len']['p25']:.0f}–{m['sentence_len']['p75']:.0f}")
            k[2].metric("Абзац", f"{m['paragraph']['mean_sentences']} реч.")
            k[3].metric("Риторичні питання", f"{m['question_rate'] * 100:.1f}%")
            k[4].metric("«Я» / 1000 слів", m["author_i_per_1k"])
            k[5].metric("Цифри / 1000 слів", m["numbers_per_1k"])
            x1, x2 = st.columns(2)
            with x1:
                st.markdown("#### Голос")
                for t in prof.get("tone", []):
                    st.markdown(f"- {t}")
                notes = prof.get("llm_notes") or {}
                if notes.get("voice"):
                    st.info(notes["voice"])
                st.markdown("#### ✅ Робити")
                for t in prof.get("do", []) + (notes.get("do") or []):
                    st.markdown(f"- {t}")
                st.markdown("#### ⛔ Уникати")
                for t in prof.get("avoid", []) + (notes.get("avoid") or []):
                    st.markdown(f"- {t}")
            with x2:
                st.markdown("#### Вступ / переходи / фінал")
                st.markdown(f"**Вступ:** {prof['intro_style']['label']}" + (f"  \n_{notes['intro']}_" if notes.get("intro") else ""))
                st.markdown(f"**Переходи:** {prof['transition_style']}" + (f"  \n_{notes['transitions']}_" if notes.get("transitions") else ""))
                st.markdown(f"**Фінал:** {prof['ending_style']['label']}" + (f"  \n_{notes['ending']}_" if notes.get("ending") else ""))
                if notes.get("sample_sentences"):
                    st.markdown("#### Характерні речення")
                    for s_ in notes["sample_sentences"]:
                        st.markdown(f"> {s_}")
                st.markdown("#### Рідкісні / небажані конструкції")
                st.caption("Кліше, яких немає у вашому корпусі:")
                st.markdown(", ".join(f"«{c}»" for c in prof.get("rare_or_absent_cliches", [])) or "—")
                if m.get("cliches"):
                    st.caption("Кліше, які все ж трапляються у корпусі:")
                    st.markdown(", ".join(f"«{c}» ×{n}" for c, n in m["cliches"].items()))
            st.markdown("#### Поширені конструкції")
            l1, l2, l3, l4 = st.columns(4)
            with l1:
                st.caption("Біграми")
                _table(prof["lexicon"]["bigrams"], "Біграма")
            with l2:
                st.caption("Триграми")
                _table(prof["lexicon"]["trigrams"], "Триграма")
            with l3:
                st.caption("Початки речень")
                _table(prof["lexicon"]["sentence_starters"], "Початок")
            with l4:
                st.caption("Переходи")
                _table(list(prof.get("transitions", {}).items()), "Маркер")
            with st.expander("Частотна лексика"):
                _table(prof["lexicon"]["top_words"], "Слово")

    with t_prompt:
        st.caption("Цей текст підставляється в системний промпт при генерації та редагуванні. Можна скопіювати в будь-який інший інструмент.")
        st.code(svc.profile_prompt(profile), language="markdown")
        js = svc.profile_json(profile)
        st.download_button("⬇️ Завантажити профіль JSON", js, file_name=f"style_profile_{profile.id}.json", mime="application/json")
        with st.expander("JSON"):
            st.json(prof, expanded=False)
        with st.expander("🗑 Видалити профіль"):
            if st.button("Видалити профіль назавжди", key="del_prof"):
                session.delete(profile)
                session.commit()
                st.session_state.pop("profile_id", None)
                st.rerun()
