from __future__ import annotations

import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import DB_PATH, UPLOAD_DIR, load_llm_settings, provider_model
from app.db.models import LLMRunLog
from app.llm import PROVIDERS, LLMError, LLMService, build_provider
from app.ui.common import fmt_dt

FEATURES = [
    ("Проєкти, імпорт файлів, чанкінг", "✅", "✅"),
    ("Витяг фактів", "евристика", "✅ точніше"),
    ("Профіль стилю (метрики)", "✅", "✅ + якісний опис голосу"),
    ("Структура", "шаблон + автопідбір фактів", "✅ генерація з дослідження"),
    ("Генерація сценарію", "шаблонний каркас з фактами", "✅"),
    ("Інструменти редагування", "лише «кліше» і «емоційність»", "✅ усі"),
    ("«Чи звучить як я?»", "✅ діагностика й оцінка", "✅ + переписаний варіант"),
    ("«Факт чи інтерпретація»", "✅ евристика", "✅ точніше"),
]


def render(session: Session) -> None:
    st.title("Налаштування")
    s = load_llm_settings()

    st.subheader("ШІ-провайдер")
    rows = []
    for name in PROVIDERS:
        p = build_provider(name, settings=s)
        rows.append({"Провайдер": name, "Модель за замовчуванням": provider_model(name), "Статус": "налаштовано" if p.is_configured() else "не налаштовано", "Що потрібно": "" if p.is_configured() else p.config_hint()})
    st.dataframe(rows, hide_index=True, width="stretch")

    opts = [*PROVIDERS, "none"]
    cur = st.session_state.get("llm_provider", s.provider)
    c1, c2 = st.columns(2)
    prov = c1.selectbox("Провайдер для цієї сесії", opts, index=opts.index(cur) if cur in opts else 0, format_func=lambda x: "вимкнути ШІ" if x == "none" else x)
    default_model = s.model if prov == s.provider else provider_model(prov)
    model = c2.text_input("Модель", st.session_state.get("llm_model") or default_model if prov == cur else default_model, disabled=prov == "none")
    b1, b2 = st.columns(2)
    if b1.button("Застосувати", type="primary"):
        st.session_state["llm_provider"] = prov
        st.session_state["llm_model"] = model.strip() if model.strip() != default_model else ""
        st.rerun()
    if b2.button("Перевірити з'єднання", disabled=prov == "none"):
        svc = LLMService(build_provider(prov, model or None, s), session=session)
        if not svc.available:
            st.error(svc.unavailable_reason())
        else:
            with st.spinner("Надсилаю тестовий запит…"):
                try:
                    out = svc.complete("Відповідай одним словом.", "Скажи «працює».", purpose="ping", max_tokens=20)
                    st.success(f"Відповідь моделі: {out.strip()}")
                except LLMError as e:
                    st.error(str(e))
    st.caption("Вибір тут діє до перезапуску сторінки. Постійні налаштування — у файлі `.env` (див. `.env.example`).")

    with st.expander("Як налаштувати ключі (.env)"):
        st.code(
            "cp .env.example .env\n"
            "# Anthropic\nDEFAULT_LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=sk-ant-...\n\n"
            "# або OpenAI\nDEFAULT_LLM_PROVIDER=openai\nOPENAI_API_KEY=sk-...\n\n"
            "# або локально через Ollama\nDEFAULT_LLM_PROVIDER=ollama\nOLLAMA_BASE_URL=http://localhost:11434\nOLLAMA_MODEL=llama3.1",
            language="bash",
        )
        st.caption("Після зміни .env перезапустіть `streamlit run main.py`.")

    st.subheader("Що працює без ШІ")
    st.dataframe([{"Функція": a, "Без ШІ": b, "З ШІ": c} for a, b, c in FEATURES], hide_index=True, width="stretch")

    st.subheader("Журнал викликів ШІ")
    logs = list(session.scalars(select(LLMRunLog).order_by(LLMRunLog.id.desc()).limit(100)))
    if not logs:
        st.caption("Викликів ще не було.")
    else:
        st.dataframe(
            [
                {"Час": fmt_dt(l.created_at), "Задача": l.purpose, "Провайдер": l.provider, "Модель": l.model, "Промпт, симв.": l.prompt_chars, "Відповідь, симв.": l.response_chars, "мс": l.duration_ms, "OK": l.success, "Помилка": l.error[:200]}
                for l in logs
            ],
            hide_index=True,
            width="stretch",
        )

    st.subheader("Дані")
    st.markdown(f"- База даних: `{DB_PATH}`\n- Завантажені файли: `{UPLOAD_DIR}`")
    st.caption("Усі дані зберігаються локально. Для резервної копії достатньо скопіювати папку data/.")
