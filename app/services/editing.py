"""Операції редагування тексту (через LLM; частина має офлайн-евристику)."""
from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.db.models import EditOperation, Project
from app.llm import LLMService
from app.services.style_service import profile_prompt
from app.utils import text as T

OPERATIONS: dict[str, dict] = {
    "shorten": {
        "label": "Скоротити",
        "instruction": "Скороти текст приблизно на 30–40%. Прибери повтори, воду, вставні конструкції. Збережи всі факти і маркери [F…].",
    },
    "sharpen": {
        "label": "Зробити гостріше",
        "instruction": "Зроби текст гострішим: коротші речення, сильніші дієслова, чіткі акценти. Не додавай фактів і не посилюй твердження понад джерела.",
    },
    "concrete": {
        "label": "Додати конкретики",
        "instruction": "Заміни абстракції конкретикою: хто, що, де, коли, скільки — ЛИШЕ з того, що вже є в тексті або в наданих фактах. Де конкретики бракує — постав позначку [ПОТРІБЕН ФАКТ].",
    },
    "my_style": {
        "label": "Переписати ближче до мого стилю",
        "instruction": "Перепиши текст максимально близько до профілю стилю автора: ритм, довжина речень, типові переходи, стриманість.",
    },
    "reduce_cliches": {
        "label": "Прибрати кліше",
        "instruction": "Прибери кліше, канцелярит, штампи й порожні підсилювачі. Заміни їх прямими формулюваннями.",
    },
    "documentary": {
        "label": "Зробити документальніше",
        "instruction": "Зроби текст документальнішим: більше атрибуції («за даними…», «у документі…»), менше оцінок, чітке розділення факту й інтерпретації.",
    },
    "less_emotional": {
        "label": "Менш емоційно",
        "instruction": "Зменши емоційність: прибери оклики, емоційні епітети, драматизацію. Залиш факти говорити самі за себе.",
    },
    "transition": {
        "label": "Додати перехід між блоками",
        "instruction": "Напиши 1–3 речення переходу між блоком А і блоком Б у стилі автора. Перехід має логічно зв'язати думки, без кліше. Поверни ЛИШЕ текст переходу.",
    },
}

EDIT_SYSTEM = """Ти — редактор українського документального YouTube-сценарію.
Правила: не вигадуй фактів; зберігай маркери джерел [F12] біля відповідних речень; не посилюй твердження.
Повертай ЛИШЕ відредагований текст, без коментарів і лапок.

{style}"""


def offline_reduce_cliches(text: str) -> str:
    out = text
    for c in sorted(T.CLICHES, key=len, reverse=True):
        tail = r"" if c.endswith("що") else r"(?:,?\s*що(?![\w]))?"
        out = re.sub(r"(?i)(?<![\w])" + re.escape(c) + r"(?![\w])" + tail + r"[,:]?\s*", "", out)
    # велика літера на початку речень після видалення
    out = re.sub(r"(^|[.!?]\s+)([a-zа-яіїєґ])", lambda m: m.group(1) + m.group(2).upper(), out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


def offline_less_emotional(text: str) -> str:
    out = re.sub(r"!+", ".", text)
    for w in T.EMOTIVE_WORDS:
        out = re.sub(r"(?i)(?<![\w])" + re.escape(w) + r"(?![\w])\s*", "", out)
    return re.sub(r"[ \t]{2,}", " ", out).strip()


OFFLINE = {"reduce_cliches": offline_reduce_cliches, "less_emotional": offline_less_emotional}


def run_operation(
    session: Session,
    project: Project,
    op: str,
    text: str,
    llm: LLMService | None,
    draft_block_id: int | None = None,
    text_b: str = "",
) -> tuple[str, str, int]:
    """Повертає (результат, метод: llm|heuristic, id запису EditOperation)."""
    spec = OPERATIONS[op]
    if llm is not None and llm.available:
        if op == "transition":
            prompt = f"{spec['instruction']}\n\nБЛОК А (кінець):\n{T.truncate(text, 1500)}\n\nБЛОК Б (початок):\n{T.truncate(text_b, 1500)}"
        else:
            prompt = f"{spec['instruction']}\n\nТЕКСТ:\n{text}"
        result = llm.complete(EDIT_SYSTEM.format(style=profile_prompt(project.style_profile)), prompt, purpose=f"edit_{op}", max_tokens=3000)
        method = "llm"
    elif op in OFFLINE:
        result = OFFLINE[op](text)
        method = "heuristic"
    else:
        raise RuntimeError("Ця операція потребує ШІ-провайдера. Налаштуйте ключ у .env (див. «Налаштування»).")
    rec = EditOperation(
        project_id=project.id,
        draft_block_id=draft_block_id,
        operation=op,
        input_text=text if op != "transition" else f"{text}\n\n---\n\n{text_b}",
        output_text=result.strip(),
    )
    session.add(rec)
    session.commit()
    return result.strip(), method, rec.id


def mark_applied(session: Session, op_id: int) -> None:
    rec = session.get(EditOperation, op_id)
    if rec is not None:
        rec.applied = True
        session.commit()
