"""«Чи звучить це як я?» — порівняння метрик фрагмента з профілем стилю.

Оцінка — евристична: 100 мінус штрафи за конкретні відхилення. Кожен штраф має діагностику,
тому число завжди пояснюване. LLM (якщо є) дає переписаний варіант.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.llm import LLMService
from app.services.style_analyzer import profile_to_prompt, text_metrics
from app.utils import text as T

# Орієнтир, якщо профілю ще немає: «документальний стриманий стиль».
DEFAULT_METRICS = {
    "sentence_len": {"mean": 13.0, "median": 12.0, "p25": 8.0, "p75": 17.0, "p90": 23.0, "stdev": 6.5},
    "sentence_mix": {"short_share": 0.3, "medium_share": 0.6, "long_share": 0.1},
    "paragraph": {"mean_sentences": 3.5, "mean_words": 45.0, "p75_words": 60.0},
    "question_rate": 0.06,
    "exclamation_rate": 0.01,
    "author_i_per_1k": 4.0,
    "abstract_ratio": 0.09,
    "numbers_per_1k": 12.0,
    "cliches_per_1k": 0.2,
    "emotive_per_1k": 0.8,
}


@dataclass
class Diagnostic:
    severity: str  # high | medium | low
    message: str
    penalty: int
    examples: list[str] = field(default_factory=list)


@dataclass
class StyleCheckResult:
    score: int
    label: str
    diagnostics: list[Diagnostic]
    metrics: dict
    reference: dict
    used_default_reference: bool
    rewrite: str = ""


def _label(score: int) -> str:
    if score >= 85:
        return "Дуже схоже на тебе"
    if score >= 70:
        return "Схоже, з дрібними відхиленнями"
    if score >= 50:
        return "Частково схоже"
    return "Не схоже на твій стиль"


def _own_phrases(profile: dict | None) -> set[str]:
    """Фрази, які автор сам регулярно вживає: кліше з корпусу (≥2 рази) і фірмові формули."""
    if not profile:
        return set()
    own = {c for c, n in (profile.get("metrics", {}).get("cliches") or {}).items() if n >= 2}
    own |= {f for f, _ in profile.get("formulas", [])}
    return own


def check_style(text: str, profile: dict | None) -> StyleCheckResult:
    ref = (profile or {}).get("metrics") or DEFAULT_METRICS
    used_default = not (profile and profile.get("metrics"))
    m = text_metrics(text)
    sents = T.split_sentences(text)
    diags: list[Diagnostic] = []
    rs, ts = ref["sentence_len"], m["sentence_len"]

    # 1. довжина речень
    if m["sentences"]:
        if ts["mean"] > rs["p75"] * 1.3:
            longest = sorted(sents, key=lambda s: -len(T.words(s)))[:2]
            diags.append(Diagnostic("high", f"Речення значно довші за твій звичний діапазон: в середньому {ts['mean']} слів проти твоїх {rs['mean']} (типово {rs['p25']:.0f}–{rs['p75']:.0f}).", 20, [T.truncate(s, 220) for s in longest]))
        elif ts["mean"] > rs["p75"]:
            diags.append(Diagnostic("medium", f"Речення трохи довші, ніж зазвичай: {ts['mean']} слів проти {rs['mean']}.", 10))
        elif ts["mean"] < rs["p25"] * 0.8 and m["sentences"] >= 4:
            diags.append(Diagnostic("medium", f"Речення коротші за звичні ({ts['mean']} проти {rs['mean']} слів) — текст може звучати рвано.", 8))
    # 2. дуже довгі речення
    limit = max(rs.get("p90", 23), 22)
    too_long = [s for s in sents if len(T.words(s)) > limit * 1.2]
    if too_long:
        diags.append(Diagnostic("medium", f"{len(too_long)} речень довші за {limit * 1.2:.0f} слів — у твоєму корпусі такі рідкісні.", min(6 * len(too_long), 15), [T.truncate(s, 220) for s in too_long[:3]]))
    # 3. ритм
    if m["sentences"] >= 5 and ts["stdev"] < rs.get("stdev", 6) * 0.5:
        diags.append(Diagnostic("low", "Монотонний ритм: речення майже однакової довжини. У тебе короткі речення чергуються з середніми.", 6))
    if m["sentences"] >= 5 and m["sentence_mix"]["short_share"] < ref["sentence_mix"]["short_share"] * 0.4:
        diags.append(Diagnostic("medium", f"Бракує коротких рубаних речень: {m['sentence_mix']['short_share']*100:.0f}% проти твоїх {ref['sentence_mix']['short_share']*100:.0f}%. Бракує компактного документального ритму.", 8))
    # 4. абстракції
    if m["abstract_ratio"] > max(ref["abstract_ratio"] * 1.5, 0.1):
        abstract_words = sorted({w for w in T.words_lower(text) if T.is_abstract(w)})[:10]
        diags.append(Diagnostic("high" if m["abstract_ratio"] > ref["abstract_ratio"] * 2 else "medium", f"Забагато абстрактних іменників: {m['abstract_ratio']*100:.0f}% змістових слів проти твоїх {ref['abstract_ratio']*100:.0f}%.", 12, [", ".join(abstract_words)]))
    # 5. кліше — крім тих, що регулярно трапляються в корпусі автора (це його звороти, а не чужі штампи)
    own = _own_phrases(profile)
    foreign = {c: n for c, n in m["cliches"].items() if c not in own}
    if foreign:
        n = sum(foreign.values())
        diags.append(Diagnostic("high", f"Кліше та шаблонні фрази ({n}): у твоєму корпусі вони рідкісні або відсутні.", min(8 * n, 24), [", ".join(f"«{c}»" for c in foreign)]))
    # 6. шаблонний вступ
    first = sents[0].lower() if sents else ""
    gi = [g for g in T.GENERIC_INTROS if g in first and not any(g in o for o in own)]
    if gi:
        diags.append(Diagnostic("high", "Шаблонний вступ: починається з загальної фрази замість конкретики.", 10, [T.truncate(sents[0], 200)]))
    # 7. конкретика
    if m["words"] >= 60 and ref["numbers_per_1k"] >= 6 and m["numbers_per_1k"] < ref["numbers_per_1k"] * 0.3:
        diags.append(Diagnostic("medium", f"Мало конкретики: {m['numbers_per_1k']:.0f} цифр/дат на 1000 слів проти твоїх {ref['numbers_per_1k']:.0f}.", 10))
    # 8. емоційність
    if m["exclamation_rate"] > ref["exclamation_rate"] + 0.05:
        diags.append(Diagnostic("medium", f"Забагато знаків оклику: {m['exclamation_rate']*100:.0f}% речень проти твоїх {ref['exclamation_rate']*100:.0f}%.", 8))
    if m["emotive_per_1k"] > ref["emotive_per_1k"] + 3:
        diags.append(Diagnostic("medium", "Емоційні оцінки («шок», «жах», «неймовірно») частіші, ніж у тебе.", 8, [", ".join(m["emotive"])]))
    # 9. риторичні питання
    if m["sentences"] >= 4 and m["question_rate"] > ref["question_rate"] + 0.15:
        diags.append(Diagnostic("low", f"Риторичних питань більше, ніж зазвичай: {m['question_rate']*100:.0f}% проти {ref['question_rate']*100:.0f}%.", 6))
    # 10. присутність автора
    if m["words"] >= 80 and m["author_i_per_1k"] > ref["author_i_per_1k"] * 3 + 5:
        diags.append(Diagnostic("low", f"Забагато «я/мені»: {m['author_i_per_1k']:.0f} на 1000 слів проти твоїх {ref['author_i_per_1k']:.0f}. Особистий голос має бути дозованим.", 6))
    # 11. абзаци
    if m["paragraph"]["mean_words"] > ref["paragraph"].get("p75_words", ref["paragraph"]["mean_words"]) * 1.5 and m["words"] >= 80:
        diags.append(Diagnostic("low", f"Абзаци задовгі: ≈{m['paragraph']['mean_words']:.0f} слів проти твоїх ≈{ref['paragraph']['mean_words']:.0f}.", 6))
    # 12. нетипові звороти: частка біграм фрагмента, яких немає в корпусі — лише як інформація
    if profile and profile.get("lexicon"):
        common = {g for g, _ in profile["lexicon"].get("bigrams", [])}
        text_low = text.lower()
        hits = [g for g in common if g in text_low]
        if hits:
            diags.append(Diagnostic("low", "Плюс: є типові для тебе звороти.", 0, [", ".join(f"«{h}»" for h in hits[:6])]))

    penalty = sum(d.penalty for d in diags)
    score = max(0, min(100, 100 - penalty))
    diags.sort(key=lambda d: {"high": 0, "medium": 1, "low": 2}[d.severity])
    return StyleCheckResult(score=score, label=_label(score), diagnostics=diags, metrics=m, reference=ref, used_default_reference=used_default)


REWRITE_SYSTEM = """Ти — редактор, який переписує текст у голосі конкретного українського документального автора.
Не додавай фактів, не посилюй твердження, зберігай маркери [F…]. Повертай ЛИШЕ переписаний текст.

{style}"""


def suggest_rewrite(text: str, result: StyleCheckResult, profile: dict | None, llm: LLMService) -> str:
    issues = "\n".join(f"- {d.message}" for d in result.diagnostics if d.penalty > 0) or "- суттєвих відхилень немає, лише відшліфуй"
    prompt = f"Діагностика відхилень від стилю автора:\n{issues}\n\nПерепиши текст, усунувши ці відхилення:\n{text}"
    return llm.complete(REWRITE_SYSTEM.format(style=profile_to_prompt(profile or {})), prompt, purpose="style_rewrite", max_tokens=2500).strip()
