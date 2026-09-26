"""«Факт чи інтерпретація»: класифікація речень + пошук надто сильних тверджень.

Класи:
- fact            — перевірюване твердження (цифри, дати, дії, атрибуція);
- interpretation  — висновок/оцінка автора на основі фактів;
- speculation     — припущення, гіпотеза;
- rhetorical      — риторичне/емоційне обрамлення (питання, оклики, емоційні оцінки).

Евристика працює офлайн; LLM-режим дає точнішу класифікацію й переформулювання.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.db.models import ExtractedFact
from app.llm import LLMService
from app.services.drafting import cited_ids, strip_citations
from app.services.retrieval import BM25, token_overlap
from app.utils import text as T

CLASS_LABELS = {
    "fact": "Фактичне твердження",
    "interpretation": "Інтерпретація",
    "speculation": "Спекуляція / припущення",
    "rhetorical": "Риторика / емоційне обрамлення",
}
CLASS_COLORS = {"fact": "#3B82F6", "interpretation": "#A855F7", "speculation": "#F59E0B", "rhetorical": "#EF4444"}

SOFTEN = [
    (r"\bстворені(?:,)? (?:для того, )?щоб\b", "розроблені так, щоб"),
    (r"\bзадумані(?:,)? щоб\b", "побудовані так, щоб"),
    (r"\bвсі\b", "багато"), (r"\bусі\b", "багато"), (r"\bВсі\b", "Багато"), (r"\bУсі\b", "Багато"),
    (r"\bзавжди\b", "часто"), (r"\bЗавжди\b", "Часто"),
    (r"\bніколи\b", "рідко"), (r"\bНіколи\b", "Рідко"),
    (r"\bніхто\b", "мало хто"), (r"\bНіхто\b", "Мало хто"),
    (r"\bповністю\b", "значною мірою"),
    (r"\bабсолютно\s", ""), (r"\bбезумовно,?\s", ""), (r"\bбеззаперечно,?\s", ""),
    (r"\bочевидно\b", "ймовірно"), (r"\bОчевидно\b", "Ймовірно"),
    (r"\bдоведено\b", "є дані"), (r"\bДоведено\b", "Є дані"),
    (r"\bспеціально\s", ""), (r"\bнавмисно\s", ""),
    (r"\bзмушують\b", "підштовхують"), (r"\bзмушує\b", "підштовхує"),
    (r"\bруйнують\b", "можуть шкодити"), (r"\bруйнує\b", "може шкодити"),
    (r"\bзнищують\b", "послаблюють"), (r"\bзнищує\b", "послаблює"),
    (r"\bкожен\b", "чимало"), (r"\bкожна\b", "чимало"), (r"\bєдина причина\b", "одна з причин"),
    (r"\bвиключно\b", "переважно"), (r"\bгарантовано\b", "з високою ймовірністю"),
]


@dataclass
class SentenceAnalysis:
    sentence: str
    cls: str
    reasons: list[str] = field(default_factory=list)
    too_strong: bool = False
    strength_issues: list[str] = field(default_factory=list)
    supporting_fact_ids: list[int] = field(default_factory=list)
    support_score: float = 0.0
    suggestion: str = ""


def classify_sentence(sentence: str) -> tuple[str, list[str]]:
    s = strip_citations(sentence).strip()
    low = s.lower()
    reasons: list[str] = []
    n_words = len(T.words(s))
    has_num = bool(re.search(r"\d", s))
    proper = sum(1 for w in T.words(s)[1:] if w[:1].isupper() and not w.isupper())
    reporting = T.count_phrases(low, T.REPORTING_VERBS)
    hedges = T.count_phrases(low, T.HEDGES)
    opinion = T.count_phrases(low, T.OPINION_MARKERS)
    emotive = T.count_phrases(low, T.EMOTIVE_WORDS)

    if s.endswith("?"):
        return "rhetorical", ["питання"]
    if s.endswith("!") or (emotive and not has_num and not reporting):
        return "rhetorical", ["оклик/емоційна лексика: " + ", ".join(emotive) if emotive else "оклик"]
    if hedges and not reporting:
        return "speculation", ["маркери припущення: " + ", ".join(hedges)]
    if opinion and not (has_num and reporting):
        reasons.append("маркери оцінки/висновку: " + ", ".join(opinion))
        return "interpretation", reasons
    if has_num or reporting or proper >= 2:
        if has_num:
            reasons.append("містить цифри/дати")
        if reporting:
            reasons.append("атрибуція: " + ", ".join(reporting))
        if proper >= 2:
            reasons.append("називає конкретних акторів")
        return "fact", reasons
    if n_words <= 5:
        return "rhetorical", ["коротка фраза без фактичного змісту"]
    abstract = sum(1 for w in T.words_lower(s) if T.is_abstract(w))
    if abstract >= 2:
        return "interpretation", ["узагальнення з абстрактними поняттями"]
    return "interpretation", ["немає перевірюваних деталей (цифр, дат, імен, атрибуції)"]


def soften(sentence: str) -> str:
    out = sentence
    for pat, rep in SOFTEN:
        out = re.sub(pat, rep, out)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out[:1].upper() + out[1:] if out else out


def _numbers(s: str) -> set[str]:
    return {n.replace(",", ".") for n in re.findall(r"\d+(?:[.,]\d+)?", s)}


def analyze_paragraph(text: str, facts: list[ExtractedFact]) -> list[SentenceAnalysis]:
    sents = T.split_sentences(text)
    bm = BM25([f"{f.statement} {f.snippet}" for f in facts]) if facts else None
    by_id = {f.id: f for f in facts}
    out: list[SentenceAnalysis] = []
    for s in sents:
        cls, reasons = classify_sentence(s)
        a = SentenceAnalysis(sentence=s, cls=cls, reasons=reasons)
        clean = strip_citations(s)
        low = clean.lower()
        # підтримка у джерелах
        cited = [i for i in cited_ids(s) if i in by_id]
        if cited:
            a.supporting_fact_ids = cited
            a.support_score = max(token_overlap(clean, by_id[i].statement + " " + by_id[i].snippet) for i in cited)
        elif bm is not None:
            top = bm.top(clean, k=2, min_score=0.5)
            cand = [(facts[i], token_overlap(clean, facts[i].statement + " " + facts[i].snippet)) for i, _ in top]
            cand = [(f, ov) for f, ov in cand if ov >= 0.3]
            if cand:
                a.supporting_fact_ids = [f.id for f, _ in cand]
                a.support_score = max(ov for _, ov in cand)
        # сила твердження
        absolutes = T.count_phrases(low, T.ABSOLUTE_MARKERS)
        if absolutes and cls != "speculation":
            a.strength_issues.append("абсолютні формулювання: " + ", ".join(f"«{x}»" for x in absolutes))
        if cls == "fact" and facts and not a.supporting_fact_ids:
            a.strength_issues.append("фактичне твердження без знайденого джерела в дослідженні")
        if cls == "fact" and a.supporting_fact_ids:
            src_nums = set().union(*[_numbers(by_id[i].statement + " " + by_id[i].snippet) for i in a.supporting_fact_ids])
            diff = _numbers(clean) - src_nums
            if diff and src_nums:
                a.strength_issues.append("цифри не знайдено в джерелі: " + ", ".join(sorted(diff)))
            weak = [by_id[i] for i in a.supporting_fact_ids if by_id[i].confidence < 0.5]
            if weak and not T.count_phrases(low, T.HEDGES + ["за даними", "за оцінками"]):
                a.strength_issues.append("джерело має низьку впевненість, а речення звучить категорично")
        if cls == "interpretation" and absolutes:
            a.strength_issues.append("інтерпретація подана як беззаперечний факт")
        a.too_strong = bool(a.strength_issues)
        if a.too_strong:
            sug = soften(clean)
            if "без знайденого джерела" in " ".join(a.strength_issues):
                sug = "[ПОТРІБНЕ ДЖЕРЕЛО] " + sug
            if "низьку впевненість" in " ".join(a.strength_issues) and not sug.lower().startswith("за"):
                sug = "За наявними даними, " + sug[:1].lower() + sug[1:]
            a.suggestion = sug if sug != clean else ""
        out.append(a)
    return out


FC_SYSTEM = """Ти — фактчекер і науковий редактор українського документального YouTube-каналу.
Класифікуй кожне речення тексту і знаходь надто сильні твердження. Будь суворим до overclaiming.
Відповідай ЛИШЕ валідним JSON."""

FC_PROMPT = """Факти, що є в дослідженні (id: твердження | впевненість):
{facts}

Текст для аналізу:
<<<
{text}
>>>

Для КОЖНОГО речення тексту поверни елемент JSON-масиву:
[{{"sentence": "речення дослівно",
   "type": "fact|interpretation|speculation|rhetorical",
   "reason": "коротко чому",
   "supporting_fact_ids": [id з переліку, якщо речення ними підтверджується],
   "too_strong": true/false,
   "issue": "що саме перебільшено (якщо too_strong)",
   "suggestion": "точніше, обережніше формулювання (якщо too_strong), інакше пусто"}}]

Приклад: «Казино створені, щоб люди втрачали контроль.» → too_strong; suggestion: «Багато механік казино розроблені, щоб збільшувати тривалість і повторюваність гри.»"""


def analyze_with_llm(text: str, facts: list[ExtractedFact], llm: LLMService) -> list[SentenceAnalysis]:
    if facts:
        bm = BM25([f"{f.statement} {f.snippet}" for f in facts])
        top = [facts[i] for i, _ in bm.top(text, k=20, min_score=0.1)]
    else:
        top = []
    fl = "\n".join(f"{f.id}: {f.statement} | {f.confidence:.1f}" for f in top) or "(немає)"
    data = llm.complete_json(FC_SYSTEM, FC_PROMPT.format(facts=fl, text=strip_citations(text)), purpose="fact_vs_interpretation", max_tokens=4000)
    if isinstance(data, dict):
        data = data.get("sentences", [])
    valid = {f.id for f in facts}
    out = []
    for it in data or []:
        if not isinstance(it, dict):
            continue
        cls = it.get("type", "interpretation")
        cls = cls if cls in CLASS_LABELS else "interpretation"
        a = SentenceAnalysis(
            sentence=str(it.get("sentence", "")),
            cls=cls,
            reasons=[str(it.get("reason", ""))] if it.get("reason") else [],
            too_strong=bool(it.get("too_strong")),
            strength_issues=[str(it["issue"])] if it.get("issue") else [],
            supporting_fact_ids=[int(i) for i in it.get("supporting_fact_ids", []) if str(i).isdigit() and int(i) in valid],
            suggestion=str(it.get("suggestion") or ""),
        )
        out.append(a)
    return out


def summary_counts(items: list[SentenceAnalysis]) -> dict[str, int]:
    c = {k: 0 for k in CLASS_LABELS}
    for a in items:
        c[a.cls] = c.get(a.cls, 0) + 1
    c["too_strong"] = sum(1 for a in items if a.too_strong)
    return c
