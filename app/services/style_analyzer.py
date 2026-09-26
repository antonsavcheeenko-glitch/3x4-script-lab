"""Аналіз стилю автора за корпусом попередніх сценаріїв.

Pipeline:
1. абзаци → речення → слова;
2. описова статистика (довжини речень/абзаців, розподіл);
3. n-грами, типові початки речень, лексика;
4. маркери присутності автора, риторичні питання, переходи, кліше;
5. евристика «абстрактне vs конкретне» (суфікси абстрактних іменників, цифри, власні назви);
6. людський опис (summary, tone, do/avoid) + машинний профіль (dict → JSON / prompt).
"""
from __future__ import annotations

import re
import statistics
from collections import Counter

from app.utils import text as T

PROFILE_VERSION = 1


# ---------------------------------------------------------------- метрики тексту

def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    vs = sorted(values)
    k = (len(vs) - 1) * q
    f = int(k)
    c = min(f + 1, len(vs) - 1)
    return vs[f] + (vs[c] - vs[f]) * (k - f)


def text_metrics(text: str) -> dict:
    """Метрики одного тексту (або корпусу, склеєного через \\n\\n)."""
    paragraphs = T.split_paragraphs(text)
    sentences = T.split_sentences(text)
    sent_lens = [len(T.words(s)) for s in sentences]
    sent_lens = [n for n in sent_lens if n > 0]
    all_words = T.words(text)
    lw = [w.lower() for w in all_words]
    n_words = max(len(all_words), 1)
    n_sent = max(len(sent_lens), 1)
    low = T.normalize(text).lower()

    para_sent = [len(T.split_sentences(p)) for p in paragraphs]
    para_words = [len(T.words(p)) for p in paragraphs]

    questions = sum(1 for s in sentences if s.rstrip("»\"”) ").endswith("?"))
    exclaims = sum(1 for s in sentences if s.rstrip("»\"”) ").endswith("!"))

    i_count = sum(1 for w in lw if w in T.AUTHOR_I)
    we_count = sum(1 for w in lw if w in T.AUTHOR_WE)
    you_count = sum(1 for w in lw if w in T.DIRECT_ADDRESS)

    content_words = [w for w in lw if w not in T.STOPWORDS and len(w) > 2 and not w.isdigit()]
    abstract = sum(1 for w in content_words if T.is_abstract(w))
    numbers = len(re.findall(r"\d+(?:[.,]\d+)?", text))
    # власні назви: слово з великої літери не на початку речення
    proper = 0
    for s in sentences:
        ws = T.words(s)
        proper += sum(1 for w in ws[1:] if w[:1].isupper() and not w.isupper())

    cliches = T.count_phrases(low, T.CLICHES)
    transitions = T.count_phrases(low, T.TRANSITIONS)
    emotive = T.count_phrases(low, T.EMOTIVE_WORDS)

    per_k = 1000 / n_words
    # «фраза» — відрізок між будь-якими розділовими знаками (включно з комами): одиниця ритму на слух
    phrases = [len(T.words(x)) for x in re.split(r"[.!?…;:,—–()]+", text)]
    phrases = [n for n in phrases if n > 0]
    run_on = sum(1 for n in sent_lens if n > 60)
    short = sum(1 for n in sent_lens if n <= 8)
    long_ = sum(1 for n in sent_lens if n > 22)
    return {
        "words": len(all_words),
        "sentences": len(sent_lens),
        "paragraphs": len(paragraphs),
        "sentence_len": {
            "mean": round(statistics.mean(sent_lens), 1) if sent_lens else 0,
            "median": round(statistics.median(sent_lens), 1) if sent_lens else 0,
            "p25": round(_pct(sent_lens, 0.25), 1),
            "p75": round(_pct(sent_lens, 0.75), 1),
            "p90": round(_pct(sent_lens, 0.90), 1),
            "stdev": round(statistics.pstdev(sent_lens), 1) if sent_lens else 0,
        },
        "phrase_len": {
            "mean": round(statistics.mean(phrases), 1) if phrases else 0,
            "median": round(statistics.median(phrases), 1) if phrases else 0,
        },
        "run_on_share": round(run_on / n_sent, 3),
        "sentence_mix": {
            "short_share": round(short / n_sent, 3),
            "medium_share": round((n_sent - short - long_) / n_sent, 3),
            "long_share": round(long_ / n_sent, 3),
        },
        "paragraph": {
            "mean_sentences": round(statistics.mean(para_sent), 1) if para_sent else 0,
            "mean_words": round(statistics.mean(para_words), 1) if para_words else 0,
            "p75_words": round(_pct(para_words, 0.75), 1),
        },
        "question_rate": round(questions / n_sent, 3),
        "exclamation_rate": round(exclaims / n_sent, 3),
        "author_i_per_1k": round(i_count * per_k, 2),
        "author_we_per_1k": round(we_count * per_k, 2),
        "address_you_per_1k": round(you_count * per_k, 2),
        "abstract_ratio": round(abstract / max(len(content_words), 1), 3),
        "numbers_per_1k": round(numbers * per_k, 2),
        "proper_nouns_per_1k": round(proper * per_k, 2),
        "cliches": cliches,
        "cliches_per_1k": round(sum(cliches.values()) * per_k, 2),
        "transitions": transitions,
        "emotive": emotive,
        "emotive_per_1k": round(sum(emotive.values()) * per_k, 2),
        "lexical_diversity": round(len(set(lw)) / n_words, 3),
    }


# ---------------------------------------------------------------- n-грами і лексика

def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def top_ngrams(texts: list[str], n: int, limit: int = 20, min_count: int = 2) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in texts:
        for s in T.split_sentences(t):
            toks = T.words_lower(s)
            for g in _ngrams(toks, n):
                # відкидаємо n-грами лише зі стоп-слів і з цифрами
                if all(w in T.STOPWORDS for w in g) or any(w.isdigit() for w in g):
                    continue
                c[" ".join(g)] += 1
    return [(g, k) for g, k in c.most_common(limit * 3) if k >= min_count][:limit]


def recurring_formulas(texts: list[str], n_min: int = 3, n_max: int = 7, limit: int = 15) -> list[tuple[str, int]]:
    """Фірмові формули: повторювані фрази, що трапляються в більшості текстів корпусу.

    1) n-грами (3–7 слів, хоча б одне не стоп-слово), що є в ≥ половині текстів (мінімум 2);
    2) фрагменти, які перекриваються (кінець одного = початок іншого), склеюються в одну довшу фразу,
       якщо склеєна фраза теж трапляється в стількох самих текстах;
    3) вкладені фрагменти прибираються.
    Повертає (фраза, кількість текстів).
    """
    if len(texts) < 2:
        return []
    min_docs = max(2, (len(texts) + 1) // 2)
    docs = [" " + " ".join(T.words_lower(t)) + " " for t in texts]

    def df(phrase: str) -> int:
        return sum(1 for d in docs if f" {phrase} " in d)

    doc_freq: Counter = Counter()
    for d in docs:
        toks = d.split()
        seen = set()
        for n in range(n_min, n_max + 1):
            for g in _ngrams(toks, n):
                if any(w not in T.STOPWORDS for w in g):
                    seen.add(" ".join(g))
        doc_freq.update(seen)
    phrases = {g: k for g, k in doc_freq.items() if k >= min_docs}

    # склеювання фрагментів, що перекриваються
    changed = True
    while changed:
        changed = False
        items = sorted(phrases.items(), key=lambda x: -len(x[0]))
        for a, ka in items:
            if a not in phrases:
                continue
            ta = a.split()
            for b, kb in items:
                if b == a or b not in phrases or kb != ka:
                    continue
                tb = b.split()
                for ov in range(min(len(ta), len(tb)) - 1, 1, -1):
                    if ta[-ov:] == tb[:ov]:
                        merged = " ".join(ta + tb[ov:])
                        k = df(merged)
                        if k >= ka:
                            phrases.pop(a, None)
                            phrases.pop(b, None)
                            phrases[merged] = k
                            changed = True
                        break
                if changed:
                    break
            if changed:
                break
    # прибрати вкладені
    result: list[tuple[str, int]] = []
    for g, k in sorted(phrases.items(), key=lambda x: (-len(x[0]), -x[1])):
        if any(f" {g} " in f" {r} " and k <= rk for r, rk in result):
            continue
        result.append((g, k))
    result.sort(key=lambda x: (-x[1], -len(x[0].split())))
    return result[:limit]


def top_content_words(texts: list[str], limit: int = 30) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in texts:
        c.update(w for w in T.words_lower(t) if w not in T.STOPWORDS and len(w) > 3 and not w.isdigit())
    return c.most_common(limit)


def sentence_starters(texts: list[str], limit: int = 15) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in texts:
        for s in T.split_sentences(t):
            toks = T.words_lower(s)
            if toks:
                c[toks[0] if toks[0] not in T.STOPWORDS or len(toks) == 1 else " ".join(toks[:2])] += 1
    return [(k, v) for k, v in c.most_common(limit) if v >= 2]


# ---------------------------------------------------------------- інтро / переходи / фінали

def classify_intro(first_paragraph: str) -> str:
    low = first_paragraph.lower()
    sents = T.split_sentences(first_paragraph)
    first = sents[0] if sents else first_paragraph
    if any(g in low for g in T.GENERIC_INTROS):
        return "generic"
    if re.search(r"\b(1[89]|20)\d{2}\b", first) or re.search(r"\b(січня|лютого|березня|квітня|травня|червня|липня|серпня|вересня|жовтня|листопада|грудня)\b", first.lower()):
        return "scene_date"
    if re.search(r"\d", first):
        return "number"
    if first.rstrip().endswith("?"):
        return "question"
    if T.words_lower(first)[:1] and T.words_lower(first)[0] in T.AUTHOR_I:
        return "personal"
    if len(T.words(first)) <= 10:
        return "short_statement"
    return "statement"


INTRO_LABELS = {
    "scene_date": "Сцена з конкретною датою/місцем (заходить через подію)",
    "number": "Відкриття цифрою або конкретним фактом",
    "question": "Відкриття питанням",
    "personal": "Особистий вхід від першої особи",
    "short_statement": "Коротке рубане твердження",
    "statement": "Розгорнуте твердження-теза",
    "generic": "Шаблонне привітання/анонс (\"у цьому відео…\")",
}


def classify_ending(last_paragraph: str) -> str:
    low = last_paragraph.lower()
    if re.search(r"підпис|лайк|коментар|дзвіночок|patreon|патреон|підтрима", low):
        return "cta"
    sents = T.split_sentences(last_paragraph)
    last = sents[-1] if sents else last_paragraph
    if last.rstrip().endswith("?"):
        return "open_question"
    if T.starts_with_phrase(low, ["отже", "зрештою", "у підсумку", "підсумуємо", "тож"]) or any(
        m in low for m in ["отже", "у підсумку", "зрештою"]
    ):
        return "summary"
    if len(T.words(last)) <= 8:
        return "punchline"
    return "reflection"


ENDING_LABELS = {
    "cta": "Заклик (підписка/підтримка) наприкінці",
    "open_question": "Відкрите питання, що лишається глядачу",
    "summary": "Стислий підсумок («отже», «зрештою»)",
    "punchline": "Коротка фінальна фраза-панчлайн",
    "reflection": "Спокійна рефлексія / висновок без пафосу",
}


def paragraph_openers(texts: list[str]) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for p in T.split_paragraphs(t)[1:]:
            ph = T.starts_with_phrase(p.lower(), T.TRANSITIONS)
            if ph:
                c[ph] += 1
    return c


# ---------------------------------------------------------------- профіль

def _level(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value < low:
        return labels[0]
    if value > high:
        return labels[2]
    return labels[1]


def build_profile(texts: list[str], name: str = "Мій стиль") -> dict:
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise ValueError("Корпус порожній: завантажте хоча б один сценарій.")
    corpus = "\n\n".join(texts)
    m = text_metrics(corpus)
    per_doc = [text_metrics(t) for t in texts]

    intros = Counter(classify_intro(T.split_paragraphs(t)[0]) for t in texts if T.split_paragraphs(t))
    endings = Counter(classify_ending(T.split_paragraphs(t)[-1]) for t in texts if T.split_paragraphs(t))
    openers = paragraph_openers(texts)

    sl = m["sentence_len"]
    tone_parts = []
    tone_parts.append(
        _level(sl["mean"], 11, 18, ("короткі рубані речення", "речення середньої довжини", "довгі розгорнуті речення"))
    )
    tone_parts.append(_level(m["numbers_per_1k"], 6, 15, ("мало цифр", "помірна кількість цифр", "висока щільність цифр і дат")))
    tone_parts.append(
        _level(m["emotive_per_1k"] + m["exclamation_rate"] * 100, 1.5, 5, ("стримана емоційність", "помірна емоційність", "висока емоційність"))
    )
    tone_parts.append(
        _level(m["author_i_per_1k"], 2, 10, ("автор майже не присутній у тексті", "дозована присутність автора («я»)", "сильна авторська присутність"))
    )
    tone_parts.append(_level(m["question_rate"], 0.04, 0.12, ("мало риторичних питань", "помірно риторичних питань", "багато риторичних питань")))
    tone_parts.append(_level(m["abstract_ratio"], 0.08, 0.15, ("конкретна лексика", "баланс конкретного й абстрактного", "схильність до абстракцій")))

    corpus_cliches = m["cliches"]
    avoid_cliches = [c for c in T.CLICHES if c not in corpus_cliches][:20]

    do: list[str] = []
    avoid: list[str] = []
    do.append(
        f"Типове речення ≈{sl['median']:.0f} слів (медіана; діапазон p25–p75: {sl['p25']:.0f}–{sl['p75']:.0f}). "
        f"Середнє {sl['mean']:.0f} — його підтягують довгі речення."
    )
    if m["sentence_mix"]["short_share"] >= 0.2:
        do.append(f"Регулярно вставляти короткі речення (≤8 слів): у корпусі це {m['sentence_mix']['short_share']*100:.0f}% речень.")
    do.append(f"Абзац ≈{m['paragraph']['mean_sentences']:.0f} речень / ≈{m['paragraph']['mean_words']:.0f} слів.")
    if m["numbers_per_1k"] >= 6:
        do.append(f"Спиратися на цифри, дати, імена: ≈{m['numbers_per_1k']:.0f} числових згадок на 1000 слів.")
    if m["proper_nouns_per_1k"] >= 20:
        do.append("Називати конкретних людей, компанії, місця замість узагальнень.")
    top_trans = [t for t, _ in Counter(m["transitions"]).most_common(6)]
    if top_trans:
        do.append("Типові сполучні слова/переходи: " + ", ".join(f"«{t}»" for t in top_trans) + ".")
    if 0 < m["author_i_per_1k"] <= 10:
        do.append("Дозволяти коротку авторську репліку від першої особи, але не робити її центром тексту.")
    if m["question_rate"] >= 0.03:
        do.append(f"Використовувати риторичні питання дозовано (≈{m['question_rate']*100:.0f}% речень), як «двері» в новий блок.")

    ph = m["phrase_len"]
    do.append(f"Ритм на слух: фраза між розділовими знаками ≈{ph['mean']:.0f} слів (медіана {ph['median']:.0f}).")
    if m["run_on_share"] >= 0.1:
        do.append(
            f"Фірмова риса: довгі речення-ланцюжки через кому ({m['run_on_share']*100:.0f}% речень > 60 слів). "
            "Для читабельності сценарію їх варто ділити, зберігаючи інтонацію ланцюжка."
        )
    else:
        avoid.append(f"Речення довші за {max(sl['p90'], 25):.0f} слів — у корпусі вони рідкісні.")
    if m["cliches_per_1k"] < 1:
        avoid.append("Шаблонні вступи та кліше («варто зазначити», «у сучасному світі», «давайте розберемося»).")
    if m["emotive_per_1k"] < 2:
        avoid.append("Емоційні оцінки без фактів («шок», «жах», «неймовірно»).")
    if m["exclamation_rate"] < 0.03:
        avoid.append("Знаки оклику — автор ними майже не користується.")
    if m["abstract_ratio"] < 0.12:
        avoid.append("Нагромадження абстрактних іменників (-ість, -ння, -ція).")
    avoid.append("Сильніші твердження, ніж дозволяють джерела («всі», «завжди», «доведено»).")

    intro_key = intros.most_common(1)[0][0] if intros else "statement"
    ending_key = endings.most_common(1)[0][0] if endings else "reflection"
    transition_style = (
        "Абзаци часто починаються з маркерів: " + ", ".join(f"«{k}»" for k, _ in openers.most_common(5))
        if openers
        else "Переходи між блоками без явних маркерів — через новий факт або питання."
    )

    profile = {
        "version": PROFILE_VERSION,
        "name": name,
        "corpus": {"documents": len(texts), "words": m["words"], "sentences": m["sentences"]},
        "metrics": m,
        "per_document_sentence_mean": [d["sentence_len"]["mean"] for d in per_doc],
        "tone": tone_parts,
        "formulas": recurring_formulas(texts),
        "lexicon": {
            "top_words": top_content_words(texts, 30),
            "bigrams": top_ngrams(texts, 2, 20),
            "trigrams": top_ngrams(texts, 3, 15),
            "sentence_starters": sentence_starters(texts, 15),
        },
        "transitions": dict(Counter(m["transitions"]).most_common(15)),
        "paragraph_openers": dict(openers.most_common(10)),
        "intro_style": {"key": intro_key, "label": INTRO_LABELS[intro_key], "distribution": dict(intros)},
        "transition_style": transition_style,
        "ending_style": {"key": ending_key, "label": ENDING_LABELS[ending_key], "distribution": dict(endings)},
        "do": do,
        "avoid": avoid,
        "rare_or_absent_cliches": avoid_cliches,
        "gonzo_share": "10–20%",
        "llm_notes": {},
    }
    profile["summary"] = profile_summary(profile)
    return profile


def profile_summary(profile: dict) -> str:
    m = profile["metrics"]
    sl = m["sentence_len"]
    lines = [
        f"Корпус: {profile['corpus']['documents']} текст(и), {profile['corpus']['words']} слів.",
        f"Речення: в середньому {sl['mean']} слів (медіана {sl['median']}, типовий діапазон {sl['p25']}–{sl['p75']}).",
        f"Короткі речення (≤8 слів): {m['sentence_mix']['short_share']*100:.0f}%, довгі (>22): {m['sentence_mix']['long_share']*100:.0f}%.",
        f"Абзац: ≈{m['paragraph']['mean_sentences']} речень, ≈{m['paragraph']['mean_words']} слів.",
        f"Риторичні питання: {m['question_rate']*100:.1f}% речень. «Я/мені»: {m['author_i_per_1k']} на 1000 слів.",
        f"Цифри: {m['numbers_per_1k']} на 1000 слів. Частка абстрактних слів: {m['abstract_ratio']*100:.1f}%.",
        "Тон: " + "; ".join(profile["tone"]) + ".",
    ]
    return "\n".join(lines)


def profile_to_prompt(profile: dict) -> str:
    """Структурований опис стилю для підстановки в промпт генерації."""
    if not profile:
        return "Профіль стилю не задано. Пиши документально, стримано, конкретно, короткими й середніми реченнями."
    m = profile.get("metrics", {})
    sl = m.get("sentence_len", {})
    notes = profile.get("llm_notes") or {}
    parts = [
        "ПРОФІЛЬ СТИЛЮ АВТОРА (виведено з його попередніх сценаріїв):",
        f"- Жанр: документально-аналітичний YouTube-сценарій українською, висока щільність фактів, 10–20% гонзо-енергії.",
        f"- Типова довжина речення (медіана): {sl.get('median', 12)} слів, середня {sl.get('mean', 14)}; діапазон {sl.get('p25', 8)}–{sl.get('p75', 18)}. "
        f"Коротких речень (≤8 слів) ≈{m.get('sentence_mix', {}).get('short_share', 0.25)*100:.0f}%.",
        f"- Абзац: ≈{m.get('paragraph', {}).get('mean_sentences', 3)} речень.",
        f"- Тон: {'; '.join(profile.get('tone', []))}.",
        f"- Типовий вступ: {profile.get('intro_style', {}).get('label', '')}.",
        f"- Переходи між блоками: {profile.get('transition_style', '')}",
        f"- Типовий фінал: {profile.get('ending_style', {}).get('label', '')}.",
        "- РОБИ:",
        *[f"  • {x}" for x in profile.get("do", [])],
        "- УНИКАЙ:",
        *[f"  • {x}" for x in profile.get("avoid", [])],
    ]
    ph = m.get("phrase_len")
    if ph:
        parts.append(f"- Ритм на слух: фраза між розділовими знаками ≈{ph['mean']} слів.")
    formulas = [f for f, _ in profile.get("formulas", [])[:8]]
    if formulas:
        parts.append("- Фірмові формули автора (використовуй у відповідних місцях, не частіше ніж у оригіналі): " + "; ".join(f"«{f}»" for f in formulas))
    starters = [s for s, _ in profile.get("lexicon", {}).get("sentence_starters", [])[:8]]
    if starters:
        parts.append("- Типові початки речень: " + ", ".join(f"«{s}»" for s in starters))
    if profile.get("rare_or_absent_cliches"):
        parts.append("- Фрази, яких автор НЕ використовує: " + ", ".join(f"«{c}»" for c in profile["rare_or_absent_cliches"][:12]))
    if notes.get("voice"):
        parts.append(f"- Голос (якісний опис): {notes['voice']}")
    for key, title in (("do", "Додатково РОБИ"), ("avoid", "Додатково УНИКАЙ")):
        if notes.get(key):
            parts.append(f"- {title}: " + "; ".join(notes[key]))
    if notes.get("sample_sentences"):
        parts.append("- Приклади характерних речень автора: " + " / ".join(notes["sample_sentences"][:5]))
    return "\n".join(parts)
