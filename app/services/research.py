"""Дослідження: імпорт джерел, чанкінг, витяг фактів (евристика або LLM)."""
from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import UPLOAD_DIR
from app.db.models import ExtractedFact, Project, SourceChunk, SourceDocument
from app.llm import LLMService
from app.parsers import chunk_text, parse_bytes
from app.services.retrieval import token_overlap
from app.utils import text as T

# ---------------------------------------------------------------- тип джерела

_SOURCE_HINTS = [
    ("legal", r"\bзакон\w*|\bпостанов\w*|\bстаття\s+\d|\bкодекс\w*|\bсуд\w*|\bухвал\w*|\bвирок\w*|\bнаказ\w*|реєстр"),
    ("academic", r"\bdoi\b|\babstract\b|\bанотаці\w*|\bвибірк\w*|\bреспондент\w*|\bметодологі\w*|\bжурнал\w*|\bet al\b|\bдослідник\w*"),
    ("interview", r"\bінтерв'ю\b|\bрозмов\w*|^\s*(питання|відповідь|q|a)\s*:|^\s*—\s"),
    ("report", r"\bзвіт\w*|\breport\b|\bаналітичн\w*|\bмоніторинг\w*|\bрічн\w* звіт"),
    ("media", r"\bповідомля\w*|\bджерело виданн\w*|\bновин\w*|\bжурналіст\w*|\bредакці\w*"),
    ("note", r"\bнотатк\w*|\btodo\b|\bперевірити\b|\bідея\b|\bмої думки\b"),
]


def guess_source_type(filename: str, text: str) -> str:
    name = filename.lower()
    for key, words in (
        ("interview", ["interview", "інтерв"]),
        ("note", ["note", "nota", "notes", "нотат"]),
        ("legal", ["law", "закон", "постанова", "rishennya", "рішення"]),
        ("report", ["report", "звіт"]),
        ("academic", ["paper", "study", "досліджен"]),
    ):
        if any(w in name for w in words):
            return key
    sample = T.normalize(text[:6000]).lower()
    best, best_n = "unknown", 0
    for key, pattern in _SOURCE_HINTS:
        n = len(re.findall(pattern, sample, flags=re.M))
        if n > best_n:
            best, best_n = key, n
    return best if best_n >= 2 else "unknown"


# ---------------------------------------------------------------- імпорт

def ingest_document(
    session: Session,
    project: Project,
    filename: str,
    data: bytes,
    source_type: str | None = None,
    url: str = "",
    save_file: bool = True,
) -> SourceDocument:
    parsed = parse_bytes(filename, data)
    file_path = ""
    if save_file:
        target_dir = UPLOAD_DIR / str(project.id)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / Path(filename).name
        i = 1
        while target.exists():
            target = target_dir / f"{Path(filename).stem}_{i}{Path(filename).suffix}"
            i += 1
        target.write_bytes(data)
        file_path = str(target)
    doc = SourceDocument(
        project_id=project.id,
        filename=filename,
        file_type=parsed.file_type,
        file_path=file_path,
        title=parsed.title,
        source_type=source_type or guess_source_type(filename, parsed.text),
        url=url,
        text=parsed.text,
        char_count=len(parsed.text),
    )
    session.add(doc)
    session.flush()
    for ch in chunk_text(parsed.text, page_offsets=parsed.page_offsets):
        session.add(
            SourceChunk(document_id=doc.id, idx=ch.idx, text=ch.text, start_char=ch.start_char, end_char=ch.end_char, page=ch.page)
        )
    session.commit()
    session.refresh(doc)
    return doc


def add_text_note(session: Session, project: Project, title: str, text: str, source_type: str = "note") -> SourceDocument:
    name = (title.strip() or "Нотатка") + ".txt"
    return ingest_document(session, project, name, text.encode("utf-8"), source_type=source_type, save_file=False)


# ---------------------------------------------------------------- евристичний витяг фактів

_MONTHS = r"січ\w*|лют\w*|берез\w*|квіт\w*|трав\w*|черв\w*|лип\w*|серп\w*|верес\w*|жовт\w*|листопад\w*|груд\w*"
_TYPE_BONUS = {"academic": 0.1, "report": 0.1, "legal": 0.1, "media": 0.0, "interview": -0.05, "note": -0.15, "unknown": -0.05}


def categorize(sentence: str) -> str:
    low = sentence.lower()
    if re.search(r"«[^»]{15,}»|\"[^\"]{15,}\"", sentence) and re.search(r"сказав|сказала|заявив|заявила|каже|пояснив|пояснила|за словами", low):
        return "quote"
    if re.search(r"\bзакон\w*|\bпостанов\w*|\bсуд\w*|\bвирок\w*|\bстатт\w* \d|\bліценз\w*|\bрегулятор\w*", low):
        return "legal"
    if re.search(r"\d+(?:[.,]\d+)?\s*(%|відсот|млн|млрд|тис|грн|дол|\$|€|євро|разів)", low):
        return "number"
    if re.search(r"\b(1[89]|20)\d{2}\b", low) or re.search(_MONTHS, low):
        return "event"
    if re.search(r"механізм|алгоритм|схем\w|працю\w|влаштован|через те|завдяки|тому що|відбуваєт", low):
        return "mechanism"
    words = T.words(sentence)
    if sum(1 for w in words[1:] if w[:1].isupper()) >= 2:
        return "actor"
    if re.search(r"\d", sentence):
        return "number"
    return "context"


_PREFIX_RE = re.compile(r"^\s*(?:[-*•–—]\s+|\d+[.)]\s+|(?:питання|відповідь|q|a|в|о)\s*:\s*)+", flags=re.I)


def clean_statement(sentence: str) -> str:
    """Прибирає маркери списків і префікси «Питання:/Відповідь:» з початку речення."""
    s = _PREFIX_RE.sub("", sentence).strip()
    return s[:1].upper() + s[1:] if s else s


def score_sentence(sentence: str) -> float:
    low = sentence.lower()
    n_words = len(T.words(sentence))
    if n_words < 6 or n_words > 60:
        return 0.0
    s = 0.0
    if re.search(r"\d", sentence):
        s += 2.0
    if re.search(r"\b(1[89]|20)\d{2}\b", sentence) or re.search(_MONTHS, low):
        s += 1.0
    proper = sum(1 for w in T.words(sentence)[1:] if w[:1].isupper() and not w.isupper())
    s += min(proper, 3) * 0.5
    if T.count_phrases(low, T.REPORTING_VERBS):
        s += 1.0
    if 10 <= n_words <= 40:
        s += 0.5
    if sentence.rstrip().endswith("?"):
        s -= 2.5
    if T.count_phrases(low, ["я думаю", "мені здається", "на мою думку", "можливо", "мабуть", "перевірити"]):
        s -= 1.0
    return s


def heuristic_facts(doc: SourceDocument, max_facts: int = 25, min_score: float = 2.0) -> list[dict]:
    """Повертає список кандидатів у факти: statement, snippet, category, confidence, chunk_id."""
    candidates = []
    for chunk in doc.chunks:
        sents = T.split_sentences(chunk.text)
        for i, s in enumerate(sents):
            s = clean_statement(s)
            sc = score_sentence(s)
            if sc < min_score:
                continue
            ctx = " ".join(sents[max(0, i - 1) : i + 2])
            conf = max(0.2, min(0.9, 0.25 + sc * 0.1 + _TYPE_BONUS.get(doc.source_type, 0)))
            candidates.append(
                {
                    "statement": s.strip(),
                    "snippet": T.truncate(ctx, 600),
                    "category": categorize(s),
                    "confidence": round(conf, 2),
                    "chunk_id": chunk.id,
                    "score": sc,
                }
            )
    candidates.sort(key=lambda c: -c["score"])
    # прибираємо майже дублікати
    result: list[dict] = []
    for c in candidates:
        if any(token_overlap(c["statement"], r["statement"]) > 0.8 for r in result):
            continue
        result.append(c)
        if len(result) >= max_facts:
            break
    return result


# ---------------------------------------------------------------- LLM-витяг

FACT_SYSTEM = (
    "Ти — дослідник-фактчекер для українського документального YouTube-каналу. "
    "Витягуй із джерела лише перевірювані твердження (факти, цифри, дати, дії акторів, механізми, цитати). "
    "Не додавай нічого, чого немає в тексті. Не посилюй формулювання. Відповідай ЛИШЕ валідним JSON."
)

FACT_PROMPT = """Джерело: «{title}» (тип: {stype}).

Текст фрагмента:
<<<
{text}
>>>

Поверни JSON-масив до {limit} елементів такого вигляду:
[{{"statement": "стисле самодостатнє твердження українською (≤ 35 слів)",
   "snippet": "ДОСЛІВНА цитата з тексту, що підтверджує твердження (≤ 60 слів)",
   "category": "number|event|actor|mechanism|quote|legal|context|other",
   "confidence": 0.0-1.0  // наскільки впевнено джерело це стверджує (оцінка, припущення → нижче)
}}]
Якщо корисних фактів немає — поверни []."""


def _batches(chunks: list[SourceChunk], max_chars: int = 6000) -> list[list[SourceChunk]]:
    out: list[list[SourceChunk]] = []
    cur: list[SourceChunk] = []
    size = 0
    for c in chunks:
        if cur and size + len(c.text) > max_chars:
            out.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c.text)
    if cur:
        out.append(cur)
    return out


def _locate_chunk(snippet: str, chunks: list[SourceChunk]) -> SourceChunk | None:
    probe = " ".join(snippet.split())[:80].lower()
    for c in chunks:
        if probe and probe in " ".join(c.text.split()).lower():
            return c
    best = max(chunks, key=lambda c: token_overlap(snippet, c.text), default=None)
    return best


def llm_facts(llm: LLMService, doc: SourceDocument, per_batch: int = 12, max_batches: int = 8) -> list[dict]:
    result: list[dict] = []
    for batch in _batches(doc.chunks)[:max_batches]:
        text = "\n\n".join(c.text for c in batch)
        data = llm.complete_json(
            FACT_SYSTEM,
            FACT_PROMPT.format(title=doc.title or doc.filename, stype=doc.source_type, text=text, limit=per_batch),
            purpose="extract_facts",
        )
        if isinstance(data, dict):
            data = data.get("facts", [])
        for item in data or []:
            if not isinstance(item, dict) or not item.get("statement"):
                continue
            snippet = str(item.get("snippet") or "")
            ch = _locate_chunk(snippet or item["statement"], batch)
            try:
                conf = float(item.get("confidence", 0.6))
            except (TypeError, ValueError):
                conf = 0.6
            cat = item.get("category", "other")
            result.append(
                {
                    "statement": str(item["statement"]).strip(),
                    "snippet": snippet.strip() or (ch.text[:400] if ch else ""),
                    "category": cat if cat in {"number", "event", "actor", "mechanism", "quote", "legal", "context", "other"} else "other",
                    "confidence": round(max(0.0, min(1.0, conf)), 2),
                    "chunk_id": ch.id if ch else None,
                }
            )
    return result


def extract_facts(
    session: Session, project: Project, doc: SourceDocument, llm: LLMService | None = None, replace: bool = True
) -> tuple[list[ExtractedFact], str]:
    """Витягує факти з документа. Повертає (факти, метод)."""
    method = "llm" if llm is not None and llm.available else "heuristic"
    items = llm_facts(llm, doc) if method == "llm" else heuristic_facts(doc)
    if replace:
        for f in list(doc.facts):
            if f.method != "manual" and not f.used_in_script and not f.starred:
                session.delete(f)
        session.flush()
    existing = [f.statement for f in doc.facts]
    created = []
    for it in items:
        if any(token_overlap(it["statement"], e) > 0.85 for e in existing):
            continue
        f = ExtractedFact(
            project_id=project.id,
            document_id=doc.id,
            chunk_id=it.get("chunk_id"),
            statement=it["statement"],
            snippet=it["snippet"],
            confidence=it["confidence"],
            category=it["category"],
            method=method,
        )
        session.add(f)
        created.append(f)
        existing.append(it["statement"])
    session.commit()
    return created, method


def confidence_label(c: float) -> str:
    if c >= 0.75:
        return "висока"
    if c >= 0.5:
        return "середня"
    return "низька"
