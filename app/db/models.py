"""ORM-моделі. Одна SQLite-база, всі дані локальні."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


SOURCE_TYPES = {
    "academic": "Академічне",
    "media": "Медіа",
    "interview": "Інтерв'ю",
    "note": "Нотатка",
    "report": "Звіт",
    "legal": "Юридичне / документ",
    "unknown": "Невідомо",
}

FACT_CATEGORIES = {
    "number": "Цифра / статистика",
    "event": "Подія / дата",
    "actor": "Актор / організація",
    "mechanism": "Механізм / як працює",
    "quote": "Цитата",
    "legal": "Норма / рішення",
    "context": "Контекст",
    "other": "Інше",
}

BLOCK_TYPES = {
    "hook": "Хук",
    "context": "Контекст",
    "key_question": "Ключове питання",
    "mechanism": "Механізм",
    "actors": "Актори",
    "ukrainian_angle": "Український кут",
    "evidence": "Докази",
    "counterpoint": "Контраргумент",
    "conclusion": "Висновок",
    "custom": "Довільний блок",
}


class StyleProfile(Base):
    """Профіль стилю автора. Глобальний (не прив'язаний до проєкту), проєкт обирає профіль."""

    __tablename__ = "style_profiles"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    summary: Mapped[str] = mapped_column(Text, default="")
    profile: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    corpus: Mapped[list[StyleCorpusDocument]] = relationship(
        back_populates="profile", cascade="all, delete-orphan"
    )


class StyleCorpusDocument(Base):
    __tablename__ = "style_corpus_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    profile_id: Mapped[int] = mapped_column(ForeignKey("style_profiles.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(300))
    text: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    profile: Mapped[StyleProfile] = relationship(back_populates="corpus")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    topic: Mapped[str] = mapped_column(Text, default="")
    description: Mapped[str] = mapped_column(Text, default="")
    style_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("style_profiles.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    style_profile: Mapped[StyleProfile | None] = relationship()
    documents: Mapped[list[SourceDocument]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    facts: Mapped[list[ExtractedFact]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    outline: Mapped[list[OutlineBlock]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="OutlineBlock.position"
    )
    drafts: Mapped[list[ScriptDraft]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )
    edits: Mapped[list[EditOperation]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class SourceDocument(Base):
    __tablename__ = "source_documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    filename: Mapped[str] = mapped_column(String(300))
    file_type: Mapped[str] = mapped_column(String(10))
    file_path: Mapped[str] = mapped_column(String(500), default="")
    title: Mapped[str] = mapped_column(String(300), default="")
    source_type: Mapped[str] = mapped_column(String(20), default="unknown")
    url: Mapped[str] = mapped_column(String(500), default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    text: Mapped[str] = mapped_column(Text, default="")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    project: Mapped[Project] = relationship(back_populates="documents")
    chunks: Mapped[list[SourceChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", order_by="SourceChunk.idx"
    )
    facts: Mapped[list[ExtractedFact]] = relationship(back_populates="document")


class SourceChunk(Base):
    __tablename__ = "source_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("source_documents.id", ondelete="CASCADE"))
    idx: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    start_char: Mapped[int] = mapped_column(Integer)
    end_char: Mapped[int] = mapped_column(Integer)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)

    document: Mapped[SourceDocument] = relationship(back_populates="chunks")


class ExtractedFact(Base):
    __tablename__ = "extracted_facts"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"), nullable=True
    )
    chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("source_chunks.id", ondelete="SET NULL"), nullable=True
    )
    statement: Mapped[str] = mapped_column(Text)
    snippet: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    category: Mapped[str] = mapped_column(String(30), default="other")
    method: Mapped[str] = mapped_column(String(20), default="heuristic")  # heuristic | llm | manual
    used_in_script: Mapped[bool] = mapped_column(Boolean, default=False)
    starred: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    project: Mapped[Project] = relationship(back_populates="facts")
    document: Mapped[SourceDocument | None] = relationship(back_populates="facts")
    chunk: Mapped[SourceChunk | None] = relationship()


class OutlineBlock(Base):
    __tablename__ = "outline_blocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    position: Mapped[int] = mapped_column(Integer, default=0)
    block_type: Mapped[str] = mapped_column(String(30), default="custom")
    title: Mapped[str] = mapped_column(String(300))
    notes: Mapped[str] = mapped_column(Text, default="")
    target_words: Mapped[int] = mapped_column(Integer, default=250)
    fact_ids: Mapped[list[int]] = mapped_column(JSON, default=list)  # факти, закріплені за блоком

    project: Mapped[Project] = relationship(back_populates="outline")


class ScriptDraft(Base):
    __tablename__ = "script_drafts"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    style_profile_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    project: Mapped[Project] = relationship(back_populates="drafts")
    blocks: Mapped[list[DraftBlock]] = relationship(
        back_populates="draft", cascade="all, delete-orphan", order_by="DraftBlock.position"
    )


class DraftBlock(Base):
    """Текст одного блоку сценарію. Кілька варіантів для одного outline-блоку; is_selected — активний."""

    __tablename__ = "draft_blocks"

    id: Mapped[int] = mapped_column(primary_key=True)
    draft_id: Mapped[int] = mapped_column(ForeignKey("script_drafts.id", ondelete="CASCADE"))
    outline_block_id: Mapped[int | None] = mapped_column(
        ForeignKey("outline_blocks.id", ondelete="SET NULL"), nullable=True
    )
    position: Mapped[int] = mapped_column(Integer, default=0)
    title: Mapped[str] = mapped_column(String(300), default="")
    text: Mapped[str] = mapped_column(Text, default="")
    used_fact_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    variant: Mapped[int] = mapped_column(Integer, default=1)
    is_selected: Mapped[bool] = mapped_column(Boolean, default=True)
    method: Mapped[str] = mapped_column(String(20), default="llm")  # llm | template | manual
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    draft: Mapped[ScriptDraft] = relationship(back_populates="blocks")


class EditOperation(Base):
    __tablename__ = "edit_operations"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"))
    draft_block_id: Mapped[int | None] = mapped_column(
        ForeignKey("draft_blocks.id", ondelete="SET NULL"), nullable=True
    )
    operation: Mapped[str] = mapped_column(String(50))
    input_text: Mapped[str] = mapped_column(Text)
    output_text: Mapped[str] = mapped_column(Text, default="")
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    project: Mapped[Project] = relationship(back_populates="edits")


class LLMRunLog(Base):
    __tablename__ = "llm_run_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provider: Mapped[str] = mapped_column(String(30))
    model: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str] = mapped_column(String(80))
    prompt_chars: Mapped[int] = mapped_column(Integer, default=0)
    response_chars: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
