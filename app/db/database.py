"""Підключення до SQLite та фабрика сесій."""
from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import DB_PATH, ensure_dirs
from app.db.models import Base

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _enable_fk(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def make_engine(url: str) -> Engine:
    engine = create_engine(url, connect_args={"check_same_thread": False})
    event.listen(engine, "connect", _enable_fk)
    Base.metadata.create_all(engine)
    return engine


def init_db(url: str | None = None) -> sessionmaker:
    """Створює (якщо треба) базу і повертає фабрику сесій. Ідемпотентно."""
    global _engine, _SessionLocal
    if _SessionLocal is not None and url is None:
        return _SessionLocal
    if url is None:
        ensure_dirs()
        url = f"sqlite:///{DB_PATH}"
    _engine = make_engine(url)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _SessionLocal


def get_session() -> Session:
    return init_db()()
