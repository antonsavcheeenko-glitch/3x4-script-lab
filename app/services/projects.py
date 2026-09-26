"""Проєкти: CRUD + демо-проєкт."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import SAMPLES_DIR
from app.db.models import Project, StyleProfile
from app.services import outline as outline_svc
from app.services import research as research_svc
from app.services import style_service

DEMO_TITLE = "ДЕМО: Як онлайн-казино утримують гравців"
DEMO_PROFILE = "ДЕМО-профіль автора"


def list_projects(session: Session) -> list[Project]:
    return list(session.scalars(select(Project).order_by(Project.updated_at.desc())))


def get_project(session: Session, project_id: int) -> Project | None:
    return session.get(Project, project_id)


def create_project(session: Session, title: str, topic: str = "", description: str = "", style_profile_id: int | None = None) -> Project:
    if not title.strip():
        raise ValueError("Назва проєкту обов'язкова")
    p = Project(title=title.strip(), topic=topic.strip(), description=description.strip(), style_profile_id=style_profile_id)
    session.add(p)
    session.commit()
    return p


def update_project(session: Session, project: Project, **fields) -> Project:
    for k, v in fields.items():
        setattr(project, k, v)
    session.commit()
    return project


def delete_project(session: Session, project: Project) -> None:
    session.delete(project)
    session.commit()


def list_profiles(session: Session) -> list[StyleProfile]:
    return list(session.scalars(select(StyleProfile).order_by(StyleProfile.updated_at.desc())))


def load_demo(session: Session) -> Project:
    """Створює демо-профіль стилю (3 сценарії) і демо-проєкт (3 джерела, факти, структура)."""
    profile = session.scalars(select(StyleProfile).where(StyleProfile.name == DEMO_PROFILE)).first()
    if profile is None:
        profile = style_service.create_profile(session, DEMO_PROFILE)
        for f in sorted((SAMPLES_DIR / "style").glob("*.md")):
            style_service.add_corpus_file(session, profile, f.name, f.read_bytes())
        style_service.rebuild_profile(session, profile)

    project = create_project(
        session,
        DEMO_TITLE,
        topic="Механіки утримання гравців в онлайн-казино та як їх регулюють в Україні",
        description="Демо-проєкт на вигаданих джерелах: показує повний цикл від дослідження до чернетки.",
        style_profile_id=profile.id,
    )
    for f in sorted((SAMPLES_DIR / "research").iterdir()):
        doc = research_svc.ingest_document(session, project, f.name, f.read_bytes(), save_file=False)
        research_svc.extract_facts(session, project, doc, llm=None)
    session.refresh(project)
    outline_svc.apply_template(session, project)
    return project
