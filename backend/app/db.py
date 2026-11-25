"""Database models (SQLite via SQLModel)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Column, JSON
from sqlmodel import Field, Session, SQLModel, create_engine

from . import config


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    labels: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    # "assisted" shows the AI tools, "manual" hides them (used for the effort study).
    mode: str = "assisted"
    created_at: datetime = Field(default_factory=utcnow)


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    filename: str  # original file name, used in exports
    stored_name: str  # unique name on disk
    width: int
    height: int
    status: str = "new"  # new | done
    created_at: datetime = Field(default_factory=utcnow)


class Annotation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="image.id", index=True)
    label: str
    # Bounding box in original image pixels (top-left x/y, width, height).
    x: float
    y: float
    w: float
    h: float
    # Optional polygon [[x, y], ...] in image pixels (from the segmenter).
    polygon: Optional[list] = Field(default=None, sa_column=Column(JSON))
    # manual | ai_detect | ai_segment  (where the shape originally came from)
    source: str = "manual"
    score: Optional[float] = None
    # True if the user changed an AI suggestion before keeping it.
    edited: bool = False
    updated_at: datetime = Field(default_factory=utcnow)


class Suggestion(SQLModel, table=True):
    """A detector suggestion computed ahead of time by batch pre-labelling.

    Not an annotation: the editor shows it as a dashed suggestion, and it is
    removed once the image is saved (accepted suggestions become annotations).
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="image.id", index=True)
    label: str
    x: float
    y: float
    w: float
    h: float
    score: float
    model: str = ""


class WorkSession(SQLModel, table=True):
    """One visit to an image in the editor, ending with a save.

    These rows are the raw data for the manual-vs-assisted effort evaluation.
    """

    id: Optional[int] = Field(default=None, primary_key=True)
    image_id: int = Field(foreign_key="image.id", index=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    mode: str
    active_seconds: float
    n_actions: int = 0  # every user interaction that changed the canvas
    n_drawn: int = 0  # boxes drawn by hand
    n_accepted: int = 0  # AI suggestions accepted unchanged
    n_adjusted: int = 0  # AI suggestions accepted after editing
    n_rejected: int = 0  # AI suggestions deleted
    n_relabelled: int = 0
    n_ai_calls: int = 0
    n_final: int = 0  # annotations saved
    created_at: datetime = Field(default_factory=utcnow)


_engine = None


def engine():
    global _engine
    if _engine is None:
        config.ensure_dirs()
        _engine = create_engine(
            f"sqlite:///{config.DB_PATH}", connect_args={"check_same_thread": False}
        )
        SQLModel.metadata.create_all(_engine)
    return _engine


def reset_engine() -> None:
    """Used by tests to point the app at a fresh database."""
    global _engine
    _engine = None


def get_session():
    with Session(engine()) as session:
        yield session
