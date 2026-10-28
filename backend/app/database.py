"""SQLAlchemy engine, session factory and declarative base for QuantEdge."""
import os
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATABASE_URL = os.environ.get(
    "QUANTEDGE_DATABASE_URL", f"sqlite:///{(BACKEND_DIR / 'quantedge.db').as_posix()}"
)


class Base(DeclarativeBase):
    pass


def enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
    """SQLite ignores foreign keys unless asked on every connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(url: str = DATABASE_URL, **kwargs) -> Engine:
    engine = create_engine(url, connect_args={"check_same_thread": False}, **kwargs)
    event.listen(engine, "connect", enable_sqlite_fk)
    return engine


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session that is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(bind: Engine | None = None) -> None:
    """Create all tables (importing app.models registers them on Base)."""
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=bind or engine)
