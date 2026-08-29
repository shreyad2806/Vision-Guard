"""SQLAlchemy database engine and session factory."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from apps.core.config import settings


def _ensure_sqlite_dir(url: str) -> None:
    """Create the parent directory for an SQLite database URL if needed."""
    if url.startswith("sqlite:"):
        # sqlite:///relative/path.db or sqlite:////absolute/path.db
        path_part = url.split("sqlite:///", 1)[-1]
        db_path = Path(path_part)
        db_path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_dir(settings.database_url)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # SQLite-specific
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


def get_db():
    """FastAPI dependency that yields a DB session and closes it after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
