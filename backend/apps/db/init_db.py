"""Create all tables on startup (idempotent)."""

from apps.db.database import Base, engine
from apps.models.analysis import AnalysisRecord  # noqa: F401 — ensures model is registered


def init_db() -> None:
    """Create tables that don't yet exist."""
    Base.metadata.create_all(bind=engine)
