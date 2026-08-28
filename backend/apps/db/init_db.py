"""Create all tables on startup (idempotent)."""

import json
import logging

from sqlalchemy import inspect, text

from apps.db.database import Base, engine
from apps.models.analysis import AnalysisRecord  # noqa: F401 — ensures model is registered

logger = logging.getLogger(__name__)

# Columns added in Phase 3 (Analytics Readiness Score)
_PHASE3_COLUMNS = [
    ("analytics_readiness_score", "FLOAT NOT NULL DEFAULT 0.0"),
    ("analytics_readiness_status", "VARCHAR(50) NOT NULL DEFAULT ''"),
    ("analytics_readiness_details_json", "TEXT NOT NULL DEFAULT '{}'"),
]

# Columns added in Phase 4 (Smart-City Context)
_PHASE4_COLUMNS = [
    ("context", "VARCHAR(50) NOT NULL DEFAULT 'CCTV Surveillance'"),
    ("context_impacts_json", "TEXT NOT NULL DEFAULT '[]'"),
]

# Columns added in Phase 6 (Issue Explainability)
_PHASE6_COLUMNS = [
    ("issue_explanations_json", "TEXT NOT NULL DEFAULT '[]'"),
]

_DEFAULT_DETAILS = json.dumps({
    "base_quality_score": 0.0,
    "blur_penalty": 0.0,
    "exposure_penalty": 0.0,
    "noise_penalty": 0.0,
    "corruption_penalty": 0.0,
    "information_penalty": 0.0,
})


def _add_missing_columns() -> None:
    """Add any Phase 3 / Phase 4 columns that are missing from an existing table."""
    inspector = inspect(engine)
    if not inspector.has_table("analyses"):
        return

    existing = {col["name"] for col in inspector.get_columns("analyses")}
    all_columns = _PHASE3_COLUMNS + _PHASE4_COLUMNS + _PHASE6_COLUMNS
    with engine.begin() as conn:
        for col_name, col_type in all_columns:
            if col_name not in existing:
                ddl = f"ALTER TABLE analyses ADD COLUMN {col_name} {col_type}"
                logger.info("Running migration: %s", ddl)
                conn.execute(text(ddl))


def _backfill_phase3_records() -> None:
    """Fill in sensible defaults for old records missing Phase 3 details."""
    inspector = inspect(engine)
    if not inspector.has_table("analyses"):
        return

    existing = {col["name"] for col in inspector.get_columns("analyses")}
    if "analytics_readiness_details_json" not in existing:
        return

    with engine.begin() as conn:
        # Update records where details is empty '{}'
        conn.execute(text(
            "UPDATE analyses SET analytics_readiness_details_json = :details "
            "WHERE analytics_readiness_details_json = '{}'"
        ), {"details": _DEFAULT_DETAILS})


def init_db() -> None:
    """Create tables that don't yet exist and ensure Phase 3 columns exist."""
    _add_missing_columns()
    Base.metadata.create_all(bind=engine)
    _backfill_phase3_records()
