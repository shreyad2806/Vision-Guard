"""Load and manage the trained scikit-learn model, scaler, and metadata.

Artifacts are expected at:
  backend/models/quality_model.joblib
  backend/models/feature_scaler.joblib
  backend/models/model_metadata.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from apps.core.config import settings

logger = logging.getLogger(__name__)

_model = None
_scaler = None
_metadata = None
_model_dir: Path | None = None


def _get_model_dir() -> Path:
    """Determine the models directory."""
    global _model_dir
    if _model_dir is None:
        _model_dir = Path(settings.model_path).parent
    return _model_dir


def load_model() -> object | None:
    """Attempt to load the trained model, scaler, and metadata from disk.

    Returns the model object on success, or ``None`` if unavailable.
    """
    global _model, _scaler, _metadata  # noqa: PLW0603

    model_dir = _get_model_dir()
    model_path = model_dir / "quality_model.joblib"
    scaler_path = model_dir / "feature_scaler.joblib"
    metadata_path = model_dir / "model_metadata.json"

    if not model_path.exists():
        logger.info("No trained model found at %s — using fallback.", model_path)
        return None

    try:
        import joblib

        _model = joblib.load(str(model_path))
        logger.info("Model loaded from %s", model_path)

        if scaler_path.exists():
            _scaler = joblib.load(str(scaler_path))
            logger.info("Scaler loaded from %s", scaler_path)

        if metadata_path.exists():
            with open(metadata_path) as f:
                _metadata = json.load(f)
            logger.info(
                "Metadata loaded: model=%s version=%s",
                _metadata.get("model_name"),
                _metadata.get("model_version"),
            )

        return _model
    except Exception:
        logger.exception("Failed to load model from %s", model_path)
        return None


def get_model() -> object | None:
    """Return the cached model instance, loading it on first call."""
    global _model  # noqa: PLW0603
    if _model is None:
        load_model()
    return _model


def get_scaler() -> object | None:
    """Return the cached scaler, loading if needed."""
    global _scaler  # noqa: PLW0603
    if _scaler is None:
        load_model()
    return _scaler


def get_metadata() -> dict | None:
    """Return the cached metadata, loading if needed."""
    global _metadata  # noqa: PLW0603
    if _metadata is None:
        load_model()
    return _metadata


def is_model_loaded() -> bool:
    return get_model() is not None


def get_model_version() -> str:
    """Return the model version string from metadata, or a default."""
    meta = get_metadata()
    if meta:
        return meta.get("model_version", "unknown")
    return "feature-baseline-v1"
