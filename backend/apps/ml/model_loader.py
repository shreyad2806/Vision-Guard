"""Load and manage the trained scikit-learn model.

Artifacts:
  backend/models/model.joblib  — dict with keys "model" and "feature_names"
"""

from __future__ import annotations

import logging
from pathlib import Path

from apps.core.config import settings

logger = logging.getLogger(__name__)

_model = None
_feature_names: list[str] = []
_loaded = False


def _get_model_path() -> Path:
    return Path(settings.model_path)


def load_model() -> bool:
    """Load model.joblib once. Returns True on success."""
    global _model, _feature_names, _loaded  # noqa: PLW0603

    if _loaded:
        return _model is not None

    model_path = _get_model_path()
    if not model_path.exists():
        logger.info("No trained model found at %s", model_path)
        _loaded = True
        return False

    try:
        import joblib

        raw = joblib.load(str(model_path))

        if isinstance(raw, dict) and "model" in raw:
            _model = raw["model"]
            _feature_names = raw.get("feature_names", [])
        else:
            # Fallback: treat the loaded object as the model directly
            _model = raw
            _feature_names = []

        _loaded = True
        logger.info(
            "Model loaded from %s (features=%s)", model_path, _feature_names
        )
        return True

    except Exception:
        logger.exception("Failed to load model from %s", model_path)
        _loaded = True
        return False


def get_model() -> object | None:
    if not _loaded:
        load_model()
    return _model


def get_feature_names() -> list[str]:
    if not _loaded:
        load_model()
    return list(_feature_names)


def is_model_loaded() -> bool:
    if not _loaded:
        load_model()
    return _model is not None


def get_model_version() -> str:
    return "visionguard-iqa-v1" if is_model_loaded() else "feature-baseline-v1"
