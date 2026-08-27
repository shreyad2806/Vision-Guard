"""Load and manage the trained ML model, calibrator, and metadata.

Artifacts (v2):
  backend/models/
    model.joblib        - fitted regressor or dict with "model" key
    calibrator.joblib   - fitted IsotonicRegression calibrator (optional)
    model_metadata.json - training metadata, metrics, feature stats

Legacy fallback:
    model.joblib may contain {"model", "feature_names", "calibration"} dict
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from apps.core.config import settings

logger = logging.getLogger(__name__)

_model = None
_feature_names: list[str] = []
_calibration: dict | None = None
_calibrator = None
_metadata: dict | None = None
_loaded = False


def _get_model_dir() -> Path:
    return Path(settings.model_path).parent


def _get_model_path() -> Path:
    return Path(settings.model_path)


def load_model() -> bool:
    """Load model artifacts once. Returns True on success."""
    global _model, _feature_names, _calibration, _calibrator, _metadata, _loaded  # noqa: PLW0603

    if _loaded:
        return _model is not None

    model_dir = _get_model_dir()
    model_path = _get_model_path()

    if not model_path.exists():
        logger.info("No trained model found at %s", model_path)
        _loaded = True
        return False

    try:
        import joblib

        raw = joblib.load(str(model_path))

        # v2 format: dict with "model" key
        if isinstance(raw, dict) and "model" in raw:
            _model = raw["model"]
            _feature_names = raw.get("feature_names", [])
            _calibration = raw.get("calibration", None)
        # v1 format: raw model object
        else:
            _model = raw
            _feature_names = []
            _calibration = None

        # Try to load calibrator (v2 artifact)
        calibrator_path = model_dir / "calibrator.joblib"
        if calibrator_path.exists():
            try:
                _calibrator = joblib.load(str(calibrator_path))
                logger.info("Calibrator loaded from %s", calibrator_path)
            except Exception:
                logger.warning("Failed to load calibrator from %s", calibrator_path)
                _calibrator = None

        # Try to load metadata
        metadata_path = model_dir / "model_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    _metadata = json.load(f)
                logger.info("Metadata loaded from %s", metadata_path)
            except Exception:
                logger.warning("Failed to load metadata from %s", metadata_path)
                _metadata = None

        _loaded = True
        logger.info(
            "Model loaded from %s (features=%s, calibrator=%s, metadata=%s)",
            model_path,
            _feature_names,
            "present" if _calibrator else "absent",
            "present" if _metadata else "absent",
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


def get_calibration() -> dict | None:
    """Return calibration metadata from the model artifact, or None."""
    if not _loaded:
        load_model()
    return _calibration


def get_calibrator():
    """Return the fitted calibrator object (IsotonicRegression), or None."""
    if not _loaded:
        load_model()
    return _calibrator


def get_metadata() -> dict | None:
    """Return the full model metadata dict, or None."""
    if not _loaded:
        load_model()
    return _metadata


def is_model_loaded() -> bool:
    if not _loaded:
        load_model()
    return _model is not None


def get_model_version() -> str:
    meta = get_metadata()
    if meta and "model_version" in meta:
        return meta["model_version"]
    return "visionguard-iqa-v2.0" if is_model_loaded() else "feature-baseline-v1"
