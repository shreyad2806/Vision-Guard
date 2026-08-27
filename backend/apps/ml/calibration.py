"""Calibration: map raw ML predictions to calibrated 0-100 quality scores.

Uses IsotonicRegression fitted on validation data to ensure monotonic mapping
from raw prediction to final quality score.  Falls back to quantile-based
calibration when IsotonicRegression is unavailable.

Quality bands:
  80-100 → Excellent
  60-79  → Good
  40-59  → Fair
  20-39  → Poor
  0-19   → Critical
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Quality bands
# ---------------------------------------------------------------------------
QUALITY_BANDS: dict[str, dict[str, Any]] = {
    "excellent": {
        "min": 80,
        "label": "Excellent",
        "status": "excellent",
        "recommendation": "Image quality is suitable for automated analysis.",
    },
    "good": {
        "min": 60,
        "label": "Good",
        "status": "good",
        "recommendation": "Image quality is generally reliable, with minor degradation.",
    },
    "fair": {
        "min": 40,
        "label": "Fair",
        "status": "fair",
        "recommendation": "Image quality is usable but some degradation may affect downstream analysis.",
    },
    "poor": {
        "min": 20,
        "label": "Poor",
        "status": "poor",
        "recommendation": "Image quality is significantly degraded. Review or recapture is recommended.",
    },
    "critical": {
        "min": 0,
        "label": "Critical",
        "status": "critical",
        "recommendation": "Image quality is critically degraded and unreliable for automated analysis. Recapture or manual review is required.",
    },
}

_BAND_ORDER = ["excellent", "good", "fair", "poor", "critical"]


# ---------------------------------------------------------------------------
# Quality label / assessment helpers
# ---------------------------------------------------------------------------
def get_quality_assessment(score: float) -> dict[str, str]:
    """Return quality label and status for a 0-100 score."""
    for band_key in _BAND_ORDER:
        band = QUALITY_BANDS[band_key]
        if score >= band["min"]:
            return {"label": band["label"], "status": band["status"]}
    return {"label": "Critical", "status": "critical"}


def get_quality_recommendation(score: float) -> str:
    """Return the recommendation text for a 0-100 score."""
    for band_key in _BAND_ORDER:
        band = QUALITY_BANDS[band_key]
        if score >= band["min"]:
            return band["recommendation"]
    return QUALITY_BANDS["critical"]["recommendation"]


def get_quality_summary(
    score: float, assessment: dict[str, str], issues: list[dict]
) -> str:
    """Generate a human-readable quality summary."""
    label = assessment["label"]
    issue_names = ", ".join(i.get("title", i.get("type", "")) for i in issues)

    if label == "Excellent" and not issues:
        return (
            "Image quality is excellent with no significant issues detected. "
            "The image is well-suited for all downstream analysis tasks."
        )
    if label == "Excellent":
        return (
            f"The image demonstrates excellent overall quality (score: {score}/100). "
            "Minor issues were noted but do not materially affect quality assessment."
        )
    if label == "Good" and not issues:
        return (
            f"Image quality is good (score: {score}/100). "
            "The image meets quality standards for automated analysis."
        )
    if label == "Good":
        return (
            f"The image demonstrates good overall quality (score: {score}/100). "
            f"Minor conditions detected ({issue_names}) but are unlikely to significantly "
            "affect downstream analysis."
        )
    if label == "Fair":
        if issues:
            return (
                f"The image exhibits moderate quality issues ({issue_names}) "
                f"with a quality score of {score}/100. "
                "These conditions may reduce the reliability of downstream analysis. "
                "Consider image enhancement or manual review."
            )
        return (
            f"Image quality is moderate (score: {score}/100). "
            "While no specific defects were detected, the overall quality may "
            "reduce the reliability of downstream analysis."
        )
    if label == "Poor":
        if issues:
            return (
                f"Significant quality degradation detected ({issue_names}) "
                f"with a quality score of {score}/100. "
                "Image quality is substantially compromised. "
                "Review or recapture is recommended."
            )
        return (
            f"Image quality is poor (score: {score}/100). "
            "Review or recapture is recommended."
        )
    return (
        f"Image quality is critically degraded (score: {score}/100). "
        "The image is unreliable for automated analysis. "
        "Recapture or manual review is required."
    )


# ---------------------------------------------------------------------------
# Calibration from raw prediction to 0-100 score
# ---------------------------------------------------------------------------
def apply_calibration(
    raw_prediction: float,
    calibrator: object | None = None,
    calibration_meta: dict | None = None,
) -> float:
    """Map a raw model prediction to a calibrated 0-100 quality score.

    Uses fitted IsotonicRegression calibrator if available.
    Otherwise falls back to min-max normalization using calibration metadata.

    The mapping is monotonic: higher raw_prediction → higher quality_score.
    """
    if calibrator is not None and hasattr(calibrator, "predict"):
        try:
            arr = np.array([[raw_prediction]])
            calibrated = float(calibrator.predict(arr)[0])
            return max(0.0, min(100.0, round(calibrated, 2)))
        except Exception:
            pass

    # Fallback: min-max normalization using calibration metadata
    if calibration_meta:
        pred_min = calibration_meta.get("pred_min", 0.0)
        pred_max = calibration_meta.get("pred_max", 100.0)
        if pred_max > pred_min:
            normalized = 100.0 * (raw_prediction - pred_min) / (pred_max - pred_min)
            return max(0.0, min(100.0, round(normalized, 2)))

    # Ultimate fallback: clip directly
    return max(0.0, min(100.0, round(float(raw_prediction), 2)))


# ---------------------------------------------------------------------------
# Backward-compatible wrappers (used by old inference.py)
# ---------------------------------------------------------------------------
def normalize_prediction_to_quality_score(
    raw_dmos: float, cal: dict | None = None
) -> float:
    """Map raw prediction to 0-100 score (legacy wrapper)."""
    if cal is None:
        return max(0.0, min(100.0, raw_dmos))
    pred_min = cal.get("pred_min", cal.get("target_percentile_1", 0.0))
    pred_max = cal.get("pred_max", cal.get("target_percentile_99", 100.0))
    if pred_max > pred_min:
        score = 100.0 * (raw_dmos - pred_min) / (pred_max - pred_min)
    else:
        score = 50.0
    return max(0.0, min(100.0, score))


def calibrate_quality_score(
    raw_score: float, cal: dict | None = None
) -> float:
    """Calibrate a 0-100 score (legacy wrapper)."""
    return max(0.0, min(100.0, round(raw_score, 1)))
