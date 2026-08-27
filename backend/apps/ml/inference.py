"""Model inference layer — bridges features → quality decision.

The trained RandomForestRegressor predicts a raw DMOS score from KADID-10K
(range ~1.0 to ~4.93, higher = better quality).  This module:

  1. Builds the 5-element feature vector matching the training order.
  2. Runs model.predict() to get raw DMOS.
  3. Maps DMOS → 0–100 quality score.
  4. Derives a quality label, issues list, summary, and recommendation.
"""

from __future__ import annotations

import time

import numpy as np

from apps.ml.model_loader import get_model, get_feature_names

# ---------------------------------------------------------------------------
# DMOS → quality thresholds
# DMOS range: 1.0 (worst) ~ 4.93 (best)  →  mapped to 0–100
# Quality labels (matching frontend expectations):
#   >= 75  →  ACCEPTABLE
#   >= 45  →  DEGRADED
#   <  45  →  DEFECTIVE
# ---------------------------------------------------------------------------

_THRESHOLD_ACCEPTABLE = 75
_THRESHOLD_DEGRADED = 45

# KADID-10K DMOS range
_DMOS_MIN = 1.0
_DMOS_MAX = 5.0


def _dmos_to_score(dmos: float) -> int:
    """Map raw DMOS prediction to a 0–100 quality score."""
    normalised = (dmos - _DMOS_MIN) / (_DMOS_MAX - _DMOS_MIN) * 100.0
    return int(round(max(0.0, min(100.0, normalised))))


def _label_from_score(score: int) -> str:
    if score >= _THRESHOLD_ACCEPTABLE:
        return "ACCEPTABLE"
    if score >= _THRESHOLD_DEGRADED:
        return "DEGRADED"
    return "DEFECTIVE"


# ---------------------------------------------------------------------------
# Issue detection — rule-based, derived from extracted features
# ---------------------------------------------------------------------------

def _detect_issues(features: dict[str, float]) -> list[dict]:
    """Identify specific quality issues from extracted features."""
    issues: list[dict] = []

    brightness = features.get("brightness", 128)
    contrast = features.get("contrast", 50)
    sharpness = features.get("sharpness", 100)
    saturation = features.get("saturation", 50)
    edge_density = features.get("edge_density", 0.05)

    # Blur / low sharpness
    if sharpness < 50:
        severity = "HIGH" if sharpness < 15 else "MEDIUM"
        conf = min(0.99, 0.6 + (50 - sharpness) / 80)
        issues.append({
            "type": "Low Sharpness",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "The image may contain blur or insufficient detail. "
                "Edge information is below the expected threshold for reliable analysis."
            ),
        })

    # Underexposure
    if brightness < 60:
        severity = "HIGH" if brightness < 30 else "MEDIUM"
        conf = min(0.99, 0.6 + (60 - brightness) / 80)
        issues.append({
            "type": "Low Brightness",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "The image appears darker than expected. "
                "Key visual features in shadowed areas may be lost."
            ),
        })

    # Overexposure
    if brightness > 200:
        severity = "HIGH" if brightness > 230 else "MEDIUM"
        conf = min(0.99, 0.6 + (brightness - 200) / 60)
        issues.append({
            "type": "High Brightness",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "The image may be overexposed. "
                "Highlight clipping may have occurred."
            ),
        })

    # Low contrast
    if contrast < 25:
        severity = "HIGH" if contrast < 12 else "MEDIUM"
        conf = min(0.99, 0.6 + (25 - contrast) / 40)
        issues.append({
            "type": "Low Contrast",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "The image has limited separation between light and dark regions. "
                "Dynamic range may be too compressed for reliable analysis."
            ),
        })

    # Low saturation
    if saturation < 15:
        conf = min(0.99, 0.5 + (15 - saturation) / 30)
        issues.append({
            "type": "Low Saturation",
            "severity": "LOW",
            "confidence": round(conf, 2),
            "explanation": (
                "Colors appear relatively muted. "
                "This may affect color-based feature extraction."
            ),
        })

    # High edge density (can indicate noise or overly busy scene)
    if edge_density > 0.25:
        conf = min(0.99, 0.5 + (edge_density - 0.25) / 0.3)
        issues.append({
            "type": "High Edge Density",
            "severity": "LOW",
            "confidence": round(conf, 2),
            "explanation": (
                "The image contains a high amount of visual detail or texture. "
                "This may indicate noise or an overly complex scene."
            ),
        })

    return issues


# ---------------------------------------------------------------------------
# Summary and recommendation
# ---------------------------------------------------------------------------

def _recommendation(label: str) -> str:
    if label == "ACCEPTABLE":
        return "Safe for downstream analysis."
    if label == "DEGRADED":
        return "Use with caution or consider image enhancement."
    return "Recapture the image or perform manual review before downstream use."


def _summary(label: str, issues: list[dict], score: int) -> str:
    if label == "ACCEPTABLE" and not issues:
        return (
            "Image quality is suitable for downstream visual analysis. "
            "The image meets all quality thresholds for automated processing."
        )
    if label == "ACCEPTABLE":
        return (
            f"The image demonstrates acceptable overall visual quality (score: {score}/100). "
            "Minor issues were detected but are unlikely to significantly affect "
            "downstream computer vision analysis."
        )
    if label == "DEGRADED":
        if issues:
            types = ", ".join(i["type"] for i in issues)
            return (
                f"The image exhibits degraded quality ({types}). "
                "These conditions may reduce the reliability of downstream analysis. "
                "Manual review or image enhancement is recommended."
            )
        return (
            f"The image quality is below the acceptable threshold (score: {score}/100). "
            "While no specific defects were detected, the overall quality may "
            "reduce the reliability of downstream analysis. "
            "Manual review or image enhancement is recommended."
        )
    return (
        f"Significant visual degradation detected (score: {score}/100). "
        "Image quality is critically low and unsuitable for automated analysis. "
        "Recapture or manual review is required."
    )


# ---------------------------------------------------------------------------
# Confidence estimation via tree variance
# ---------------------------------------------------------------------------

def _estimate_confidence(model, vector_scaled: np.ndarray) -> float:
    """Estimate confidence from individual tree predictions."""
    try:
        if hasattr(model, "estimators_"):
            tree_preds = np.array([
                tree.predict(vector_scaled)[0]
                for tree in model.estimators_
            ])
            tree_preds = np.clip(tree_preds, _DMOS_MIN, _DMOS_MAX)
            std = float(np.std(tree_preds))
            # Low std → high confidence.  std of ~0.3 → ~0.95, std of ~1.5 → ~0.65
            return round(max(0.5, min(0.99, 1.0 - std / 3.0)), 3)
    except Exception:
        pass
    return 0.75


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def predict_quality(model_features: dict[str, float], all_features: dict[str, float]) -> dict:
    """Run inference and return the full analysis result.

    Parameters
    ----------
    model_features : dict with the 5 keys matching training
        (brightness, contrast, sharpness, saturation, edge_density).
    all_features : dict with all 12 statistics for UI display.
    """
    start = time.perf_counter()
    model = get_model()
    feature_names = get_feature_names()

    if model is not None and feature_names:
        # ---- ML model path ----
        vector = np.array([[model_features.get(k, 0.0) for k in feature_names]])
        raw_dmos = float(model.predict(vector)[0])
        score = _dmos_to_score(raw_dmos)
        confidence = _estimate_confidence(model, vector)
    else:
        # ---- Rule-based fallback ----
        score = _fallback_score(model_features)
        confidence = 0.70
        raw_dmos = _DMOS_MIN + score / 100.0 * (_DMOS_MAX - _DMOS_MIN)

    label = _label_from_score(score)
    issues = _detect_issues(model_features)
    rec = _recommendation(label)
    summ = _summary(label, issues, score)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "quality_score": score,
        "quality_label": label,
        "analysis_confidence": round(confidence * 100, 1),
        "issues": issues,
        "statistics": all_features,
        "summary": summ,
        "recommendation": rec,
        "processing_time_ms": elapsed_ms,
    }


def _fallback_score(features: dict[str, float]) -> int:
    """Simple weighted score when no model is loaded."""
    weights = {
        "sharpness": 0.30,
        "brightness": 0.15,
        "contrast": 0.15,
        "saturation": 0.10,
        "edge_density": 0.30,
    }
    ranges: dict[str, tuple[float, float]] = {
        "sharpness": (0, 300),
        "brightness": (0, 255),
        "contrast": (0, 100),
        "saturation": (0, 150),
        "edge_density": (0, 0.5),
    }
    total = 0.0
    for key, weight in weights.items():
        lo, hi = ranges[key]
        val = features.get(key, lo)
        normed = max(0.0, min(1.0, (val - lo) / (hi - lo) if hi > lo else 0.0))
        total += normed * weight
    return int(round(total * 100))
