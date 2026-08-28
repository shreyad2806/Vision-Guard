"""Model inference layer: calibrated features -> quality decision.

Pipeline:
  1. Build feature vector from extracted features.
  2. Run model.predict() to get raw prediction.
  3. Apply IsotonicRegression calibrator to get calibrated 0-100 score.
  4. Clip to [0, 100].
  5. Derive quality label, issues, explanations, and recommendation.
"""

from __future__ import annotations

import time

import numpy as np

from typing import Any, Protocol

class PredictorModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

from apps.ml.model_loader import get_model, get_feature_names, get_calibrator, get_metadata, get_calibration
from apps.ml.calibration import (
    apply_calibration,
    get_quality_assessment,
    get_quality_recommendation,
    get_quality_summary,
)
from apps.ml.issue_detector import detect_issues

from apps.ml.readiness import calculate_analytics_readiness
from apps.ml.context_definitions import get_context_impacts
from apps.ml.explainability import generate_issue_explanations


# ---------------------------------------------------------------------------
# Confidence estimation via tree variance
# ---------------------------------------------------------------------------
def _estimate_confidence(model: PredictorModel, vector: np.ndarray) -> float:
    """Estimate confidence from individual tree predictions.

    Low disagreement among trees -> high confidence.
    Returns confidence in 0-100 scale.
    """
    try:
        if hasattr(model, "estimators_"):
            # Cast estimators to ensure tree.predict is type-safe
            from typing import cast
            estimators = cast(list[PredictorModel], getattr(model, "estimators_", []))
            tree_preds = np.array([
                float(tree.predict(vector)[0])
                for tree in estimators
            ])
            std = float(np.std(tree_preds))
            # Map std to confidence: std ~0 → 100, std ~5 → 50
            confidence = max(50.0, min(99.0, 100.0 - std * 10.0))
            return round(confidence, 1)
    except Exception:
        pass
    return 75.0


# ---------------------------------------------------------------------------
# Rule-based fallback when no model is loaded
# ---------------------------------------------------------------------------
def _fallback_score(features: dict[str, float]) -> float:
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
    return total * 100.0


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------
def predict_quality(
    model_features: dict[str, float],
    all_features: dict[str, float],
    context: str = "CCTV Surveillance",
) -> dict:
    """Run calibrated inference and return the full analysis result.

    Parameters
    ----------
    model_features : dict with the keys matching training order
        (brightness, contrast, sharpness, saturation, edge_density).
    all_features : dict with all 12+ statistics for UI display.
    context : pipeline context for smart-city analytics.

    Returns
    -------
    dict with quality_score, quality_label, raw_prediction, issues,
    metrics, explanations, recommendation, confidence, processing_time_ms, etc.
    """
    from typing import cast
    start = time.perf_counter()
    raw_model = get_model()
    model = cast(PredictorModel, raw_model) if raw_model is not None else None
    feature_names = get_feature_names()
    calibrator = get_calibrator()
    metadata = get_metadata()
    calibration_meta = get_calibration()

    raw_prediction = None

    if model is not None and feature_names:
        # ---- ML model path ----
        vector = np.array([[model_features.get(k, 0.0) for k in feature_names]])

        # Guard against NaN/Inf in features
        if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        raw_prediction = float(model.predict(vector)[0])

        # Guard against NaN prediction
        if np.isnan(raw_prediction) or np.isinf(raw_prediction):
            raw_prediction = 50.0

        confidence = _estimate_confidence(model, vector)
    else:
        # ---- Rule-based fallback ----
        raw_prediction = _fallback_score(model_features)
        confidence = 70.0

    # --- Step 1: Apply calibration (raw prediction → 0-100) ---
    # Build calibration meta for the calibrator
    cal_meta = None
    if calibration_meta:
        cal_meta = {
            "pred_min": calibration_meta.get("pred_min", 0.0),
            "pred_max": calibration_meta.get("pred_max", 100.0),
        }
    elif metadata and "calibration" in metadata:
        cal_meta = metadata["calibration"]

    quality_score = apply_calibration(raw_prediction, calibrator, cal_meta)

    # --- Step 2: Quality assessment ---
    assessment = get_quality_assessment(quality_score)

    # --- Step 3: Detect issues from features ---
    feature_stats = None
    if metadata and "feature_statistics" in metadata:
        feature_stats = {"feature_thresholds": _build_thresholds_from_stats(metadata["feature_statistics"])}
    issues = detect_issues(all_features)
    
    readiness = calculate_analytics_readiness(
        quality_score=quality_score,
        issues=issues,
    )

    # --- Step 4: Generate explanations ---
    explanations = _generate_explanations(all_features, feature_stats)

    # --- Step 5: Recommendation and summary ---
    recommendation = get_quality_recommendation(quality_score)
    summary = get_quality_summary(quality_score, assessment, issues)

    # --- Step 6: Context-specific impacts ---
    context_impacts = get_context_impacts(issues, context)

    # --- Step 7: Issue-driven explainability (Phase 6) ---
    issue_explanations = generate_issue_explanations(issues)

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "quality_score": int(round(quality_score)),
        "quality_label": assessment["label"],
        "quality_assessment": assessment,
        "analysis_confidence": round(confidence, 1),
        "raw_prediction": round(raw_prediction, 4),
        "issues": issues,
        "metrics": all_features,
        "statistics": all_features,
        "explanations": explanations,
        "summary": summary,
        "recommendation": recommendation,
        "processing_time_ms": elapsed_ms,
        "analytics_readiness_score": readiness["score"],
        "analytics_readiness_status": readiness["status"],
        "analytics_readiness_details": readiness,
        "context": context,
        "context_impacts": context_impacts,
        "issue_explanations": issue_explanations,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_thresholds_from_stats(feature_statistics: dict) -> dict:
    """Build issue detection thresholds from training feature statistics."""
    thresholds = {}
    for fname, stats in feature_statistics.items():
        entry = {}
        if "p5" in stats:
            entry["p5"] = stats["p5"]
        if "p10" in stats:
            entry["p10"] = stats["p10"]
        else:
            # Compute from available data
            if "min" in stats and "max" in stats:
                entry["p10"] = stats.get("p10", stats.get("min", 0))
        if "p90" in stats:
            entry["p90"] = stats["p90"]
        if "p95" in stats:
            entry["p95"] = stats["p95"]
        if "p25" in stats:
            entry["optimal_low"] = stats["p25"]
        if "p75" in stats:
            entry["optimal_high"] = stats["p75"]
        thresholds[fname] = entry
    return thresholds


def _generate_explanations(
    features: dict[str, float], feature_stats: dict | None = None
) -> list[str]:
    """Generate per-feature explanations."""
    explanations: list[str] = []
    br = features.get("brightness", 128.0)
    co = features.get("contrast", 50.0)
    sh = features.get("sharpness", 100.0)
    sa = features.get("saturation", 50.0)
    ed = features.get("edge_density", 0.05)

    # Use thresholds from stats or defaults
    bt = {}
    ct_val = 18.0
    st_val = 100.0
    sat_val = 15.0
    et_val = 0.03

    if feature_stats and "feature_thresholds" in feature_stats:
        t = feature_stats["feature_thresholds"]
        bt = t.get("brightness", {})
        ct_val = t.get("contrast", {}).get("p10", 18.0)
        st_val = t.get("sharpness", {}).get("p10", 100.0)
        sat_val = t.get("saturation", {}).get("p10", 15.0)
        et_val = t.get("edge_density", {}).get("p10", 3.0)

    b_low = bt.get("p10", 55.0)
    b_high = bt.get("p90", 190.0)
    b_opt_lo = bt.get("optimal_low", 80.0)
    b_opt_hi = bt.get("optimal_high", 170.0)

    # Brightness
    if br < b_low:
        explanations.append("Brightness is below the calibrated range, indicating underexposure.")
    elif br > b_high:
        explanations.append("Brightness exceeds the calibrated range, indicating overexposure.")
    elif br < b_opt_lo:
        explanations.append("Brightness is somewhat below the optimal range.")
    elif br > b_opt_hi:
        explanations.append("Brightness is somewhat above the optimal range.")
    else:
        explanations.append("Brightness is within the expected range.")

    # Contrast
    if co < ct_val:
        explanations.append("Image contrast is below the calibrated range, limiting separation between light and dark regions.")
    elif co < 38.0:
        explanations.append("Image contrast is moderate but below the optimal range.")
    else:
        explanations.append("Contrast is sufficient for visible detail separation.")

    # Sharpness
    if sh < st_val:
        explanations.append("Sharpness is significantly below the calibrated threshold. The image may be blurry.")
    elif sh < 250.0:
        explanations.append("Sharpness is below the optimal range but some edge detail remains.")
    else:
        explanations.append("Sharpness is sufficient for visible edge detail.")

    # Saturation
    if sa < sat_val:
        explanations.append("Image saturation is well below the calibrated range. Colors are heavily desaturated.")
    elif sa < 35.0:
        explanations.append("Image saturation is moderately below the calibrated range.")
    else:
        explanations.append("Saturation is within the expected range for natural color reproduction.")

    # Edge density
    if ed < et_val:
        explanations.append("Edge density is very low, suggesting limited structural content.")
    elif ed < 9.0:
        explanations.append("Edge density is below the median range for training images.")
    else:
        explanations.append("Edge density is sufficient for structural feature extraction.")

    return explanations
