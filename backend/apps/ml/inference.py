"""Model inference layer — bridges features → quality decision.

This module is the single place where feature vectors are turned into
a quality score, label, issues list, and summary.

When the trained model is loaded, prediction goes through:
  features → scaler.transform() → model.predict() → quality score

Otherwise a transparent rule-based fallback is applied.
"""

from __future__ import annotations

import time

import numpy as np

from apps.ml.model_loader import get_model, get_scaler, get_metadata

# ---- Quality label thresholds ----
_THRESHOLD_ACCEPTABLE = 75   # >= 75
_THRESHOLD_DEGRADED = 45     # >= 45

# ---- Confidence estimation ----

def _estimate_confidence(score: int, features: dict[str, float]) -> float:
    """Estimate confidence using model prediction variance.

    For the trained RandomForest regressor, we estimate uncertainty by
    examining the prediction stability across individual trees.
    Falls back to feature-based heuristic if tree predictions are unavailable.
    """
    model = get_model()
    scaler = get_scaler()
    metadata = get_metadata()

    if model is not None and scaler is not None and metadata is not None:
        try:
            feature_names = metadata.get("feature_names", [])
            vector = np.array([[features.get(k, 0.0) for k in feature_names]])
            vector_scaled = scaler.transform(vector)

            # For RandomForest, get predictions from individual trees
            if hasattr(model, "estimators_"):
                tree_preds = np.array([
                    tree.predict(vector_scaled)[0]
                    for tree in model.estimators_
                ])
                tree_preds = np.clip(tree_preds, 0, 100)
                std = float(np.std(tree_preds))
                # Lower std → higher confidence
                # std of ~5 → ~0.95, std of ~20 → ~0.7
                confidence = max(0.5, min(0.99, 1.0 - std / 40.0))
                return round(confidence, 3)

            # For HistGradientBoosting or other models
            if hasattr(model, "n_estimators"):
                # Use feature distance from training distribution as proxy
                meta = get_metadata()
                tm = meta.get("validation_metrics", {})
                mae = tm.get("mae", 20.0)
                # Convert MAE to confidence: lower MAE → higher confidence
                confidence = max(0.5, min(0.99, 1.0 - mae / 60.0))
                return round(confidence, 3)
        except Exception:
            pass

    # Fallback: heuristic based on how clearly the score falls in a category
    if score >= 85 or score <= 20:
        return 0.92
    if score >= 75 or score <= 30:
        return 0.85
    if score >= 60 or score <= 40:
        return 0.75
    return 0.70


# ---- Issue detection (feature-based) ----

def _detect_issues(features: dict[str, float]) -> list[dict]:
    """Identify specific quality issues from extracted features."""
    issues: list[dict] = []

    sharpness = features.get("sharpness", 0)
    brightness = features.get("brightness", 128)
    contrast = features.get("contrast", 50)
    noise = features.get("noise_estimate", 0)
    entropy = features.get("entropy", 7)
    underexp = features.get("underexposure_pct", 0)
    overexp = features.get("overexposure_pct", 0)

    # Blur detection
    if sharpness < 35:
        severity = "HIGH" if sharpness < 15 else "MEDIUM"
        conf = min(0.99, 0.6 + (35 - sharpness) / 50)
        issues.append({
            "type": "Blur",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "Insufficient edge detail detected. "
                "The image may reduce the reliability of downstream object detection."
            ),
        })

    # Image noise
    if noise > 15:
        severity = "HIGH" if noise > 30 else "MEDIUM"
        conf = min(0.99, 0.5 + noise / 60)
        issues.append({
            "type": "Image Noise",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "Significant image noise detected. "
                "This may degrade feature extraction and analysis accuracy."
            ),
        })

    # Underexposure
    if underexp > 10 or brightness < 50:
        severity = "HIGH" if brightness < 30 or underexp > 30 else "MEDIUM"
        conf = min(0.99, 0.6 + max(60 - brightness, underexp * 2) / 100)
        issues.append({
            "type": "Underexposure",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "Image is underexposed. Key visual features in shadowed areas may be lost."
            ),
        })

    # Overexposure
    if overexp > 10 or brightness > 210:
        severity = "HIGH" if brightness > 230 or overexp > 25 else "MEDIUM"
        conf = min(0.99, 0.6 + max(brightness - 200, overexp * 2) / 60)
        issues.append({
            "type": "Overexposure",
            "severity": severity,
            "confidence": round(conf, 2),
            "explanation": (
                "Image is overexposed. Highlight clipping may have occurred."
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
                "Low contrast detected. Limited dynamic range reduces feature separability."
            ),
        })

    # Severe degradation
    if entropy < 4:
        issues.append({
            "type": "Severe Degradation",
            "severity": "HIGH",
            "confidence": round(min(0.99, 0.7 + (4 - entropy) / 10), 2),
            "explanation": (
                "Very low information content. The image may be severely degraded."
            ),
        })

    return issues


# ---- Summary and recommendation ----

def _label_from_score(score: int) -> str:
    """Convert a 0-100 quality score into a quality label."""
    if score >= _THRESHOLD_ACCEPTABLE:
        return "ACCEPTABLE"
    if score >= _THRESHOLD_DEGRADED:
        return "DEGRADED"
    return "DEFECTIVE"


def _recommendation(label: str) -> str:
    """Generate a recommendation based on the quality label."""
    if label == "ACCEPTABLE":
        return "Safe for downstream analysis."
    if label == "DEGRADED":
        return "Use with caution or consider image enhancement."
    return "Recapture the image or perform manual review before downstream use."


def _summary(label: str, issues: list[dict], score: int) -> str:
    """Generate a human-readable quality summary."""
    if label == "ACCEPTABLE" and not issues:
        return "Image quality is suitable for downstream visual analysis."
    if label == "ACCEPTABLE":
        return (
            f"The image demonstrates acceptable overall visual quality (score: {score}/100). "
            "Minor issues were detected but are unlikely to significantly affect "
            "downstream computer vision analysis."
        )
    if label == "DEGRADED":
        types = ", ".join(i["type"] for i in issues)
        return (
            f"The image exhibits degraded quality ({types}). "
            "These conditions may reduce the reliability of downstream analysis. "
            "Manual review or image enhancement is recommended."
        )
    # DEFECTIVE
    return (
        f"Significant visual degradation detected (score: {score}/100). "
        "Image quality is critically low and unsuitable for automated analysis. "
        "Recapture or manual review is required."
    )


def predict_quality(features: dict[str, float]) -> dict:
    """Run inference on extracted features and return the analysis result.

    If a trained scikit-learn model is available, it will be used.
    Otherwise a transparent rule-based fallback is applied.
    """
    start = time.perf_counter()
    model = get_model()
    scaler = get_scaler()
    metadata = get_metadata()

    model_used = False

    if model is not None and scaler is not None and metadata is not None:
        # ---- Real ML model path ----
        try:
            feature_names = metadata.get("feature_names", [])
            vector = np.array([[features.get(k, 0.0) for k in feature_names]])
            vector_scaled = scaler.transform(vector)

            # Predict quality score
            raw_score = float(model.predict(vector_scaled)[0])
            score = int(round(max(0.0, min(100.0, raw_score))))
            label = _label_from_score(score)
            confidence = _estimate_confidence(score, features)
            issues = _detect_issues(features)
            rec = _recommendation(label)
            summ = _summary(label, issues, score)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            model_used = True

            return {
                "quality_score": score,
                "quality_label": label,
                "analysis_confidence": round(confidence * 100, 1),
                "issues": issues,
                "statistics": features,
                "summary": summ,
                "recommendation": rec,
                "processing_time_ms": elapsed_ms,
            }
        except Exception:
            # Fall through to rule-based if model inference fails
            pass

    # ---- Rule-based fallback ----
    # Weighted composite score
    weights = {
        "sharpness": 0.25,
        "brightness": 0.15,
        "contrast": 0.15,
        "noise_estimate": 0.20,
        "entropy": 0.15,
        "saturation": 0.10,
    }
    ranges: dict[str, tuple[float, float]] = {
        "sharpness": (0, 300),
        "brightness": (0, 255),
        "contrast": (0, 100),
        "noise_estimate": (0, 60),
        "entropy": (0, 8),
        "saturation": (0, 150),
    }

    total = 0.0
    for key, weight in weights.items():
        lo, hi = ranges[key]
        val = features.get(key, lo)
        normed = max(0.0, min(1.0, (val - lo) / (hi - lo) if hi > lo else 0.0))
        if key == "noise_estimate":
            normed = 1.0 - normed
        total += normed * weight

    score = int(round(total * 100))
    label = _label_from_score(score)
    issues = _detect_issues(features)
    confidence = _estimate_confidence(score, features)
    rec = _recommendation(label)
    summ = _summary(label, issues, score)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return {
        "quality_score": score,
        "quality_label": label,
        "analysis_confidence": round(confidence * 100, 1),
        "issues": issues,
        "statistics": features,
        "summary": summ,
        "recommendation": rec,
        "processing_time_ms": elapsed_ms,
    }
