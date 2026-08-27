"""Feature-based issue detection.

Issues are detected from actual extracted feature values using
percentile-derived thresholds from the combined training data.
"""

from __future__ import annotations

import numpy as np


# Default thresholds (can be overridden by calibration data)
# These are derived from typical combined KADID + KonIQ distributions.
DEFAULT_THRESHOLDS: dict[str, dict[str, float]] = {
    "brightness": {
        "p5": 35.0,     # severely underexposed
        "p10": 55.0,    # underexposed
        "p90": 190.0,   # overexposed
        "p95": 210.0,   # severely overexposed
        "optimal_low": 80.0,
        "optimal_high": 170.0,
    },
    "contrast": {
        "p5": 12.0,
        "p10": 18.0,    # low contrast
        "optimal_low": 35.0,
    },
    "sharpness": {
        "p5": 30.0,
        "p10": 80.0,    # low sharpness / blur
        "optimal_low": 200.0,
    },
    "saturation": {
        "p5": 5.0,
        "p10": 15.0,    # low saturation
        "p95": 120.0,   # oversaturated
    },
    "noise_estimate": {
        "p90": 25.0,    # high noise
        "p95": 40.0,    # severe noise
    },
    "entropy": {
        "p10": 4.5,     # low information content
    },
    "dynamic_range": {
        "p10": 60.0,    # limited dynamic range
    },
    "edge_density": {
        "p5": 3.0,
        "p10": 5.0,     # low structural content
    },
}


def _severity_from_deviation(value: float, threshold: float, ref_range: float, direction: str = "below") -> str:
    """Determine severity based on how far the value deviates from threshold."""
    if ref_range <= 0:
        return "medium"
    if direction == "below":
        deviation = threshold - value
    else:
        deviation = value - threshold
    if deviation <= 0:
        return "low"
    ratio = deviation / ref_range
    if ratio < 0.3:
        return "low"
    elif ratio < 0.7:
        return "medium"
    return "high"


def _confidence_from_deviation(value: float, threshold: float, ref_range: float, direction: str = "below") -> float:
    """Compute confidence (0-1) based on deviation from threshold."""
    if ref_range <= 0:
        return 0.6
    if direction == "below":
        deviation = threshold - value
    else:
        deviation = value - threshold
    if deviation <= 0:
        return 0.0
    return round(min(0.99, 0.5 + (deviation / ref_range) * 0.49), 3)


def detect_issues(
    features: dict[str, float],
    feature_stats: dict | None = None,
) -> list[dict]:
    """Detect quality issues from extracted features.

    Args:
        features: Extracted image features (all 12+ statistics).
        feature_stats: Optional dict with percentile thresholds from training.
                       If None, uses DEFAULT_THRESHOLDS.

    Returns:
        List of detected issues, each with type, title, severity,
        confidence, feature_value, threshold, and description.
    """
    # Use provided thresholds or defaults
    thresholds = DEFAULT_THRESHOLDS
    if feature_stats and "feature_thresholds" in feature_stats:
        thresholds = feature_stats["feature_thresholds"]

    issues: list[dict] = []

    # --- Brightness issues ---
    bt = thresholds.get("brightness", {})
    br = features.get("brightness", 128.0)

    p5 = bt.get("p5", 35.0)
    p10 = bt.get("p10", 55.0)
    p90 = bt.get("p90", 190.0)
    p95 = bt.get("p95", 210.0)
    opt_lo = bt.get("optimal_low", 80.0)
    opt_hi = bt.get("optimal_high", 170.0)

    # Severe underexposure
    if br < p5:
        ref_range = max(opt_lo - p5, 1.0)
        issues.append({
            "type": "severe_underexposure",
            "title": "Severe Underexposure",
            "severity": "HIGH",
            "confidence": _confidence_from_deviation(br, p5, ref_range, "below"),
            "feature_value": round(br, 2),
            "threshold": p5,
            "description": "Image is extremely dark. Most shadow detail is lost and the image is unreliable for visual analysis.",
        })
    elif br < p10:
        ref_range = max(opt_lo - p10, 1.0)
        issues.append({
            "type": "low_brightness",
            "title": "Low Brightness",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(br, p10, ref_range, "below"),
            "feature_value": round(br, 2),
            "threshold": p10,
            "description": "Image appears darker than the expected range. Shadow detail may be partially lost.",
        })

    # Severe overexposure
    if br > p95:
        ref_range = max(p95 - opt_hi, 1.0)
        issues.append({
            "type": "severe_overexposure",
            "title": "Severe Overexposure",
            "severity": "HIGH",
            "confidence": _confidence_from_deviation(br, p95, ref_range, "above"),
            "feature_value": round(br, 2),
            "threshold": p95,
            "description": "Image is extremely bright with significant highlight clipping. Detail in bright areas is lost.",
        })
    elif br > p90:
        ref_range = max(p90 - opt_hi, 1.0)
        issues.append({
            "type": "high_brightness",
            "title": "High Brightness",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(br, p90, ref_range, "above"),
            "feature_value": round(br, 2),
            "threshold": p90,
            "description": "Image may be overexposed. Some highlight clipping may have occurred.",
        })

    # --- Contrast issues ---
    ct = thresholds.get("contrast", {})
    co = features.get("contrast", 50.0)
    c_p10 = ct.get("p10", 18.0)
    c_opt = ct.get("optimal_low", 35.0)

    if co < c_p10:
        ref_range = max(c_opt - c_p10, 1.0)
        issues.append({
            "type": "low_contrast",
            "title": "Low Contrast",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(co, c_p10, ref_range, "below"),
            "feature_value": round(co, 2),
            "threshold": c_p10,
            "description": "Image has limited separation between light and dark regions. Dynamic range may be too compressed.",
        })

    # --- Sharpness / Blur issues ---
    st = thresholds.get("sharpness", {})
    sh = features.get("sharpness", 100.0)
    s_p5 = st.get("p5", 30.0)
    s_p10 = st.get("p10", 80.0)
    s_opt = st.get("optimal_low", 200.0)

    if sh < s_p5:
        ref_range = max(s_opt - s_p5, 1.0)
        issues.append({
            "type": "severe_blur",
            "title": "Severe Blur",
            "severity": "HIGH",
            "confidence": _confidence_from_deviation(sh, s_p5, ref_range, "below"),
            "feature_value": round(sh, 2),
            "threshold": s_p5,
            "description": "Image is severely blurred with very low edge detail. Edge information is critically below the expected threshold.",
        })
    elif sh < s_p10:
        ref_range = max(s_opt - s_p10, 1.0)
        issues.append({
            "type": "low_sharpness",
            "title": "Low Sharpness",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(sh, s_p10, ref_range, "below"),
            "feature_value": round(sh, 2),
            "threshold": s_p10,
            "description": "Image appears soft or slightly blurry. Edge detail is below the expected threshold.",
        })

    # --- Noise issues ---
    nt = thresholds.get("noise_estimate", {})
    noise = features.get("noise_estimate", 0.0)
    n_p90 = nt.get("p90", 25.0)
    n_p95 = nt.get("p95", 40.0)

    if noise > n_p95:
        ref_range = max(n_p95 - 10.0, 1.0)
        issues.append({
            "type": "high_noise",
            "title": "High Noise",
            "severity": "HIGH",
            "confidence": _confidence_from_deviation(noise, n_p95, ref_range, "above"),
            "feature_value": round(noise, 2),
            "threshold": n_p95,
            "description": "Image contains very high levels of noise. This significantly degrades visual quality and feature extraction.",
        })
    elif noise > n_p90:
        ref_range = max(n_p90 - 10.0, 1.0)
        issues.append({
            "type": "moderate_noise",
            "title": "Moderate Noise",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(noise, n_p90, ref_range, "above"),
            "feature_value": round(noise, 2),
            "threshold": n_p90,
            "description": "Image contains elevated noise levels which may affect the reliability of visual analysis.",
        })

    # --- Saturation issues ---
    sat_t = thresholds.get("saturation", {})
    sa = features.get("saturation", 50.0)
    sat_p10 = sat_t.get("p10", 15.0)
    sat_p95 = sat_t.get("p95", 120.0)

    if sa < sat_p10:
        ref_range = max(40.0 - sat_p10, 1.0)
        issues.append({
            "type": "low_saturation",
            "title": "Low Saturation",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(sa, sat_p10, ref_range, "below"),
            "feature_value": round(sa, 2),
            "threshold": sat_p10,
            "description": "Colors appear muted or desaturated. This may reduce the effectiveness of color-based analysis.",
        })

    if sa > sat_p95:
        ref_range = max(sat_p95 - 60.0, 1.0)
        issues.append({
            "type": "oversaturation",
            "title": "Oversaturation",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(sa, sat_p95, ref_range, "above"),
            "feature_value": round(sa, 2),
            "threshold": sat_p95,
            "description": "Colors appear unnaturally oversaturated. This may indicate aggressive color processing.",
        })

    # --- Entropy issues ---
    et = thresholds.get("entropy", {})
    ent = features.get("entropy", 6.0)
    e_p10 = et.get("p10", 4.5)

    if ent < e_p10:
        ref_range = max(7.0 - e_p10, 1.0)
        issues.append({
            "type": "low_entropy",
            "title": "Low Information Content",
            "severity": "LOW",
            "confidence": _confidence_from_deviation(ent, e_p10, ref_range, "below"),
            "feature_value": round(ent, 2),
            "threshold": e_p10,
            "description": "Image has low information content. This may indicate a featureless or heavily compressed scene.",
        })

    # --- Dynamic range issues ---
    drt = thresholds.get("dynamic_range", {})
    dr = features.get("dynamic_range", 100.0)
    dr_p10 = drt.get("p10", 60.0)

    if dr < dr_p10:
        ref_range = max(120.0 - dr_p10, 1.0)
        issues.append({
            "type": "limited_dynamic_range",
            "title": "Limited Dynamic Range",
            "severity": "LOW",
            "confidence": _confidence_from_deviation(dr, dr_p10, ref_range, "below"),
            "feature_value": round(dr, 2),
            "threshold": dr_p10,
            "description": "Image has a narrow tonal range. This may limit the separation of features across intensity levels.",
        })

    # --- Edge density issues ---
    edt = thresholds.get("edge_density", {})
    ed = features.get("edge_density", 0.05)
    ed_p10 = edt.get("p10", 5.0)

    if ed < ed_p10:
        ref_range = max(15.0 - ed_p10, 1.0)
        issues.append({
            "type": "low_structural_content",
            "title": "Low Structural Content",
            "severity": "MEDIUM",
            "confidence": _confidence_from_deviation(ed, ed_p10, ref_range, "below"),
            "feature_value": round(ed, 2),
            "threshold": ed_p10,
            "description": "Image contains few detectable edges or structural features. This may indicate a flat, featureless, or heavily blurred scene.",
        })

    return issues
