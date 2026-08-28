"""Phase 6 — Issue-driven explainability for VisionGuard.

For every detected issue, generates a structured explanation containing:
  - issue        : human-readable issue label
  - evidence     : {metric, value, threshold} from the actual detector
  - why_it_matters : smart-city / downstream-analytics impact
  - recommendation : actionable remediation advice

All text is driven by the real issue type, severity, and metric values
produced by the issue_detector — nothing is hard-coded to specific images.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Human-readable issue labels
# ---------------------------------------------------------------------------

_ISSUE_LABELS: dict[tuple[str, str], str] = {
    # (issue_type, severity) → label
    ("severe_blur", "critical"): "Severe Blur",
    ("insufficient_sharpness", "high"): "Insufficient Sharpness",
    ("underexposure", "critical"): "Severe Underexposure",
    ("underexposure", "high"): "Underexposure",
    ("overexposure", "critical"): "Severe Overexposure",
    ("overexposure", "high"): "Overexposure",
    ("image_noise", "critical"): "Severe Noise",
    ("image_noise", "high"): "Moderate Noise",
    ("severe_visual_degradation", "critical"): "Severe Visual Degradation",
    ("visual_degradation", "high"): "Visual Degradation",
    ("potential_visual_defect", "high"): "Potential Visual Defect",
    ("low_contrast", "high"): "Low Contrast",
    ("limited_dynamic_range", "high"): "Limited Dynamic Range",
    ("low_color_information", "low"): "Low Colour Information",
}


def _issue_label(issue_type: str, severity: str) -> str:
    """Return a human-readable label for the issue."""
    return _ISSUE_LABELS.get((issue_type, severity), issue_type.replace("_", " ").title())


# ---------------------------------------------------------------------------
# "Why it matters" explanations — smart-city / downstream-analytics focused
# ---------------------------------------------------------------------------

_WHY_IT_MATTERS: dict[str, dict[str, str]] = {
    # issue_type → {severity → text}  with severity-specific variants
    "severe_blur": {
        "critical": (
            "Critical blur destroys fine detail needed for object detection, "
            "face recognition, license-plate reading, and pedestrian tracking. "
            "Downstream analytics may fail entirely on this frame."
        ),
    },
    "insufficient_sharpness": {
        "high": (
            "Reduced sharpness degrades edge features that object detectors and "
            "classifiers rely on. Detection of small, distant, or fast-moving "
            "targets is likely less reliable."
        ),
    },
    "underexposure": {
        "critical": (
            "Severe underexposure hides scene content in dark regions. Pedestrians, "
            "vehicles, road markings, and other critical objects may be invisible "
            "to automated detection pipelines."
        ),
        "high": (
            "Underexposure reduces contrast in shadow areas, lowering detection "
            "confidence for people, vehicles, and infrastructure elements."
        ),
    },
    "overexposure": {
        "critical": (
            "Severe overexposure washes out bright regions, destroying surface "
            "textures, signage, and lane markings that analytics systems depend on."
        ),
        "high": (
            "Overexposed areas lose detail in highlights, reducing the reliability "
            "of colour-based classification and surface analysis."
        ),
    },
    "image_noise": {
        "critical": (
            "Severe sensor noise obscures object boundaries and introduces false "
            "edges. Feature extraction, segmentation, and tracking algorithms "
            "produce unreliable results."
        ),
        "high": (
            "Elevated noise interferes with edge detection and texture analysis, "
            "reducing the accuracy of downstream computer-vision models."
        ),
    },
    "severe_visual_degradation": {
        "critical": (
            "Multiple independent quality failures indicate the image has lost "
            "substantial usable visual information. Automated analytics are "
            "likely unreliable for this frame."
        ),
    },
    "visual_degradation": {
        "high": (
            "Several image-quality indicators are degraded simultaneously, "
            "reducing the overall reliability of feature extraction and "
            "object detection."
        ),
    },
    "potential_visual_defect": {
        "high": (
            "Unusual visual characteristics suggest a possible capture, sensor, "
            "or compression artefact. Affected regions may produce false positives "
            "or missed detections."
        ),
    },
    "low_contrast": {
        "high": (
            "Low contrast makes it harder to separate objects from their "
            "background, reducing segmentation accuracy and detection confidence."
        ),
    },
    "limited_dynamic_range": {
        "high": (
            "A narrow tonal range compresses scene detail, limiting the "
            "effectiveness of gradient-based feature detectors and "
            "histogram-based analysis."
        ),
    },
    "low_color_information": {
        "low": (
            "Desaturated imagery limits colour-based analytics such as "
            "vehicle classification, vegetation mapping, and appearance-based "
            "re-identification."
        ),
    },
}


def _why_it_matters(issue_type: str, severity: str) -> str:
    """Return the 'why it matters' text for an issue."""
    issue_map = _WHY_IT_MATTERS.get(issue_type, {})
    # Prefer severity-specific text, fall back to any available variant
    if severity in issue_map:
        return issue_map[severity]
    if issue_map:
        return next(iter(issue_map.values()))
    return (
        "This issue may reduce the reliability of downstream smart-city "
        "computer-vision analytics."
    )


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

_RECOMMENDATIONS: dict[str, str] = {
    "severe_blur": (
        "Consider recapturing the image with a stabilised camera, faster "
        "shutter speed, or improved autofocus. If recapture is not possible, "
        "apply deblurring algorithms before analytics."
    ),
    "insufficient_sharpness": (
        "Improve camera focus or use a lens with higher resolving power. "
        "Software sharpening may partially recover lost detail."
    ),
    "underexposure": (
        "Increase camera exposure or gain, add supplementary lighting, "
        "or apply adaptive histogram equalisation to recover shadow detail."
    ),
    "overexposure": (
        "Reduce camera exposure or apply dynamic range compression. "
        "ND filters or auto-exposure adjustments can prevent highlight clipping."
    ),
    "image_noise": (
        "Apply denoising filters (e.g., bilateral or non-local means) before "
        "analytics, or use a camera with a larger sensor and lower ISO."
    ),
    "severe_visual_degradation": (
        "This image is unreliable for automated analysis. Recapture with "
        "calibrated settings, or flag for manual review."
    ),
    "visual_degradation": (
        "Consider image restoration techniques or recapture if possible. "
        "Lower confidence thresholds when feeding this frame to detectors."
    ),
    "potential_visual_defect": (
        "Inspect the capture pipeline for sensor or compression issues. "
        "Cross-validate detections with adjacent frames when available."
    ),
    "low_contrast": (
        "Apply contrast enhancement (e.g., CLAHE) or adjust camera "
        "exposure settings to increase tonal separation."
    ),
    "limited_dynamic_range": (
        "Use HDR capture or tone-mapping to expand the usable tonal range. "
        "Gradient-domain processing may also help."
    ),
    "low_color_information": (
        "Verify white-balance settings and ensure adequate lighting. "
        "Colour correction may partially restore useful chrominance data."
    ),
}


def _recommendation(issue_type: str) -> str:
    """Return the recommendation text for an issue type."""
    return _RECOMMENDATIONS.get(
        issue_type,
        "Review the image quality and consider recapturing or enhancing "
        "the image before running downstream analytics.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_issue_explanations(
    issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate structured explainability for every detected issue.

    Parameters
    ----------
    issues :
        Output of ``detect_issues()`` — each dict must contain
        ``type``, ``severity``, ``metric``, ``value``, ``threshold``.

    Returns
    -------
    list[dict]
        One entry per issue, each with keys:
        ``issue``, ``evidence``, ``why_it_matters``, ``recommendation``.
    """
    explanations: list[dict[str, Any]] = []

    for issue in issues:
        issue_type = issue.get("type", "unknown")
        severity = issue.get("severity", "moderate")
        metric = issue.get("metric", "unknown")
        value = issue.get("value", 0.0)
        threshold = issue.get("threshold", 0.0)

        explanations.append({
            "issue": _issue_label(issue_type, severity),
            "evidence": {
                "metric": metric,
                "value": value,
                "threshold": threshold,
            },
            "why_it_matters": _why_it_matters(issue_type, severity),
            "recommendation": _recommendation(issue_type),
        })

    return explanations
