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


# ---------------------------------------------------------------------------
# Advanced explainability: Issue-driven primary explanation
# ---------------------------------------------------------------------------

# Severity priority (higher = more important)
_SEVERITY_PRIORITY = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "low": 1,
}


# Downstream analytics impact by issue type
_DOWNSTREAM_IMPACT: dict[str, str] = {
    "severe_blur": (
        "Object detection, person tracking, and license-plate reading may fail "
        "entirely. Fine detail required for classification is destroyed."
    ),
    "insufficient_sharpness": (
        "Edge-based detection algorithms may miss small or distant objects. "
        "Face recognition and plate reading confidence may decrease."
    ),
    "underexposure": (
        "Dark regions hide pedestrians, vehicles, and road markings. Detection "
        "reliability is reduced in shadowed areas."
    ),
    "overexposure": (
        "Bright regions lose surface detail, reducing the reliability of "
        "colour-based classification and surface analysis."
    ),
    "image_noise": (
        "Sensor noise creates false edges and textures, interfering with "
        "feature extraction and object boundary detection."
    ),
    "severe_visual_degradation": (
        "Multiple quality failures indicate substantial loss of usable visual "
        "information. Automated analytics are likely unreliable."
    ),
    "visual_degradation": (
        "Reduced visual information across multiple signals may lower detection "
        "confidence and increase false-positive rates."
    ),
    "potential_visual_defect": (
        "Possible capture or compression artefacts may produce unreliable "
        "detections in affected regions."
    ),
    "low_contrast": (
        "Weak contrast limits object-background separation, reducing "
        "segmentation accuracy and detection confidence."
    ),
    "limited_dynamic_range": (
        "A narrow tonal range compresses scene detail, limiting the "
        "effectiveness of gradient-based detectors."
    ),
    "low_color_information": (
        "Desaturated imagery limits colour-based analytics such as vehicle "
        "classification and appearance-based re-identification."
    ),
}


def _select_primary_issue(issues: list[dict]) -> dict | None:
    """Select the primary issue using deterministic priority rules.
    
    Priority: highest severity -> highest confidence -> first in list.
    """
    if not issues:
        return None
    
    def sort_key(issue: dict) -> tuple[int, float]:
        severity = issue.get("severity", "low")
        confidence = issue.get("confidence", 0.0)
        return (_SEVERITY_PRIORITY.get(severity, 0), confidence)
    
    return max(issues, key=sort_key)


def generate_primary_explanation(
    issues: list[dict],
    quality_score: float,
    downstream_analytics: list[dict] | None = None,
) -> dict:
    """Generate a structured primary issue explanation.
    
    Returns a dict with:
    - primary_issue: {type, severity, confidence, label, evidence, impact}
    - why_it_matters: str
    - downstream_impact: str
    - recommendation: str
    - secondary_issues: list[dict]
    - no_issues_detected: bool
    """
    primary = _select_primary_issue(issues)
    
    if primary is None:
        return {
            "primary_issue": None,
            "why_it_matters": (
                "The image does not contain detected quality problems severe "
                "enough to trigger an issue warning. Quality score and analytics "
                "readiness reflect the overall image assessment."
            ),
            "downstream_impact": (
                "No significant quality issues detected. Downstream analytics "
                "should perform normally for this image quality level."
            ),
            "recommendation": (
                "No action required. The image meets quality standards for "
                "downstream analytics."
            ),
            "secondary_issues": [],
            "no_issues_detected": True,
        }
    
    issue_type = primary.get("type", "unknown")
    severity = primary.get("severity", "moderate")
    
    # Build primary issue with evidence
    primary_issue = {
        "type": issue_type,
        "severity": severity,
        "confidence": primary.get("confidence", 1.0),
        "label": _issue_label(issue_type, severity),
        "evidence": {
            "metric": primary.get("metric", "unknown"),
            "value": primary.get("value", 0.0),
            "threshold": primary.get("threshold", 0.0),
        },
        "impact": primary.get("impact", ""),
    }
    
    # Build secondary issues
    secondary_issues = []
    for issue in issues:
        if issue is primary:
            continue
        issue_type_s = issue.get("type", "unknown")
        severity_s = issue.get("severity", "moderate")
        secondary_issues.append({
            "type": issue_type_s,
            "severity": severity_s,
            "label": _issue_label(issue_type_s, severity_s),
            "confidence": issue.get("confidence", 1.0),
        })
    
    # Get downstream impact
    downstream_text = _DOWNSTREAM_IMPACT.get(
        issue_type,
        "This issue may reduce the reliability of downstream analytics."
    )
    
    # Build downstream analytics summary if available
    downstream_summary = []
    if downstream_analytics:
        for da in downstream_analytics[:4]:  # Show top 4
            downstream_summary.append({
                "analytics_type": da.get("analytics_type", ""),
                "estimated_reliability": da.get("estimated_reliability", "MEDIUM"),
            })
    
    return {
        "primary_issue": primary_issue,
        "why_it_matters": _why_it_matters(issue_type, severity),
        "downstream_impact": downstream_text,
        "downstream_summary": downstream_summary,
        "recommendation": _recommendation(issue_type),
        "secondary_issues": secondary_issues,
        "no_issues_detected": False,
    }
