"""Downstream Analytics Impact estimation.

Based on VisionGuard's detected image-quality conditions, this module
estimates how different downstream computer-vision analytics might be
affected by the current image quality.

IMPORTANT DISCLAIMER:
This is an ESTIMATED downstream reliability assessment based on detected
image-quality conditions. It is NOT measured downstream model accuracy.
No downstream models (YOLO, OCR, etc.) are actually run or evaluated.

The terminology used is:
- "estimated reliability" (not "accuracy")
- "expected impact" (not "error rate")
- "downstream suitability" (not "model performance")
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Analytics categories and their context relevance
# ---------------------------------------------------------------------------

ANALYTICS_CATEGORIES = {
    "object_detection": {
        "name": "Object Detection",
        "description": "General object detection and classification",
        "contexts": ["CCTV Surveillance", "Traffic Monitoring", "Drone Imagery",
                      "Infrastructure Inspection", "Crowd Monitoring", "Smart Campus"],
    },
    "vehicle_detection": {
        "name": "Vehicle Detection",
        "description": "Vehicle identification and tracking",
        "contexts": ["Traffic Monitoring", "CCTV Surveillance", "Smart Campus"],
    },
    "person_detection": {
        "name": "Person Detection",
        "description": "Pedestrian and person detection",
        "contexts": ["CCTV Surveillance", "Crowd Monitoring", "Drone Imagery", "Smart Campus"],
    },
    "ocr_license_plate": {
        "name": "OCR / License Plate",
        "description": "Optical character recognition and license plate reading",
        "contexts": ["Traffic Monitoring", "CCTV Surveillance", "Smart Campus"],
    },
    "infrastructure_defect": {
        "name": "Infrastructure / Defect Detection",
        "description": "Structural defect and damage detection",
        "contexts": ["Infrastructure Inspection"],
    },
}

RELIABILITY_LEVELS = ["HIGH", "MEDIUM", "LOW", "VERY LOW"]

# ---------------------------------------------------------------------------
# Issue-to-reliability mapping rules
# ---------------------------------------------------------------------------
# Each rule maps (issue_type, severity, analytics_category) → reliability.
# More severe issues → lower reliability.
# These are DOMAIN RULES, not hard-coded results for specific images.

_ISSUE_IMPACT_RULES: dict[str, dict[str, dict[str, str]]] = {
    # severity → { analytics_category → reliability }
    "critical": {
        "object_detection": "VERY LOW",
        "vehicle_detection": "VERY LOW",
        "person_detection": "VERY LOW",
        "ocr_license_plate": "VERY LOW",
        "infrastructure_defect": "VERY LOW",
    },
    "high": {
        "object_detection": "LOW",
        "vehicle_detection": "LOW",
        "person_detection": "LOW",
        "ocr_license_plate": "LOW",
        "infrastructure_defect": "LOW",
    },
    "moderate": {
        "object_detection": "MEDIUM",
        "vehicle_detection": "MEDIUM",
        "person_detection": "MEDIUM",
        "ocr_license_plate": "LOW",
        "infrastructure_defect": "MEDIUM",
    },
    "low": {
        "object_detection": "MEDIUM",
        "vehicle_detection": "MEDIUM",
        "person_detection": "MEDIUM",
        "ocr_license_plate": "MEDIUM",
        "infrastructure_defect": "MEDIUM",
    },
}

# Issue types that specifically degrade text-related analytics (OCR)
_TEXT_SENSITIVE_ISSUES = {
    "severe_blur", "insufficient_sharpness", "image_noise", "severe_noise",
    "underexposure", "severe_underexposure", "overexposure", "severe_overexposure",
}

# Issue types that specifically degrade infrastructure inspection
_INFRA_SENSITIVE_ISSUES = {
    "severe_blur", "insufficient_sharpness", "image_noise", "severe_noise",
    "low_contrast", "limited_dynamic_range",
}

# Issue types that affect all categories
_UNIVERSAL_ISSUES = {
    "severe_visual_degradation", "potential_visual_defect",
}

# ---------------------------------------------------------------------------
# Reliability estimation logic
# ---------------------------------------------------------------------------

def _get_reliability_from_issues(
    issues: list[dict],
    quality_score: float,
) -> dict[str, str]:
    """Estimate reliability for each analytics category based on issues.

    Returns a dict mapping analytics_key → reliability level.
    """
    # Start with HIGH for all categories
    reliability = {key: "HIGH" for key in ANALYTICS_CATEGORIES}

    if not issues:
        return reliability

    # Apply issue-based penalties
    for issue in issues:
        issue_type = issue.get("type", "")
        severity = issue.get("severity", "low")

        # Get the rule-based reliability for this severity
        severity_rules = _ISSUE_IMPACT_RULES.get(severity, _ISSUE_IMPACT_RULES["low"])

        for category_key in reliability:
            rule_reliability = severity_rules.get(category_key, "MEDIUM")

            # Special handling for text-sensitive issues (OCR)
            if category_key == "ocr_license_plate" and issue_type in _TEXT_SENSITIVE_ISSUES:
                # Text analytics are more sensitive to these issues
                if severity in ("critical", "high"):
                    rule_reliability = "LOW"
                elif severity == "moderate":
                    rule_reliability = "LOW"

            # Special handling for infrastructure inspection
            if category_key == "infrastructure_defect":
                if issue_type in _INFRA_SENSITIVE_ISSUES:
                    if severity in ("critical", "high"):
                        rule_reliability = "LOW"
                    elif severity == "moderate":
                        rule_reliability = "MEDIUM"
                elif issue_type not in _INFRA_SENSITIVE_ISSUES and issue_type not in _UNIVERSAL_ISSUES:
                    # Infrastructure-specific issues may not affect general infrastructure inspection
                    continue

            # Apply the more severe (lower) reliability
            current_idx = RELIABILITY_LEVELS.index(reliability[category_key])
            rule_idx = RELIABILITY_LEVELS.index(rule_reliability)
            if rule_idx > current_idx:  # Higher index = lower reliability
                reliability[category_key] = rule_reliability

    # Quality score adjustments
    if quality_score < 20:
        # Very low quality - everything degrades
        for key in reliability:
            if reliability[key] in ("HIGH", "MEDIUM"):
                reliability[key] = "LOW"
    elif quality_score < 40:
        # Low quality - moderate degradation
        for key in reliability:
            if reliability[key] == "HIGH":
                reliability[key] = "MEDIUM"

    return reliability


def _get_primary_reason(
    analytics_key: str,
    issues: list[dict],
    quality_score: float,
) -> str:
    """Generate a human-readable primary reason for the reliability rating."""
    if not issues:
        if quality_score >= 70:
            return "No major quality issues detected. Image quality is suitable for analytics."
        elif quality_score >= 50:
            return "Minor quality limitations detected. Analytics may have reduced confidence."
        else:
            return "Low overall quality score may affect analytics performance."

    # Find the most impactful issue for this category
    category_name = ANALYTICS_CATEGORIES[analytics_key]["name"]

    # Group issues by type
    issue_types = [i.get("type", "") for i in issues]
    severities = [i.get("severity", "low") for i in issues]

    # Check for critical issues first
    if "critical" in severities:
        critical_issues = [i for i in issues if i.get("severity") == "critical"]
        issue_names = [i.get("type", "").replace("_", " ") for i in critical_issues]
        return f"Critical quality issues ({', '.join(issue_names)}) significantly reduce {category_name.lower()} reliability."

    # Check for high severity
    if "high" in severities:
        high_issues = [i for i in issues if i.get("severity") == "high"]
        issue_names = [i.get("type", "").replace("_", " ") for i in high_issues]
        return f"High-severity issues ({', '.join(issue_names)}) reduce {category_name.lower()} reliability."

    # Check for moderate issues
    if "moderate" in severities:
        mod_issues = [i for i in issues if i.get("severity") == "moderate"]
        issue_names = [i.get("type", "").replace("_", " ") for i in mod_issues]
        return f"Moderate quality issues ({', '.join(issue_names)}) may affect {category_name.lower()}."

    # Low severity
    low_issues = [i for i in issues if i.get("severity") == "low"]
    issue_names = [i.get("type", "").replace("_", " ") for i in low_issues]
    return f"Minor quality issues ({', '.join(issue_names)}) detected."


def _get_supporting_issues(
    analytics_key: str,
    issues: list[dict],
) -> list[dict]:
    """Return issues that are relevant to this analytics category."""
    supporting = []

    # All issues that could affect this category
    for issue in issues:
        issue_type = issue.get("type", "")
        severity = issue.get("severity", "low")

        # Text-sensitive issues always affect OCR
        if analytics_key == "ocr_license_plate" and issue_type in _TEXT_SENSITIVE_ISSUES:
            supporting.append({
                "issue_type": issue_type,
                "severity": severity,
                "relevance": "high",
            })
            continue

        # Infrastructure-specific issues
        if analytics_key == "infrastructure_defect" and issue_type in _INFRA_SENSITIVE_ISSUES:
            supporting.append({
                "issue_type": issue_type,
                "severity": severity,
                "relevance": "high",
            })
            continue

        # Universal issues affect all categories
        if issue_type in _UNIVERSAL_ISSUES:
            supporting.append({
                "issue_type": issue_type,
                "severity": severity,
                "relevance": "high",
            })
            continue

        # Other issues have general relevance
        if severity in ("high", "critical"):
            supporting.append({
                "issue_type": issue_type,
                "severity": severity,
                "relevance": "medium",
            })

    return supporting


def _get_expected_impact(
    analytics_key: str,
    reliability: str,
    issues: list[dict],
) -> str:
    """Generate expected impact description based on reliability and issues."""
    category_name = ANALYTICS_CATEGORIES[analytics_key]["name"]

    if reliability == "HIGH":
        return f"No significant impact expected on {category_name.lower()}."
    elif reliability == "MEDIUM":
        # Find the main issue
        main_issue = issues[0]["type"].replace("_", " ") if issues else "quality limitations"
        return f"{main_issue.title()} may reduce {category_name.lower()} confidence in some regions."
    elif reliability == "LOW":
        main_issues = [i["type"].replace("_", " ") for i in issues[:2]]
        return f"Significant impact: {', '.join(main_issues)} may cause {category_name.lower()} failures."
    else:  # VERY LOW
        return f"Image quality is insufficient for reliable {category_name.lower()}."


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def estimate_downstream_analytics(
    quality_score: float,
    issues: list[dict],
    context: str = "CCTV Surveillance",
    analytics_readiness_score: float = 0.0,
) -> list[dict]:
    """Estimate downstream analytics reliability based on image quality.

    This is an ESTIMATED downstream reliability assessment, NOT measured
    downstream model accuracy. No downstream models are actually executed.

    Parameters
    ----------
    quality_score : float
        Calibrated quality score (0-100).
    issues : list[dict]
        Detected issues from the issue detector.
    context : str
        Pipeline context (e.g., "CCTV Surveillance").
    analytics_readiness_score : float
        Analytics readiness score (0-100).

    Returns
    -------
    list[dict]
        List of downstream analytics assessments, each containing:
        - analytics_type: str (display name)
        - analytics_key: str (machine key)
        - estimated_reliability: str (HIGH/MEDIUM/LOW/VERY LOW)
        - primary_reason: str
        - expected_impact: str
        - supporting_issues: list[dict]
        - disclaimer: str
    """
    # Get reliability estimates
    reliability = _get_reliability_from_issues(issues, quality_score)

    # Build results for context-relevant categories
    results = []

    for category_key, category_info in ANALYTICS_CATEGORIES.items():
        # Only include categories relevant to the current context
        if context not in category_info["contexts"]:
            continue

        rel = reliability[category_key]
        primary_reason = _get_primary_reason(category_key, issues, quality_score)
        supporting = _get_supporting_issues(category_key, issues)
        expected_impact = _get_expected_impact(category_key, rel, issues)

        results.append({
            "analytics_type": category_info["name"],
            "analytics_key": category_key,
            "estimated_reliability": rel,
            "primary_reason": primary_reason,
            "expected_impact": expected_impact,
            "supporting_issues": supporting,
            "disclaimer": "Estimated reliability based on detected image-quality conditions; not measured model accuracy.",
        })

    return results
