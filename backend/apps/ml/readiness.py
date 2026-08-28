"""Analytics readiness scoring for downstream smart-city computer vision."""

from __future__ import annotations


def calculate_analytics_readiness(
    quality_score: float,
    issues: list[dict],
) -> dict:
    """
    Calculate whether an image is suitable for downstream analytics.

    The score starts from visual quality and applies penalties based on
    detected conditions that reduce computer vision reliability.
    """

    score = float(quality_score)

    penalties = {
        "blur": 0.0,
        "exposure": 0.0,
        "noise": 0.0,
        "degradation": 0.0,
        "information": 0.0,
    }

    for issue in issues:
        issue_type = issue.get("type", "")
        severity = issue.get("severity", "")

        # ---------------------------------------------------------
        # Blur penalties
        # ---------------------------------------------------------
        if issue_type in {
            "severe_blur",
            "insufficient_sharpness",
        }:
            if severity == "critical":
                penalties["blur"] += 30
            elif severity == "high":
                penalties["blur"] += 20
            elif severity == "moderate":
                penalties["blur"] += 10
            else:
                penalties["blur"] += 5

        # ---------------------------------------------------------
        # Exposure penalties
        # ---------------------------------------------------------
        elif issue_type in {
            "underexposure",
            "severe_underexposure",
            "overexposure",
            "severe_overexposure",
        }:
            if severity == "critical":
                penalties["exposure"] += 25
            elif severity == "high":
                penalties["exposure"] += 15
            elif severity == "moderate":
                penalties["exposure"] += 8
            else:
                penalties["exposure"] += 4

        # ---------------------------------------------------------
        # Noise penalties
        # ---------------------------------------------------------
        elif issue_type in {
            "image_noise",
            "severe_noise",
        }:
            if severity == "critical":
                penalties["noise"] += 25
            elif severity == "high":
                penalties["noise"] += 15
            elif severity == "moderate":
                penalties["noise"] += 8
            else:
                penalties["noise"] += 4

        # ---------------------------------------------------------
        # Severe degradation / corruption
        # ---------------------------------------------------------
        elif issue_type in {
            "severe_visual_degradation",
            "potential_visual_defect",
        }:
            if severity == "critical":
                penalties["degradation"] += 30
            elif severity == "high":
                penalties["degradation"] += 20
            elif severity == "moderate":
                penalties["degradation"] += 10
            else:
                penalties["degradation"] += 5

        # ---------------------------------------------------------
        # Information / scene visibility
        # ---------------------------------------------------------
        elif issue_type in {
            "low_contrast",
            "limited_dynamic_range",
            "low_information",
        }:
            if severity == "critical":
                penalties["information"] += 15
            elif severity == "high":
                penalties["information"] += 10
            elif severity == "moderate":
                penalties["information"] += 5
            else:
                penalties["information"] += 2

    total_penalty = sum(penalties.values())

    # Clamp between 0 and 100
    score = max(0.0, min(100.0, score - total_penalty))

    return {
        "score": round(score, 2),
        "status": get_readiness_status(score),
        "base_quality_score": round(float(quality_score), 2),
        "total_penalty": round(total_penalty, 2),
        "penalties": penalties,
        # Flat penalty keys for the API response
        "blur_penalty": round(penalties["blur"], 2),
        "exposure_penalty": round(penalties["exposure"], 2),
        "noise_penalty": round(penalties["noise"], 2),
        "corruption_penalty": round(penalties["degradation"], 2),
        "information_penalty": round(penalties["information"], 2),
    }


def get_readiness_status(score: float) -> str:
    """Map analytics readiness score to a smart-city readiness status."""

    if score >= 80:
        return "HIGHLY READY"

    if score >= 60:
        return "READY"

    if score >= 40:
        return "LIMITED READINESS"

    if score >= 20:
        return "NOT READY"

    return "CRITICAL / REJECT"