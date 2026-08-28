"""
Rule-based image quality issue detection.

This module maps extracted visual features to concrete, explainable
image-quality issues required by the VisionGuard project.

Required capabilities:
- Blur / insufficient sharpness
- Underexposure
- Overexposure
- Image noise
- Image corruption or severe degradation
- Potential visual defect
- Additional technically justified issues
"""

from __future__ import annotations

from typing import Mapping


def _severity_from_ratio(value: float, warning_threshold: float, severe_threshold: float) -> str:
    """
    Determine severity when larger values indicate a worse issue.

    Example:
        noise_estimate = 25
        warning_threshold = 15
        severe_threshold = 30
    """
    if value >= severe_threshold:
        return "CRITICAL"

    if value >= warning_threshold:
        return "HIGH"

    return "LOW"


def detect_issues(features: Mapping[str, float]) -> list[dict]:
    """
    Detect image quality issues using actual extracted feature thresholds.

    The detector intentionally uses:
    - Pixel-level exposure percentages as the PRIMARY exposure signal.
    - Mean brightness only as supporting evidence.
    - Multiple feature combinations for severe degradation detection.

    Parameters
    ----------
    features:
        Extracted image features.

    Returns
    -------
    list[dict]
        Detected issues with:
        - type
        - severity
        - title
        - message
        - feature
        - value
    """

    issues: list[dict] = []

    # ------------------------------------------------------------------
    # Read features safely
    # ------------------------------------------------------------------

    sharpness = float(features.get("sharpness", 0.0))
    brightness = float(features.get("brightness", 0.0))
    contrast = float(features.get("contrast", 0.0))
    noise = float(features.get("noise_estimate", 0.0))
    entropy = float(features.get("entropy", 0.0))
    saturation = float(features.get("saturation", 0.0))
    under_pct = float(features.get("underexposure_pct", 0.0))
    over_pct = float(features.get("overexposure_pct", 0.0))
    edge_density = float(features.get("edge_density", 0.0))
    dynamic_range = float(features.get("dynamic_range", 0.0))
    colorfulness = float(features.get("colorfulness", 0.0))
    texture_complexity = float(features.get("texture_complexity", 0.0))

    # ------------------------------------------------------------------
    # 1. Blur / insufficient sharpness
    # ------------------------------------------------------------------
    #
    # Variance of Laplacian:
    # lower value -> fewer high-frequency edges -> possible blur.
    #
    # Thresholds should later be calibrated against the smart-city
    # benchmark dataset.

    if sharpness < 15:
        issues.append(
            {
                "type": "severe_blur",
                "severity": "CRITICAL",
                "confidence": 1.0,
                "title": "Severe blur detected",
                "message": (
                    "Image sharpness is critically low. Fine details may not "
                    "be reliable for downstream computer vision analytics."
                ),
                "feature": "sharpness",
                "value": round(sharpness, 2),
            }
        )

    elif sharpness < 40:
        issues.append(
            {
                "type": "insufficient_sharpness",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Insufficient image sharpness",
                "message": (
                    "Image sharpness is below the recommended range and may "
                    "reduce detection reliability for small or distant objects."
                ),
                "feature": "sharpness",
                "value": round(sharpness, 2),
            }
        )

    # ------------------------------------------------------------------
    # 2. Underexposure
    # ------------------------------------------------------------------
    #
    # PRIMARY SIGNAL:
    # Percentage of pixels below the dark clipping threshold.
    #
    # SUPPORTING SIGNAL:
    # Mean brightness.
    #
    # Only ONE final underexposure issue can be created.

    brightness_underexposed = brightness < 70
    brightness_severe_underexposed = brightness < 40

    pixel_underexposed = under_pct >= 15
    pixel_severe_underexposed = under_pct >= 40

    if pixel_severe_underexposed or (
        under_pct >= 25 and brightness_severe_underexposed
    ):
        issues.append(
            {
                "type": "severe_underexposure",
                "severity": "CRITICAL",
                "confidence": 1.0,
                "title": "Severe underexposure detected",
                "message": (
                    "A large portion of the image contains critically dark "
                    "pixels, causing loss of visual detail for downstream analytics."
                ),
                "feature": "underexposure_pct",
                "value": round(under_pct, 2),
            }
        )

    elif pixel_underexposed or (
        brightness_underexposed and under_pct >= 5
    ):
        issues.append(
            {
                "type": "underexposure",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Underexposure detected",
                "message": (
                    "Dark regions occupy a significant portion of the image. "
                    "Object and scene details may be difficult to analyse."
                ),
                "feature": "underexposure_pct",
                "value": round(under_pct, 2),
            }
        )

    # ------------------------------------------------------------------
    # 3. Overexposure
    # ------------------------------------------------------------------
    #
    # PRIMARY SIGNAL:
    # Percentage of pixels above the highlight clipping threshold.
    #
    # SUPPORTING SIGNAL:
    # Mean brightness.
    #
    # Only ONE final overexposure issue can be created.

    brightness_overexposed = brightness > 185
    brightness_severe_overexposed = brightness > 220

    pixel_overexposed = over_pct >= 10
    pixel_severe_overexposed = over_pct >= 30

    if pixel_severe_overexposed or (
        over_pct >= 20 and brightness_severe_overexposed
    ):
        issues.append(
            {
                "type": "severe_overexposure",
                "severity": "CRITICAL",
                "confidence": 1.0,
                "title": "Severe overexposure detected",
                "message": (
                    "A large portion of the image is affected by clipped bright "
                    "regions, resulting in substantial loss of visual information."
                ),
                "feature": "overexposure_pct",
                "value": round(over_pct, 2),
            }
        )

    elif pixel_overexposed or (
        brightness_overexposed and over_pct >= 3
    ):
        issues.append(
            {
                "type": "overexposure",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Overexposure detected",
                "message": (
                    "Bright regions are clipping visual information and may "
                    "reduce the reliability of downstream image analytics."
                ),
                "feature": "overexposure_pct",
                "value": round(over_pct, 2),
            }
        )

    # ------------------------------------------------------------------
    # 4. Image noise
    # ------------------------------------------------------------------

    if noise >= 35:
        issues.append(
            {
                "type": "severe_noise",
                "severity": "CRITICAL",
                "confidence": 1.0,
                "title": "Severe image noise detected",
                "message": (
                    "Noise levels are high enough to significantly obscure "
                    "scene details and reduce downstream model reliability."
                ),
                "feature": "noise_estimate",
                "value": round(noise, 2),
            }
        )

    elif noise >= 15:
        issues.append(
            {
                "type": "image_noise",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Elevated image noise",
                "message": (
                    "Visible noise may interfere with object boundaries and "
                    "fine visual details."
                ),
                "feature": "noise_estimate",
                "value": round(noise, 2),
            }
        )

    # ------------------------------------------------------------------
    # 5. Image corruption / severe degradation
    # ------------------------------------------------------------------
    #
    # Important:
    # Do NOT classify one weak feature alone as severe degradation.
    #
    # Severe degradation is inferred from combinations indicating that
    # the image has lost significant usable visual information.

    degradation_signals = 0

    if sharpness < 15:
        degradation_signals += 1

    if entropy < 3.0:
        degradation_signals += 1

    if dynamic_range < 40:
        degradation_signals += 1

    if edge_density < 1.0:
        degradation_signals += 1

    if contrast < 15:
        degradation_signals += 1

    if degradation_signals >= 3:
        issues.append(
            {
                "type": "severe_visual_degradation",
                "severity": "CRITICAL",
                "confidence": 1.0,
                "title": "Severe visual degradation detected",
                "message": (
                    "Multiple image-quality signals indicate substantial loss "
                    "of usable visual information. The image may be corrupted "
                    "or severely degraded and is not recommended for reliable "
                    "downstream analytics."
                ),
                "feature": "degradation_signals",
                "value": float(degradation_signals),
            }
        )

    elif degradation_signals == 2:
        issues.append(
            {
                "type": "visual_degradation",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Significant visual degradation detected",
                "message": (
                    "Multiple quality indicators show reduced visual information. "
                    "Downstream computer vision results may be less reliable."
                ),
                "feature": "degradation_signals",
                "value": float(degradation_signals),
            }
        )

    # ------------------------------------------------------------------
    # 6. Potential visual defect
    # ------------------------------------------------------------------
    #
    # This is intentionally a broad, explainable category for unusual
    # feature combinations that do not map cleanly to blur/exposure/noise.
    #
    # Examples:
    # - extremely low color information
    # - unusually low entropy
    # - extremely low texture
    #
    # Avoid creating a defect for normal grayscale/night scenes.

    visual_defect_signals = 0

    if entropy < 2.5:
        visual_defect_signals += 1

    if texture_complexity < 0.5:
        visual_defect_signals += 1

    if dynamic_range < 30:
        visual_defect_signals += 1

    if edge_density < 0.5:
        visual_defect_signals += 1

    if (
        visual_defect_signals >= 2
        and degradation_signals < 3
    ):
        issues.append(
            {
                "type": "potential_visual_defect",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Potential visual defect detected",
                "message": (
                    "Unusual visual characteristics suggest a possible capture, "
                    "sensor, compression, or image-processing defect. Manual "
                    "inspection is recommended."
                ),
                "feature": "visual_defect_signals",
                "value": float(visual_defect_signals),
            }
        )

    # ------------------------------------------------------------------
    # 7. Additional technically justified issues
    # ------------------------------------------------------------------

    # Low contrast
    if contrast < 20 and dynamic_range >= 40:
        issues.append(
            {
                "type": "low_contrast",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Low image contrast",
                "message": (
                    "Weak contrast may reduce separation between objects and "
                    "their background."
                ),
                "feature": "contrast",
                "value": round(contrast, 2),
            }
        )

    # Very limited dynamic range
    if dynamic_range < 35 and degradation_signals < 3:
        issues.append(
            {
                "type": "limited_dynamic_range",
                "severity": "HIGH",
                "confidence": 1.0,
                "title": "Limited dynamic range",
                "message": (
                    "The image contains a narrow range of intensity values, "
                    "which may hide useful scene detail."
                ),
                "feature": "dynamic_range",
                "value": round(dynamic_range, 2),
            }
        )

    # Very low saturation can be useful to flag in daylight colour imagery,
    # but only when the scene otherwise has sufficient brightness.
    if saturation < 15 and brightness > 80 and colorfulness < 15:
        issues.append(
            {
                "type": "low_color_information",
                "severity": "LOW",
                "confidence": 1.0,
                "title": "Low colour information",
                "message": (
                    "Colour information is limited. This may indicate a "
                    "desaturated source, grayscale camera mode, or processing artifact."
                ),
                "feature": "saturation",
                "value": round(saturation, 2),
            }
        )

    return issues