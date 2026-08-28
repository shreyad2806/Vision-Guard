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

from typing import Any, Mapping


SEVERITY_LOW = "low"
SEVERITY_MODERATE = "moderate"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"


def build_issue(
    issue_type: str,
    severity: str,
    metric: str,
    value: float,
    threshold: float,
    impact: str,
    confidence: float = 1.0,
    description: str | None = None,
) -> dict[str, Any]:
    """Build a normalized, explainable issue response."""

    valid_severities = {
        SEVERITY_LOW,
        SEVERITY_MODERATE,
        SEVERITY_HIGH,
        SEVERITY_CRITICAL,
    }

    if severity not in valid_severities:
        raise ValueError(
            f"Invalid severity '{severity}'. "
            f"Expected one of: {sorted(valid_severities)}"
        )

    issue = {
        "type": issue_type,
        "severity": severity,
        "metric": metric,
        "value": round(float(value), 4),
        "threshold": round(float(threshold), 4),
        "impact": impact,
        "confidence": round(
            max(0.0, min(float(confidence), 1.0)),
            2,
        ),
    }

    if description:
        issue["description"] = description

    return issue

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

    if sharpness < 15:
        issues.append(
            build_issue(
                issue_type="severe_blur",
                severity=SEVERITY_CRITICAL,
                metric="laplacian_variance",
                value=sharpness,
                threshold=15.0,
                impact=(
                    "Fine details, object boundaries, pedestrians, and vehicles "
                    "may not be reliably detected."
                ),
                confidence=1.0,
                description=(
                    "Image sharpness is critically low, indicating severe blur "
                    "or substantial loss of high-frequency detail."
                ),
            )
        )

    elif sharpness < 40:
        issues.append(
            build_issue(
                issue_type="insufficient_sharpness",
                severity=SEVERITY_HIGH,
                metric="laplacian_variance",
                value=sharpness,
                threshold=40.0,
                impact=(
                    "Reduced edge detail may lower detection reliability for "
                    "small, distant, or fast-moving objects."
                ),
                confidence=1.0,
                description=(
                    "Image sharpness is below the recommended range."
            ),
        )
    )

    # ------------------------------------------------------------------
    # 2. Underexposure
    # ------------------------------------------------------------------

    brightness_underexposed = brightness < 70
    brightness_severe_underexposed = brightness < 40

    pixel_underexposed = under_pct >= 15
    pixel_severe_underexposed = under_pct >= 40

    if pixel_severe_underexposed or (
    under_pct >= 35 and brightness_severe_underexposed
    ):
        issues.append(
            build_issue(
                issue_type="underexposure",
                severity=SEVERITY_CRITICAL,
                metric="dark_pixel_ratio",
                value=under_pct,
                threshold=50.0,
                impact=(
                    "Severe shadow clipping may hide pedestrians, vehicles, "
                    "road markings, and other scene details."
                ),
                confidence=1.0,
                description=(
                    "A large portion of the image contains severely dark pixels."
                ),
            )
        )

    elif pixel_underexposed or (
        brightness_underexposed and under_pct >= 10
    ):
        issues.append(
            build_issue(
                issue_type="underexposure",
                severity=SEVERITY_HIGH,
                metric="dark_pixel_ratio",
                value=under_pct,
                threshold=25.0,
                impact=(
                    "May reduce pedestrian and vehicle detection reliability."
                ),
                confidence=1.0,
                description=(
                    "A significant portion of the image is underexposed."
                ),
            )
        )
    # ------------------------------------------------------------------
    # 3. Overexposure
    # ------------------------------------------------------------------

    brightness_overexposed = brightness > 185
    brightness_severe_overexposed = brightness > 220

    pixel_overexposed = over_pct >= 10
    pixel_severe_overexposed = over_pct >= 30

    if pixel_severe_overexposed or (
    over_pct >= 20 and brightness_severe_overexposed
    ):
        issues.append(
            build_issue(
                issue_type="overexposure",
                severity=SEVERITY_CRITICAL,
                metric="bright_pixel_ratio",
                value=over_pct,
                threshold=30.0,
                impact=(
                    "Clipped highlights may hide important visual details "
                    "in vehicles, road surfaces, signs, and bright regions."
                ),
                confidence=1.0,
                description=(
                    "A large portion of the image contains clipped bright pixels."
                ),
            )
        )

    elif pixel_overexposed or (
        brightness_overexposed and over_pct >= 3
    ):
        issues.append(
            build_issue(
                issue_type="overexposure",
                severity=SEVERITY_HIGH,
                metric="bright_pixel_ratio",
                value=over_pct,
                threshold=10.0,
                impact=(
                    "Highlight clipping may reduce reliability of downstream "
                    "visual analytics."
                ),
                confidence=1.0,
                description=(
                    "Bright regions are losing visual information."
                ),
            )
        )

    # ------------------------------------------------------------------
    # 4. Image noise
    # ------------------------------------------------------------------

    if noise >= 35:
        issues.append(
            build_issue(
                issue_type="image_noise",
                severity=SEVERITY_CRITICAL,
                metric="noise_estimate",
                value=noise,
                threshold=35.0,
                impact=(
                    "Severe noise may obscure object boundaries and significantly "
                    "reduce downstream analytics reliability."
                ),
                confidence=1.0,
                description="Image noise is critically high.",
            )
        )

    elif noise >= 15:
        issues.append(
            build_issue(
                issue_type="image_noise",
                severity=SEVERITY_HIGH,
                metric="noise_estimate",
                value=noise,
                threshold=15.0,
                impact=(
                    "Noise may interfere with object boundaries and fine details."
                ),
                confidence=1.0,
                description="Image contains elevated noise.",
            )
        )

    # ------------------------------------------------------------------
    # 5. Image corruption / severe degradation
    # ------------------------------------------------------------------

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
            build_issue(
                issue_type="severe_visual_degradation",
                severity=SEVERITY_CRITICAL,
                metric="degradation_signal_count",
                value=degradation_signals,
                threshold=3.0,
                impact=(
                    "Multiple quality failures indicate substantial loss of usable "
                    "visual information. The image may be unreliable for smart-city "
                    "analytics."
                ),
                confidence=1.0,
                description=(
                    "Multiple independent image-quality signals indicate severe "
                    "visual degradation or possible corruption."
                ),
            )
        )

    elif degradation_signals == 2:
        issues.append(
            build_issue(
                issue_type="visual_degradation",
                severity=SEVERITY_HIGH,
                metric="degradation_signal_count",
                value=degradation_signals,
                threshold=2.0,
                impact=(
                    "Multiple quality indicators may reduce the reliability of "
                    "downstream computer vision results."
                ),
                confidence=0.9,
                description=(
                    "Multiple image-quality signals show reduced visual information."
                ),
            )
        )

    # ------------------------------------------------------------------
    # 6. Potential visual defect
    # ------------------------------------------------------------------

    visual_defect_signals = 0

    if entropy < 2.5:
        visual_defect_signals += 1

    if texture_complexity < 0.5:
        visual_defect_signals += 1

    if dynamic_range < 30:
        visual_defect_signals += 1

    if edge_density < 0.5:
        visual_defect_signals += 1

    if visual_defect_signals >= 2 and degradation_signals < 3:
        issues.append(
            build_issue(
                issue_type="potential_visual_defect",
                severity=SEVERITY_HIGH,
                metric="visual_anomaly_signal_count",
                value=visual_defect_signals,
                threshold=2.0,
                impact=(
                    "Possible capture, sensor, compression, or processing defects "
                    "may reduce the reliability of downstream analytics."
                ),
                confidence=0.85,
                description=(
                    "Multiple unusual visual characteristics suggest a possible "
                    "visual anomaly or defect."
                ),
            )
        )

    # ------------------------------------------------------------------
    # 7. Additional technically justified issues
    # ------------------------------------------------------------------

    # Low contrast
    if contrast < 20 and dynamic_range >= 40:
        issues.append(
            build_issue(
                issue_type="low_contrast",
                severity=SEVERITY_HIGH,
                metric="contrast",
                value=contrast,
                threshold=20.0,
                impact=(
                    "Weak contrast may reduce separation between objects and "
                    "their background."
                ),
                confidence=1.0,
                description="Image contrast is below the recommended range.",
            )
        )

    # Very limited dynamic range
    if dynamic_range < 35 and degradation_signals < 3:
        issues.append(
            build_issue(
                issue_type="limited_dynamic_range",
                severity=SEVERITY_HIGH,
                metric="dynamic_range",
                value=dynamic_range,
                threshold=35.0,
                impact=(
                    "A narrow tonal range may hide useful scene details and "
                    "reduce feature separation."
                ),
                confidence=1.0,
                description="Image contains a limited range of intensity values.",
            )
        )

    # Very low saturation can be useful to flag in daylight colour imagery,
    # but only when the scene otherwise has sufficient brightness.
    if saturation < 15 and brightness > 80 and colorfulness < 15:
        issues.append(
            build_issue(
                issue_type="low_color_information",
                severity=SEVERITY_LOW,
                metric="saturation",
                value=saturation,
                threshold=15.0,
                impact=(
                    "Limited colour information may reduce the usefulness of "
                    "colour-based downstream analytics."
                ),
                confidence=0.8,
                description=(
                    "The image is significantly desaturated."
                ),
            )
        )

    return issues