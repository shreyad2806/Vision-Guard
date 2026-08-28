"""Smart-city context definitions for Phase 4.

Maps pipeline contexts to context-specific impact messages for each
detected image-quality issue. When no context-specific message exists,
the original generic issue impact is used as fallback.
"""

from __future__ import annotations

DEFAULT_CONTEXT = "CCTV Surveillance"

SUPPORTED_CONTEXTS: tuple[str, ...] = (
    "CCTV Surveillance",
    "Traffic Monitoring",
    "Crowd Monitoring",
    "Drone Imagery",
    "Infrastructure Inspection",
    "Smart Campus",
)

# Mapping: context -> issue_type -> context-specific impact message
# Only issue types that have a specialised message per context are listed.
# All other issue types fall back to the generic impact already attached
# to each issue by the issue_detector.
_CONTEXT_IMPACTS: dict[str, dict[str, str]] = {
    "CCTV Surveillance": {
        "severe_blur": "Motion-blurred or out-of-focus frames reduce the reliability of automated surveillance analytics such as person re-identification and anomaly detection.",
        "insufficient_sharpness": "Reduced sharpness impairs facial recognition and incident review from CCTV footage.",
        "underexposure": "Low-light scenes reduce person detection and event classification accuracy in surveillance feeds.",
        "severe_underexposure": "Near-dark footage is unreliable for real-time surveillance monitoring or post-event forensic review.",
        "overexposure": "Overexposed areas cause glare that obscures faces and activity in surveillance recordings.",
        "severe_overexposure": "Severe overexposure washes out critical scene detail, rendering surveillance analytics ineffective.",
        "image_noise": "Sensor noise degrades motion detection and object tracking in surveillance feeds.",
        "severe_noise": "Severe noise makes automated person and vehicle detection unreliable in surveillance footage.",
        "severe_visual_degradation": "Multiple quality failures compromise the reliability of CCTV-based situational awareness.",
        "potential_visual_defect": "A potential visual defect may mask important activity in surveillance footage.",
        "low_contrast": "Low contrast limits the ability to distinguish subjects from the background in surveillance scenes.",
        "limited_dynamic_range": "Limited dynamic range reduces detail in shadow and highlight regions of surveillance footage.",
        "low_color_information": "Desaturated images reduce the effectiveness of colour-based re-identification in surveillance.",
    },
    "Traffic Monitoring": {
        "severe_blur": "Vehicle detection and license-plate recognition may be unreliable.",
        "insufficient_sharpness": "Reduced sharpness impairs vehicle classification and licence-plate reading.",
        "underexposure": "Low-light scenes reduce vehicle and lane-marking detection accuracy.",
        "severe_underexposure": "Near-dark footage cannot support reliable traffic-flow or vehicle-count analytics.",
        "overexposure": "Glare from overexposure obscures traffic signals and road markings.",
        "severe_overexposure": "Severe overexposure washes out road detail, making traffic monitoring ineffective.",
        "image_noise": "Sensor noise degrades vehicle detection and speed-estimation algorithms.",
        "severe_noise": "Severe noise prevents reliable vehicle tracking and counting.",
        "severe_visual_degradation": "Multiple quality failures render the image unreliable for traffic-flow analytics.",
        "potential_visual_defect": "A potential visual defect may cause false positives in traffic monitoring.",
        "low_contrast": "Low contrast limits separation between vehicles and the road surface.",
        "limited_dynamic_range": "Limited dynamic range reduces visibility of vehicles in shadow and glare zones.",
        "low_color_information": "Desaturated images reduce the effectiveness of colour-based vehicle classification.",
    },
    "Crowd Monitoring": {
        "severe_blur": "Person detection and crowd-density estimation confidence may decrease.",
        "insufficient_sharpness": "Reduced sharpness impairs individual person detection in dense crowds.",
        "underexposure": "Low-light scenes reduce person detection and crowd-density estimation accuracy.",
        "severe_underexposure": "Near-dark footage cannot support reliable crowd monitoring or density estimation.",
        "overexposure": "Overexposure obscures individual people in bright crowd scenes.",
        "severe_overexposure": "Severe overexposure makes it impossible to distinguish individuals in a crowd.",
        "image_noise": "Sensor noise degrades person detection and tracking in crowded scenes.",
        "severe_noise": "Severe noise prevents reliable crowd counting and flow analysis.",
        "severe_visual_degradation": "Multiple quality failures compromise crowd-safety monitoring reliability.",
        "potential_visual_defect": "A potential visual defect may obscure part of a crowd scene.",
        "low_contrast": "Low contrast limits the ability to separate individuals in dense crowd scenes.",
        "limited_dynamic_range": "Limited dynamic range reduces detail in shadowed or brightly lit crowd areas.",
        "low_color_information": "Desaturated images reduce the effectiveness of appearance-based person re-identification.",
    },
    "Drone Imagery": {
        "severe_blur": "Aerial survey and land-use classification accuracy may be significantly reduced.",
        "insufficient_sharpness": "Reduced sharpness impairs ground-feature extraction from aerial imagery.",
        "underexposure": "Underexposed aerial images lose terrain detail needed for land-use classification.",
        "severe_underexposure": "Near-dark aerial footage cannot support reliable ground analysis.",
        "overexposure": "Overexposure from reflective surfaces obscures ground detail in drone imagery.",
        "severe_overexposure": "Severe overexposure washes out terrain features in aerial survey data.",
        "image_noise": "Sensor noise degrades terrain feature extraction and change detection.",
        "severe_noise": "Severe noise prevents reliable land-use classification from aerial imagery.",
        "severe_visual_degradation": "Multiple quality failures render aerial imagery unreliable for automated survey analysis.",
        "potential_visual_defect": "A potential visual defect may mask terrain features in drone imagery.",
        "low_contrast": "Low contrast limits the ability to distinguish land-cover types from aerial views.",
        "limited_dynamic_range": "Limited dynamic range reduces detail across varied terrain in drone surveys.",
        "low_color_information": "Desaturated aerial images reduce the effectiveness of spectral land-use analysis.",
    },
    "Infrastructure Inspection": {
        "severe_blur": "Small structural defects may not be reliably visible.",
        "insufficient_sharpness": "Reduced sharpness prevents detection of fine cracks, corrosion, or surface damage.",
        "underexposure": "Underexposed areas of a structure may hide defects in shadowed regions.",
        "severe_underexposure": "Near-dark conditions prevent any meaningful structural defect inspection.",
        "overexposure": "Overexposure washes out surface texture detail needed for defect detection.",
        "severe_overexposure": "Severe overexposure eliminates surface detail critical for infrastructure assessment.",
        "image_noise": "Sensor noise may mask fine surface defects such as hairline cracks.",
        "severe_noise": "Severe noise makes automated defect detection unreliable.",
        "severe_visual_degradation": "Multiple quality failures prevent reliable structural health assessment.",
        "potential_visual_defect": "A potential visual defect may be a real structural issue requiring closer inspection.",
        "low_contrast": "Low contrast limits the visibility of surface-level structural defects.",
        "limited_dynamic_range": "Limited dynamic range hides defects in high-contrast structural scenes.",
        "low_color_information": "Desaturated images reduce the effectiveness of corrosion and stain detection.",
    },
    "Smart Campus": {
        "severe_blur": "Building access and occupancy analytics may be unreliable.",
        "insufficient_sharpness": "Reduced sharpness impairs person and vehicle detection across campus feeds.",
        "underexposure": "Low-light scenes reduce occupancy detection and parking analytics accuracy.",
        "severe_underexposure": "Near-dark footage cannot support reliable smart-campus services.",
        "overexposure": "Glare from overexposure reduces the accuracy of campus-space utilisation analytics.",
        "severe_overexposure": "Severe overexposure renders campus monitoring feeds ineffective.",
        "image_noise": "Sensor noise degrades person detection and facial recognition on campus cameras.",
        "severe_noise": "Severe noise prevents reliable occupancy and access control analytics.",
        "severe_visual_degradation": "Multiple quality failures compromise smart-campus situational awareness.",
        "potential_visual_defect": "A potential visual defect may affect space-utilisation or security analytics.",
        "low_contrast": "Low contrast limits the ability to detect and track people in campus environments.",
        "limited_dynamic_range": "Limited dynamic range reduces detail in mixed indoor-outdoor campus scenes.",
        "low_color_information": "Desaturated images reduce the effectiveness of appearance-based person identification.",
    },
}


def get_context_impacts(
    issues: list[dict],
    context: str,
) -> list[dict]:
    """Build context-specific impact entries for detected issues.

    For each detected issue, look up a context-specific impact message.
    If none exists, fall back to the generic impact already on the issue.

    Returns a list of dicts matching the ContextImpact contract:
        {issue_type, context, impact}
    """
    impacts: list[dict] = []
    ctx_map = _CONTEXT_IMPACTS.get(context, {})

    for issue in issues:
        issue_type = issue.get("type", "")
        # Context-specific message takes priority
        impact = ctx_map.get(issue_type) or issue.get("impact", "")
        impacts.append({
            "issue_type": issue_type,
            "context": context,
            "impact": impact,
        })

    return impacts
