"""Tests for rule-based image quality issue detection."""

from apps.ml.issue_detector import detect_issues


def get_base_features() -> dict[str, float]:
    """Return a healthy baseline feature set."""

    return {
        "brightness": 120.0,
        "contrast": 40.0,
        "sharpness": 100.0,
        "noise_estimate": 5.0,
        "saturation": 50.0,
        "entropy": 6.0,
        "dynamic_range": 100.0,
        "edge_density": 10.0,
        "underexposure_pct": 5.0,
        "overexposure_pct": 2.0,
        "colorfulness": 30.0,
        "texture_complexity": 10.0,
    }


# ---------------------------------------------------------------------------
# Blur / insufficient sharpness
# ---------------------------------------------------------------------------

def test_severe_blur_detection():
    features = get_base_features()
    features["sharpness"] = 8.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "severe_blur" in issue_types


def test_insufficient_sharpness_detection():
    features = get_base_features()
    features["sharpness"] = 25.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "insufficient_sharpness" in issue_types


# ---------------------------------------------------------------------------
# Underexposure
# ---------------------------------------------------------------------------

def test_underexposure_detected_once():
    features = get_base_features()
    features["brightness"] = 60.0
    features["underexposure_pct"] = 20.0

    issues = detect_issues(features)

    exposure_issues = [
        issue
        for issue in issues
        if "underexposure" in issue["type"]
    ]

    assert len(exposure_issues) == 1
    assert exposure_issues[0]["type"] == "underexposure"


def test_severe_underexposure_detected_once():
    features = get_base_features()
    features["brightness"] = 30.0
    features["underexposure_pct"] = 50.0

    issues = detect_issues(features)

    exposure_issues = [
        issue
        for issue in issues
        if "underexposure" in issue["type"]
    ]

    assert len(exposure_issues) == 1
    assert exposure_issues[0]["type"] == "severe_underexposure"


# ---------------------------------------------------------------------------
# Overexposure
# ---------------------------------------------------------------------------

def test_overexposure_detected_once():
    features = get_base_features()
    features["brightness"] = 190.0
    features["overexposure_pct"] = 15.0

    issues = detect_issues(features)

    exposure_issues = [
        issue
        for issue in issues
        if "overexposure" in issue["type"]
    ]

    assert len(exposure_issues) == 1
    assert exposure_issues[0]["type"] == "overexposure"


def test_severe_overexposure_detected_once():
    features = get_base_features()
    features["brightness"] = 230.0
    features["overexposure_pct"] = 40.0

    issues = detect_issues(features)

    exposure_issues = [
        issue
        for issue in issues
        if "overexposure" in issue["type"]
    ]

    assert len(exposure_issues) == 1
    assert exposure_issues[0]["type"] == "severe_overexposure"


# ---------------------------------------------------------------------------
# Image noise
# ---------------------------------------------------------------------------

def test_image_noise_detection():
    features = get_base_features()
    features["noise_estimate"] = 20.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "image_noise" in issue_types


def test_severe_noise_detection():
    features = get_base_features()
    features["noise_estimate"] = 45.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "severe_noise" in issue_types


# ---------------------------------------------------------------------------
# Severe visual degradation
# ---------------------------------------------------------------------------

def test_severe_visual_degradation_detection():
    features = get_base_features()

    features["sharpness"] = 8.0
    features["entropy"] = 2.0
    features["dynamic_range"] = 25.0
    features["edge_density"] = 0.4
    features["contrast"] = 10.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "severe_visual_degradation" in issue_types


def test_single_bad_feature_does_not_trigger_severe_degradation():
    features = get_base_features()

    features["sharpness"] = 8.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "severe_visual_degradation" not in issue_types


# ---------------------------------------------------------------------------
# Potential visual defect
# ---------------------------------------------------------------------------

def test_potential_visual_defect_detection():
    features = get_base_features()

    features["entropy"] = 2.0
    features["texture_complexity"] = 0.2

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "potential_visual_defect" in issue_types


# ---------------------------------------------------------------------------
# Additional justified issues
# ---------------------------------------------------------------------------

def test_low_contrast_detection():
    features = get_base_features()

    features["contrast"] = 10.0
    features["dynamic_range"] = 80.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "low_contrast" in issue_types


def test_limited_dynamic_range_detection():
    features = get_base_features()

    features["dynamic_range"] = 30.0

    issues = detect_issues(features)

    issue_types = [issue["type"] for issue in issues]

    assert "limited_dynamic_range" in issue_types


# ---------------------------------------------------------------------------
# Healthy image
# ---------------------------------------------------------------------------

def test_healthy_features_do_not_create_critical_issues():
    features = get_base_features()

    issues = detect_issues(features)

    critical_issues = [
        issue
        for issue in issues
        if issue["severity"] == "critical"
    ]

    assert len(critical_issues) == 0