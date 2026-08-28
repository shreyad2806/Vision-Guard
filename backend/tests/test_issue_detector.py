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
# Response contract helpers
# ---------------------------------------------------------------------------


VALID_SEVERITIES = {
    "low",
    "moderate",
    "high",
    "critical",
}


def get_issue(issues: list[dict], issue_type: str) -> dict:
    """Return exactly one issue of the requested type."""
    matching = [
        issue
        for issue in issues
        if issue["type"] == issue_type
    ]

    assert len(matching) == 1, (
        f"Expected exactly one '{issue_type}' issue, "
        f"but found {len(matching)}: {matching}"
    )

    return matching[0]


def assert_issue_contract(issue: dict) -> None:
    """Validate the Phase 2 normalized issue contract."""

    required_fields = {
        "type",
        "severity",
        "metric",
        "value",
        "threshold",
        "impact",
        "confidence",
    }

    assert required_fields.issubset(issue.keys())

    assert isinstance(issue["type"], str)
    assert issue["severity"] in VALID_SEVERITIES

    assert isinstance(issue["metric"], str)
    assert isinstance(issue["value"], float)
    assert isinstance(issue["threshold"], float)
    assert isinstance(issue["impact"], str)

    assert isinstance(issue["confidence"], float)
    assert 0.0 <= issue["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# Blur / insufficient sharpness
# ---------------------------------------------------------------------------


def test_severe_blur_detection():
    features = get_base_features()
    features["sharpness"] = 8.0

    issues = detect_issues(features)

    issue = get_issue(issues, "severe_blur")

    assert issue["severity"] == "critical"
    assert_issue_contract(issue)


def test_insufficient_sharpness_detection():
    features = get_base_features()
    features["sharpness"] = 25.0

    issues = detect_issues(features)

    issue = get_issue(issues, "insufficient_sharpness")

    assert issue["severity"] in {"moderate", "high"}
    assert_issue_contract(issue)


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
        if issue["type"] == "underexposure"
    ]

    assert len(exposure_issues) == 1

    issue = exposure_issues[0]

    assert issue["severity"] in {
        "moderate",
        "high",
    }

    assert_issue_contract(issue)


def test_severe_underexposure_detected_once():
    features = get_base_features()
    features["brightness"] = 30.0
    features["underexposure_pct"] = 50.0

    issues = detect_issues(features)

    exposure_issues = [
        issue
        for issue in issues
        if issue["type"] == "underexposure"
    ]

    assert len(exposure_issues) == 1

    issue = exposure_issues[0]

    # Phase 2 keeps the issue type stable.
    # Severity communicates how severe the problem is.
    assert issue["type"] == "underexposure"
    assert issue["severity"] == "critical"

    assert_issue_contract(issue)


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
        if issue["type"] == "overexposure"
    ]

    assert len(exposure_issues) == 1

    issue = exposure_issues[0]

    assert issue["severity"] in {
        "moderate",
        "high",
    }

    assert_issue_contract(issue)


def test_severe_overexposure_detected_once():
    features = get_base_features()
    features["brightness"] = 230.0
    features["overexposure_pct"] = 40.0

    issues = detect_issues(features)

    exposure_issues = [
        issue
        for issue in issues
        if issue["type"] == "overexposure"
    ]

    assert len(exposure_issues) == 1

    issue = exposure_issues[0]

    # Stable issue type + separate severity.
    assert issue["type"] == "overexposure"
    assert issue["severity"] == "critical"

    assert_issue_contract(issue)


# ---------------------------------------------------------------------------
# Image noise
# ---------------------------------------------------------------------------


def test_image_noise_detection():
    features = get_base_features()
    features["noise_estimate"] = 20.0

    issues = detect_issues(features)

    issue = get_issue(issues, "image_noise")

    assert issue["severity"] in {
        "moderate",
        "high",
    }

    assert_issue_contract(issue)


def test_severe_noise_detection():
    features = get_base_features()
    features["noise_estimate"] = 45.0

    issues = detect_issues(features)

    noise_issues = [
        issue
        for issue in issues
        if issue["type"] == "image_noise"
    ]

    assert len(noise_issues) == 1

    issue = noise_issues[0]

    # Phase 2 uses one stable issue type.
    assert issue["type"] == "image_noise"
    assert issue["severity"] == "critical"

    assert_issue_contract(issue)


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

    issue = get_issue(issues, "severe_visual_degradation")

    assert issue["severity"] == "critical"
    assert_issue_contract(issue)


def test_single_bad_feature_does_not_trigger_severe_degradation():
    features = get_base_features()

    features["sharpness"] = 8.0

    issues = detect_issues(features)

    issue_types = [
        issue["type"]
        for issue in issues
    ]

    assert "severe_visual_degradation" not in issue_types


# ---------------------------------------------------------------------------
# Potential visual defect
# ---------------------------------------------------------------------------


def test_potential_visual_defect_detection():
    features = get_base_features()

    features["entropy"] = 2.0
    features["texture_complexity"] = 0.2

    issues = detect_issues(features)

    issue = get_issue(issues, "potential_visual_defect")

    assert issue["severity"] in {
        "high",
        "critical",
    }

    assert_issue_contract(issue)


# ---------------------------------------------------------------------------
# Additional technically justified issues
# ---------------------------------------------------------------------------


def test_low_contrast_detection():
    features = get_base_features()

    features["contrast"] = 10.0
    features["dynamic_range"] = 80.0

    issues = detect_issues(features)

    issue = get_issue(issues, "low_contrast")

    assert issue["severity"] in {
        "moderate",
        "high",
        "critical",
    }

    assert_issue_contract(issue)


def test_limited_dynamic_range_detection():
    features = get_base_features()

    features["dynamic_range"] = 30.0

    issues = detect_issues(features)

    issue = get_issue(issues, "limited_dynamic_range")

    assert issue["severity"] in {
        "moderate",
        "high",
        "critical",
    }

    assert_issue_contract(issue)


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


def test_all_detected_issues_follow_phase2_contract():
    """Every detector output must follow the normalized Phase 2 schema."""

    features = get_base_features()

    # Trigger multiple issue detectors.
    features["sharpness"] = 8.0
    features["brightness"] = 30.0
    features["underexposure_pct"] = 50.0
    features["noise_estimate"] = 45.0

    issues = detect_issues(features)

    assert len(issues) > 0

    for issue in issues:
        assert_issue_contract(issue)