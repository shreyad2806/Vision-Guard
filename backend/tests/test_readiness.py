"""Tests for analytics readiness scoring."""

from apps.ml.readiness import calculate_analytics_readiness


def test_high_quality_image_is_highly_ready():
    result = calculate_analytics_readiness(
        quality_score=90.0,
        issues=[],
    )

    assert result["score"] == 90.0
    assert result["status"] == "HIGHLY READY"


def test_blur_reduces_readiness():
    issues = [
        {
            "type": "severe_blur",
            "severity": "critical",
        }
    ]

    result = calculate_analytics_readiness(
        quality_score=90.0,
        issues=issues,
    )

    assert result["score"] < 90.0
    assert result["penalties"]["blur"] > 0


def test_underexposure_reduces_readiness():
    issues = [
        {
            "type": "underexposure",
            "severity": "high",
        }
    ]

    result = calculate_analytics_readiness(
        quality_score=85.0,
        issues=issues,
    )

    assert result["score"] == 70.0
    assert result["status"] == "READY"


def test_multiple_issues_reduce_readiness():
    issues = [
        {
            "type": "severe_blur",
            "severity": "critical",
        },
        {
            "type": "severe_underexposure",
            "severity": "critical",
        },
        {
            "type": "severe_noise",
            "severity": "high",
        },
    ]

    result = calculate_analytics_readiness(
        quality_score=90.0,
        issues=issues,
    )

    assert result["score"] < 40
    assert result["status"] == "NOT READY"


def test_score_never_goes_below_zero():
    issues = [
        {
            "type": "severe_blur",
            "severity": "critical",
        },
        {
            "type": "severe_underexposure",
            "severity": "critical",
        },
        {
            "type": "severe_noise",
            "severity": "critical",
        },
        {
            "type": "severe_visual_degradation",
            "severity": "critical",
        },
    ]

    result = calculate_analytics_readiness(
        quality_score=10.0,
        issues=issues,
    )

    assert result["score"] == 0.0
    assert result["status"] == "CRITICAL / REJECT"


def test_all_readiness_boundaries():
    test_cases = [
        (80, "HIGHLY READY"),
        (60, "READY"),
        (40, "LIMITED READINESS"),
        (20, "NOT READY"),
        (0, "CRITICAL / REJECT"),
    ]

    for quality_score, expected_status in test_cases:
        result = calculate_analytics_readiness(
            quality_score=quality_score,
            issues=[],
        )

        assert result["status"] == expected_status