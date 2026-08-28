"""Tests for Phase 6 — Issue-driven explainability (VisionGuard)."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.main import app
from apps.ml.explainability import generate_issue_explanations
from apps.ml.issue_detector import detect_issues


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_jpeg():
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def _upload(client, sample_jpeg, context=None):
    url = "/api/v1/analyze"
    if context is not None:
        url = f"{url}?context={context}"
    return client.post(
        url,
        files={"file": ("test.jpg", sample_jpeg, "image/jpeg")},
    )


# ---------------------------------------------------------------------------
# Healthy baseline features (no issues expected)
# ---------------------------------------------------------------------------

def _healthy_features() -> dict[str, float]:
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


# ===================================================================
# Unit tests — generate_issue_explanations()
# ===================================================================

class TestGenerateIssueExplanations:
    """Direct tests of the explainability module."""

    def test_empty_issues_returns_empty_list(self):
        result = generate_issue_explanations([])
        assert result == []

    def test_explanation_generated_for_severe_blur(self):
        issues = detect_issues({"sharpness": 8.0, **{
            k: v for k, v in _healthy_features().items() if k != "sharpness"
        }})
        blur_issues = [i for i in issues if i["type"] == "severe_blur"]
        assert len(blur_issues) == 1

        explanations = generate_issue_explanations(issues)
        blur_exps = [e for e in explanations if "blur" in e["issue"].lower() or "sharp" in e["issue"].lower()]
        assert len(blur_exps) >= 1

    def test_explanation_contains_actual_metric_value(self):
        """The evidence.value must match the actual metric value from the detector."""
        features = _healthy_features()
        features["sharpness"] = 12.0  # triggers severe_blur
        issues = detect_issues(features)
        blur_issue = [i for i in issues if i["type"] == "severe_blur"][0]

        explanations = generate_issue_explanations([blur_issue])
        assert len(explanations) == 1
        assert explanations[0]["evidence"]["value"] == blur_issue["value"]
        assert explanations[0]["evidence"]["value"] == 12.0

    def test_explanation_contains_actual_threshold(self):
        """The evidence.threshold must match the detection threshold."""
        features = _healthy_features()
        features["sharpness"] = 12.0
        issues = detect_issues(features)
        blur_issue = [i for i in issues if i["type"] == "severe_blur"][0]

        explanations = generate_issue_explanations([blur_issue])
        assert explanations[0]["evidence"]["threshold"] == 15.0

    def test_explanation_corresponds_to_detected_issue(self):
        """Each explanation must map 1:1 to its source issue."""
        features = _healthy_features()
        features["sharpness"] = 12.0
        features["noise_estimate"] = 40.0
        issues = detect_issues(features)
        assert len(issues) >= 2  # at least blur + noise

        explanations = generate_issue_explanations(issues)
        assert len(explanations) == len(issues)

    def test_explanation_structure_has_required_keys(self):
        features = _healthy_features()
        features["sharpness"] = 12.0
        issues = detect_issues(features)
        explanations = generate_issue_explanations(issues)
        assert len(explanations) >= 1
        exp = explanations[0]
        assert "issue" in exp
        assert "evidence" in exp
        assert "why_it_matters" in exp
        assert "recommendation" in exp
        assert isinstance(exp["issue"], str)
        assert isinstance(exp["evidence"], dict)
        assert "metric" in exp["evidence"]
        assert "value" in exp["evidence"]
        assert "threshold" in exp["evidence"]
        assert isinstance(exp["why_it_matters"], str)
        assert isinstance(exp["recommendation"], str)

    def test_explanation_non_empty_strings(self):
        features = _healthy_features()
        features["sharpness"] = 12.0
        issues = detect_issues(features)
        explanations = generate_issue_explanations(issues)
        for exp in explanations:
            assert len(exp["issue"]) > 0
            assert len(exp["why_it_matters"]) > 0
            assert len(exp["recommendation"]) > 0

    def test_all_issue_types_have_explanations(self):
        """Every issue type the detector can produce must have an explanation mapping."""
        all_issue_types = [
            ("severe_blur", {"sharpness": 8.0}),
            ("insufficient_sharpness", {"sharpness": 25.0}),
            ("underexposure", {"brightness": 30.0, "underexposure_pct": 50.0}),
            ("overexposure", {"brightness": 230.0, "overexposure_pct": 40.0}),
            ("image_noise", {"noise_estimate": 40.0}),
            ("low_contrast", {"contrast": 10.0, "dynamic_range": 80.0}),
            ("limited_dynamic_range", {"dynamic_range": 30.0}),
            ("low_color_information", {"saturation": 5.0, "brightness": 100.0, "colorfulness": 10.0}),
        ]

        for expected_type, overrides in all_issue_types:
            features = _healthy_features()
            features.update(overrides)
            issues = detect_issues(features)
            matching = [i for i in issues if i["type"] == expected_type]
            assert len(matching) >= 1, f"No {expected_type} issue detected with overrides {overrides}"

            explanations = generate_issue_explanations(matching)
            assert len(explanations) >= 1, f"No explanation for {expected_type}"
            assert len(explanations[0]["why_it_matters"]) > 0

    def test_severe_degradation_explanation(self):
        features = _healthy_features()
        features["sharpness"] = 8.0
        features["entropy"] = 2.0
        features["dynamic_range"] = 25.0
        features["edge_density"] = 0.4
        features["contrast"] = 10.0
        issues = detect_issues(features)
        degradation = [i for i in issues if i["type"] == "severe_visual_degradation"]
        if degradation:
            explanations = generate_issue_explanations(degradation)
            assert len(explanations) == 1
            assert "degradation" in explanations[0]["issue"].lower()

    def test_potential_visual_defect_explanation(self):
        features = _healthy_features()
        features["entropy"] = 2.0
        features["texture_complexity"] = 0.2
        issues = detect_issues(features)
        defect_issues = [i for i in issues if i["type"] == "potential_visual_defect"]
        if defect_issues:
            explanations = generate_issue_explanations(defect_issues)
            assert len(explanations) == 1
            assert "defect" in explanations[0]["issue"].lower()


# ===================================================================
# Unit tests — healthy image (no issues → no explanations)
# ===================================================================

class TestHealthyImageNoExplanations:
    def test_healthy_features_produce_no_issues(self):
        features = _healthy_features()
        issues = detect_issues(features)
        assert len(issues) == 0

    def test_healthy_features_produce_no_explanations(self):
        features = _healthy_features()
        issues = detect_issues(features)
        explanations = generate_issue_explanations(issues)
        assert explanations == []


# ===================================================================
# Integration tests — via the API
# ===================================================================

class TestAPIIssueExplanations:
    """Verify issue_explanations appear in the API response."""

    def test_api_response_contains_issue_explanations(self, client, sample_jpeg):
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        assert "issue_explanations" in data
        assert isinstance(data["issue_explanations"], list)

    def test_explanations_match_issue_count(self, client, sample_jpeg):
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        # The number of explanations should equal the number of issues
        assert len(data["issue_explanations"]) == len(data["issues"])

    def test_explanations_have_required_structure(self, client, sample_jpeg):
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        for exp in data["issue_explanations"]:
            assert "issue" in exp
            assert "evidence" in exp
            assert "why_it_matters" in exp
            assert "recommendation" in exp
            assert isinstance(exp["evidence"], dict)
            assert "metric" in exp["evidence"]
            assert "value" in exp["evidence"]
            assert "threshold" in exp["evidence"]

    def test_explanation_evidence_matches_issue(self, client, sample_jpeg):
        """For each issue, the corresponding explanation's evidence must match."""
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        issues = data["issues"]
        explanations = data["issue_explanations"]
        assert len(explanations) == len(issues)
        for issue, exp in zip(issues, explanations):
            assert exp["evidence"]["metric"] == issue["metric"]
            assert exp["evidence"]["value"] == issue["value"]
            assert exp["evidence"]["threshold"] == issue["threshold"]

    def test_explanations_persist_in_history(self, client, sample_jpeg):
        upload = _upload(client, sample_jpeg)
        assert upload.status_code == 201

        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        item = response.json()[0]
        assert "issue_explanations" in item
        assert isinstance(item["issue_explanations"], list)

    def test_explanations_persist_in_single_analysis(self, client, sample_jpeg):
        upload = _upload(client, sample_jpeg)
        assert upload.status_code == 201
        analysis_id = upload.json()["analysis_id"]

        response = client.get(f"/api/v1/analyses/{analysis_id}")
        assert response.status_code == 200
        data = response.json()
        assert "issue_explanations" in data
        assert isinstance(data["issue_explanations"], list)

    def test_phase1_fields_still_present(self, client, sample_jpeg):
        """Phase 1 quality_score and quality_label must not be broken."""
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        assert "quality_score" in data
        assert "quality_label" in data
        assert 0 <= data["quality_score"] <= 100

    def test_phase2_issues_still_present(self, client, sample_jpeg):
        """Phase 2 issue contract must be preserved."""
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        for issue in data["issues"]:
            assert "type" in issue
            assert "severity" in issue
            assert "metric" in issue
            assert "value" in issue
            assert "threshold" in issue
            assert "impact" in issue

    def test_phase3_readiness_still_present(self, client, sample_jpeg):
        """Phase 3 analytics readiness fields must be preserved."""
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        assert "analytics_readiness_score" in data
        assert "analytics_readiness_status" in data
        assert "analytics_readiness_details" in data
        details = data["analytics_readiness_details"]
        for key in ("base_quality_score", "blur_penalty", "exposure_penalty",
                     "noise_penalty", "corruption_penalty", "information_penalty"):
            assert key in details

    def test_phase4_context_still_present(self, client, sample_jpeg):
        """Phase 4 context and context_impacts must be preserved."""
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        assert "context" in data
        assert "context_impacts" in data
        assert data["context"] == "CCTV Surveillance"
        assert isinstance(data["context_impacts"], list)
