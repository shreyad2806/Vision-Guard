"""Tests for VisionGuard backend API endpoints."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    """Create a test client with lifespan context."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_jpeg():
    """Create a small JPEG image for testing."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    return buf


def upload_sample_image(client: TestClient, sample_jpeg):
    """Upload a sample image and return the response."""
    return client.post(
        "/api/v1/analyze",
        files={
            "file": (
                "test.jpg",
                sample_jpeg,
                "image/jpeg",
            )
        },
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:

    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")

        assert response.status_code == 200

    def test_health_body(self, client: TestClient):
        response = client.get("/health")

        data = response.json()

        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert "version" in data


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

class TestDocs:

    def test_swagger_available(self, client: TestClient):
        response = client.get("/docs")

        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Analyse Image
# ---------------------------------------------------------------------------

class TestAnalyze:

    def test_valid_image_returns_201(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

    def test_valid_image_response_structure(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        required_keys = {
            "analysis_id",
            "filename",
            "image_url",
            "quality_score",
            "quality_label",
            "analysis_confidence",
            "analytics_readiness_score",
            "analytics_readiness_status",
            "analytics_readiness_details",
            "issues",
            "statistics",
            "summary",
            "recommendation",
            "processing_time_ms",
            "model_version",
            "explanations",
        }

        assert required_keys.issubset(data.keys())

    # -----------------------------------------------------------------------
    # Phase 3 - Analytics Readiness Score
    # -----------------------------------------------------------------------

    def test_analytics_readiness_score_in_range(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        score = data["analytics_readiness_score"]

        assert isinstance(score, (int, float))
        assert 0 <= score <= 100

    def test_analytics_readiness_status_is_valid(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        valid_statuses = {
            "HIGHLY READY",
            "READY",
            "LIMITED READINESS",
            "NOT READY",
            "CRITICAL / REJECT",
        }

        assert data["analytics_readiness_status"] in valid_statuses

    def test_analytics_readiness_details_exists(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        assert "analytics_readiness_details" in data

        details = data["analytics_readiness_details"]

        assert isinstance(details, dict)

    def test_analytics_readiness_details_has_required_keys(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        details = response.json()["analytics_readiness_details"]

        required_keys = {
            "base_quality_score",
            "blur_penalty",
            "exposure_penalty",
            "noise_penalty",
            "corruption_penalty",
            "information_penalty",
        }

        assert required_keys.issubset(details.keys())

    def test_analytics_readiness_penalties_are_non_negative(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        details = response.json()["analytics_readiness_details"]

        penalty_keys = [
            "blur_penalty",
            "exposure_penalty",
            "noise_penalty",
            "corruption_penalty",
            "information_penalty",
        ]

        for key in penalty_keys:
            assert isinstance(details[key], (int, float))
            assert details[key] >= 0

    def test_base_quality_score_is_valid(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        base_quality = (
            data["analytics_readiness_details"]
            ["base_quality_score"]
        )

        assert isinstance(base_quality, (int, float))
        assert 0 <= base_quality <= 100

    def test_readiness_status_matches_score(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        score = data["analytics_readiness_score"]
        status = data["analytics_readiness_status"]

        if score >= 80:
            assert status == "HIGHLY READY"

        elif score >= 60:
            assert status == "READY"

        elif score >= 40:
            assert status == "LIMITED READINESS"

        elif score >= 20:
            assert status == "NOT READY"

        else:
            assert status == "CRITICAL / REJECT"

    # -----------------------------------------------------------------------
    # Quality Score
    # -----------------------------------------------------------------------

    def test_quality_label_is_valid(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        assert data["quality_label"] in (
            "Excellent",
            "Good",
            "Fair",
            "Poor",
            "Critical",
        )

    def test_score_in_range(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        assert 0 <= data["quality_score"] <= 100
        assert 0 <= data["analysis_confidence"] <= 100

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------

    def test_statistics_has_all_keys(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        statistics = response.json()["statistics"]

        expected_keys = [
            "sharpness",
            "brightness",
            "contrast",
            "noise_estimate",
            "entropy",
            "saturation",
        ]

        for key in expected_keys:
            assert key in statistics

    # -----------------------------------------------------------------------
    # Explanations
    # -----------------------------------------------------------------------

    def test_explanations_is_list(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        data = response.json()

        assert isinstance(data["explanations"], list)

    # -----------------------------------------------------------------------
    # Phase 2 Issue Contract
    # -----------------------------------------------------------------------

    def test_issue_schema_is_normalized(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        issues = response.json()["issues"]

        for issue in issues:
            assert "type" in issue
            assert "severity" in issue
            assert "metric" in issue
            assert "value" in issue
            assert "threshold" in issue
            assert "impact" in issue
            assert "confidence" in issue

    def test_issue_severity_is_valid(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        issues = response.json()["issues"]

        valid_severities = {
            "low",
            "moderate",
            "high",
            "critical",
        }

        for issue in issues:
            assert issue["severity"] in valid_severities

    def test_issue_values_are_numeric(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = upload_sample_image(client, sample_jpeg)

        assert response.status_code == 201

        issues = response.json()["issues"]

        for issue in issues:
            assert isinstance(issue["value"], (int, float))
            assert isinstance(issue["threshold"], (int, float))

            if issue["confidence"] is not None:
                assert isinstance(
                    issue["confidence"],
                    (int, float),
                )
                assert 0 <= issue["confidence"] <= 1

    # -----------------------------------------------------------------------
    # Invalid Uploads
    # -----------------------------------------------------------------------

    def test_invalid_extension_returns_400(
        self,
        client: TestClient,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.txt",
                    b"hello",
                    "text/plain",
                )
            },
        )

        assert response.status_code == 400

    def test_empty_file_returns_400(
        self,
        client: TestClient,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "empty.jpg",
                    b"",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400

    def test_corrupted_image_returns_400(
        self,
        client: TestClient,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "bad.jpg",
                    b"not an image",
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:

    def test_extracts_all_features(self):
        import numpy as np

        from apps.ml.feature_extractor import (
            extract_all_features,
        )

        image = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        image[:] = [128, 128, 128]

        features = extract_all_features(image)

        expected_keys = [
            "sharpness",
            "brightness",
            "contrast",
            "noise_estimate",
            "entropy",
            "saturation",
        ]

        for key in expected_keys:
            assert key in features

        assert all(
            isinstance(value, float)
            for value in features.values()
        )

    def test_model_features_match_training(self):
        import numpy as np

        from apps.ml.feature_extractor import (
            extract_model_features,
        )

        image = np.zeros(
            (100, 100, 3),
            dtype=np.uint8,
        )

        image[:] = [128, 128, 128]

        features = extract_model_features(image)

        expected_keys = {
            "brightness",
            "contrast",
            "sharpness",
            "saturation",
            "edge_density",
        }

        assert set(features.keys()) == expected_keys

        assert all(
            isinstance(value, float)
            for value in features.values()
        )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

class TestHistory:

    def test_list_returns_200(
        self,
        client: TestClient,
    ):
        response = client.get("/api/v1/analyses")

        assert response.status_code == 200

    def test_list_returns_list(
        self,
        client: TestClient,
    ):
        response = client.get("/api/v1/analyses")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_history_contains_phase3_fields(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        upload_response = upload_sample_image(
            client,
            sample_jpeg,
        )

        assert upload_response.status_code == 201

        response = client.get("/api/v1/analyses")

        assert response.status_code == 200

        analyses = response.json()

        assert len(analyses) > 0

        item = analyses[0]

        assert "analytics_readiness_score" in item
        assert "analytics_readiness_status" in item
        assert "analytics_readiness_details" in item

    def test_filter_by_label(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        upload_response = upload_sample_image(
            client,
            sample_jpeg,
        )

        assert upload_response.status_code == 201

        response = client.get(
            "/api/v1/analyses?quality_label=Excellent"
        )

        assert response.status_code == 200

        for item in response.json():
            assert item["quality_label"] == "Excellent"

    def test_pagination(
        self,
        client: TestClient,
    ):
        response = client.get(
            "/api/v1/analyses?limit=1&offset=0"
        )

        assert response.status_code == 200

        assert len(response.json()) <= 1


# ---------------------------------------------------------------------------
# Single Analysis
# ---------------------------------------------------------------------------

class TestSingleAnalysis:

    def test_get_existing(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        upload_response = upload_sample_image(
            client,
            sample_jpeg,
        )

        assert upload_response.status_code == 201

        analysis_id = upload_response.json()["analysis_id"]

        response = client.get(
            f"/api/v1/analyses/{analysis_id}"
        )

        assert response.status_code == 200

        data = response.json()

        assert data["analysis_id"] == analysis_id
        assert "image_url" in data
        assert "recommendation" in data

        # Phase 3 fields must also persist
        assert "analytics_readiness_score" in data
        assert "analytics_readiness_status" in data
        assert "analytics_readiness_details" in data

    def test_missing_returns_404(
        self,
        client: TestClient,
    ):
        response = client.get(
            "/api/v1/analyses/nonexistent"
        )

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Static Files
# ---------------------------------------------------------------------------

class TestStaticFiles:

    def test_uploaded_image_accessible(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        upload_response = upload_sample_image(
            client,
            sample_jpeg,
        )

        assert upload_response.status_code == 201

        image_url = upload_response.json()["image_url"]

        response = client.get(image_url)

        assert response.status_code == 200