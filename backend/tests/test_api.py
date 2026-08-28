"""Tests for VisionGuard backend API endpoints."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.main import app


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


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_body(self, client: TestClient):
        response = client.get("/health")
        body = response.json()

        assert body["status"] == "healthy"
        assert isinstance(body["model_loaded"], bool)
        assert "version" in body


# ---------------------------------------------------------------------------
# Docs
# ---------------------------------------------------------------------------

class TestDocs:
    def test_swagger_available(self, client: TestClient):
        response = client.get("/docs")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

class TestAnalyze:
    def test_valid_image_returns_201(self, client: TestClient, sample_jpeg):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

    def test_valid_image_response_structure(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

        data = response.json()

        required_keys = [
            "analysis_id",
            "filename",
            "image_url",
            "quality_score",
            "quality_label",
            "analysis_confidence",
            "issues",
            "statistics",
            "summary",
            "recommendation",
            "processing_time_ms",
            "model_version",
            "explanations",
        ]

        for key in required_keys:
            assert key in data

    def test_quality_label_is_valid(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

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
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

        data = response.json()

        assert 0 <= data["quality_score"] <= 100
        assert 0 <= data["analysis_confidence"] <= 100

    def test_statistics_has_all_keys(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

        statistics = response.json()["statistics"]

        expected_keys = [
            "sharpness",
            "brightness",
            "contrast",
            "noise_estimate",
            "entropy",
            "saturation",
            "underexposure_pct",
            "overexposure_pct",
            "edge_density",
            "dynamic_range",
            "colorfulness",
            "texture_complexity",
        ]

        for key in expected_keys:
            assert key in statistics

    def test_explanations_is_list(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

        data = response.json()

        assert isinstance(data["explanations"], list)

    def test_issue_schema_is_normalized(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        """
        Phase 2 contract:
        Every detected issue must contain the normalized
        smart-city quality detection fields.
        """
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

        data = response.json()

        for issue in data["issues"]:
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
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

        data = response.json()

        valid_severities = {
            "low",
            "moderate",
            "high",
            "critical",
        }

        for issue in data["issues"]:
            assert issue["severity"] in valid_severities

    def test_issue_values_are_numeric(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert response.status_code == 201

        data = response.json()

        for issue in data["issues"]:
            assert isinstance(issue["value"], (int, float))
            assert isinstance(issue["threshold"], (int, float))
            assert isinstance(issue["confidence"], (int, float))

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

        from apps.ml.feature_extractor import extract_all_features

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
            "underexposure_pct",
            "overexposure_pct",
            "edge_density",
            "dynamic_range",
            "colorfulness",
            "texture_complexity",
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
    def test_list_returns_200(self, client: TestClient):
        response = client.get("/api/v1/analyses")

        assert response.status_code == 200

    def test_list_returns_list(self, client: TestClient):
        response = client.get("/api/v1/analyses")

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_filter_by_label(
        self,
        client: TestClient,
        sample_jpeg,
    ):
        upload_response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert upload_response.status_code == 201

        response = client.get(
            "/api/v1/analyses?quality_label=Excellent"
        )

        assert response.status_code == 200

        for item in response.json():
            assert item["quality_label"] == "Excellent"

    def test_pagination(self, client: TestClient):
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
        upload_response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
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
        upload_response = client.post(
            "/api/v1/analyze",
            files={
                "file": (
                    "test.jpg",
                    sample_jpeg,
                    "image/jpeg",
                )
            },
        )

        assert upload_response.status_code == 201

        image_url = upload_response.json()["image_url"]

        response = client.get(image_url)

        assert response.status_code == 200