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


# ---- Health ----

class TestHealth:
    def test_health_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_body(self, client: TestClient):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "healthy"
        assert isinstance(body["model_loaded"], bool)
        assert "version" in body


# ---- Docs ----

class TestDocs:
    def test_swagger_available(self, client: TestClient):
        r = client.get("/docs")
        assert r.status_code == 200


# ---- Analyze ----

class TestAnalyze:
    def test_valid_image_returns_201(self, client: TestClient, sample_jpeg):
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        assert r.status_code == 201

    def test_valid_image_response_structure(self, client: TestClient, sample_jpeg):
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        data = r.json()
        assert "analysis_id" in data
        assert "filename" in data
        assert "image_url" in data
        assert "quality_score" in data
        assert "quality_label" in data
        assert "analysis_confidence" in data
        assert "issues" in data
        assert "statistics" in data
        assert "summary" in data
        assert "recommendation" in data
        assert "processing_time_ms" in data
        assert "model_version" in data

    def test_quality_label_is_valid(self, client: TestClient, sample_jpeg):
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        data = r.json()
        assert data["quality_label"] in ("ACCEPTABLE", "DEGRADED", "DEFECTIVE")

    def test_score_in_range(self, client: TestClient, sample_jpeg):
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        data = r.json()
        assert 0 <= data["quality_score"] <= 100
        assert 0 <= data["analysis_confidence"] <= 100

    def test_statistics_has_all_keys(self, client: TestClient, sample_jpeg):
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        stats = r.json()["statistics"]
        for key in ["sharpness", "brightness", "contrast", "noise_estimate", "entropy", "saturation"]:
            assert key in stats

    def test_invalid_extension_returns_400(self, client: TestClient):
        r = client.post("/api/v1/analyze", files={"file": ("test.txt", b"hello", "text/plain")})
        assert r.status_code == 400

    def test_empty_file_returns_400(self, client: TestClient):
        r = client.post("/api/v1/analyze", files={"file": ("empty.jpg", b"", "image/jpeg")})
        assert r.status_code == 400

    def test_corrupted_image_returns_400(self, client: TestClient):
        r = client.post("/api/v1/analyze", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})
        assert r.status_code == 400


# ---- Feature Extraction ----

class TestFeatureExtraction:
    def test_extracts_all_features(self):
        import cv2
        import numpy as np

        from apps.ml.feature_extractor import extract_features

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = [128, 128, 128]
        features = extract_features(img)
        expected_keys = {"sharpness", "brightness", "contrast", "noise_estimate", "entropy", "saturation"}
        assert set(features.keys()) == expected_keys
        assert all(isinstance(v, float) for v in features.values())


# ---- History ----

class TestHistory:
    def test_list_returns_200(self, client: TestClient):
        r = client.get("/api/v1/analyses")
        assert r.status_code == 200

    def test_list_returns_list(self, client: TestClient):
        r = client.get("/api/v1/analyses")
        assert isinstance(r.json(), list)

    def test_filter_by_label(self, client: TestClient, sample_jpeg):
        # Upload an image first
        client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        r = client.get("/api/v1/analyses?quality_label=ACCEPTABLE")
        assert r.status_code == 200
        for item in r.json():
            assert item["quality_label"] == "ACCEPTABLE"

    def test_pagination(self, client: TestClient):
        r = client.get("/api/v1/analyses?limit=1&offset=0")
        assert r.status_code == 200
        assert len(r.json()) <= 1


# ---- Single Analysis ----

class TestSingleAnalysis:
    def test_get_existing(self, client: TestClient, sample_jpeg):
        # Upload first
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        analysis_id = r.json()["analysis_id"]

        r = client.get(f"/api/v1/analyses/{analysis_id}")
        assert r.status_code == 200
        assert r.json()["analysis_id"] == analysis_id
        assert "image_url" in r.json()
        assert "recommendation" in r.json()

    def test_missing_returns_404(self, client: TestClient):
        r = client.get("/api/v1/analyses/nonexistent")
        assert r.status_code == 404


# ---- Static Files ----

class TestStaticFiles:
    def test_uploaded_image_accessible(self, client: TestClient, sample_jpeg):
        r = client.post("/api/v1/analyze", files={"file": ("test.jpg", sample_jpeg, "image/jpeg")})
        image_url = r.json()["image_url"]
        r = client.get(image_url)
        assert r.status_code == 200
