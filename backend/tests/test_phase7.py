"""Tests for Phase 7 — Backend Completeness (VisionGuard).

Covers:
  - GET /api/v1/model-info endpoint
  - Full pipeline end-to-end (upload → inference → persist → retrieve)
  - Oversized image rejection
  - Missing ML model graceful degradation
  - Inference / processing failure handling
  - All Phase 1–6 fields verified in a single pass
  - Database persistence round-trip
"""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.main import app


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


def _upload(client, buf=None, filename="test.jpg", context=None):
    if buf is None:
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
    url = "/api/v1/analyze"
    if context is not None:
        url = f"{url}?context={context}"
    return client.post(url, files={"file": (filename, buf, "image/jpeg")})


# ===================================================================
# 1. GET /api/v1/model-info
# ===================================================================

class TestModelInfo:
    def test_model_info_returns_200(self, client):
        resp = client.get("/api/v1/model-info")
        assert resp.status_code == 200

    def test_model_info_schema(self, client):
        data = client.get("/api/v1/model-info").json()
        assert "model_loaded" in data
        assert "model_version" in data
        assert "feature_names" in data
        assert "calibrator_loaded" in data
        assert "metadata_loaded" in data
        assert "supported_contexts" in data
        assert "supported_extensions" in data
        assert "max_upload_size_mb" in data

    def test_model_info_types(self, client):
        data = client.get("/api/v1/model-info").json()
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["model_version"], str)
        assert isinstance(data["feature_names"], list)
        assert isinstance(data["calibrator_loaded"], bool)
        assert isinstance(data["metadata_loaded"], bool)
        assert isinstance(data["supported_contexts"], list)
        assert isinstance(data["supported_extensions"], list)
        assert isinstance(data["max_upload_size_mb"], int)

    def test_model_info_supported_contexts(self, client):
        data = client.get("/api/v1/model-info").json()
        expected = {
            "CCTV Surveillance", "Traffic Monitoring", "Crowd Monitoring",
            "Drone Imagery", "Infrastructure Inspection", "Smart Campus",
        }
        assert set(data["supported_contexts"]) == expected

    def test_model_info_supported_extensions(self, client):
        data = client.get("/api/v1/model-info").json()
        assert ".jpg" in data["supported_extensions"]
        assert ".jpeg" in data["supported_extensions"]
        assert ".png" in data["supported_extensions"]
        assert ".webp" in data["supported_extensions"]

    def test_model_info_max_upload_positive(self, client):
        data = client.get("/api/v1/model-info").json()
        assert data["max_upload_size_mb"] > 0


# ===================================================================
# 2. Full pipeline end-to-end — all Phase 1–6 fields
# ===================================================================

class TestFullPipeline:
    def test_all_phase_fields_present(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()

        # Phase 1
        assert "quality_score" in data
        assert "quality_label" in data
        assert "analysis_confidence" in data
        assert 0 <= data["quality_score"] <= 100

        # Phase 2
        assert "issues" in data
        assert isinstance(data["issues"], list)

        # Phase 3
        assert "analytics_readiness_score" in data
        assert "analytics_readiness_status" in data
        assert "analytics_readiness_details" in data
        assert 0 <= data["analytics_readiness_score"] <= 100

        # Phase 4
        assert "context" in data
        assert "context_impacts" in data

        # Phase 6
        assert "issue_explanations" in data
        assert isinstance(data["issue_explanations"], list)

    def test_quality_score_range(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert 0 <= data["quality_score"] <= 100

    def test_quality_label_valid(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert data["quality_label"] in ("Excellent", "Good", "Fair", "Poor", "Critical")

    def test_readiness_status_valid(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        valid = {"HIGHLY READY", "READY", "LIMITED READINESS", "NOT READY", "CRITICAL / REJECT"}
        assert data["analytics_readiness_status"] in valid

    def test_readiness_details_penalties_non_negative(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        details = data["analytics_readiness_details"]
        for key in ("blur_penalty", "exposure_penalty", "noise_penalty",
                     "corruption_penalty", "information_penalty"):
            assert details[key] >= 0

    def test_issue_explanations_match_issues(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert len(data["issue_explanations"]) == len(data["issues"])

    def test_issue_explanations_have_structure(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        for exp in data["issue_explanations"]:
            assert "issue" in exp
            assert "evidence" in exp
            assert "why_it_matters" in exp
            assert "recommendation" in exp
            ev = exp["evidence"]
            assert "metric" in ev
            assert "value" in ev
            assert "threshold" in ev

    def test_context_impacts_match_issues(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert len(data["context_impacts"]) == len(data["issues"])

    def test_analysis_id_is_hex(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert len(data["analysis_id"]) == 32
        assert all(c in "0123456789abcdef" for c in data["analysis_id"])

    def test_model_version_is_string(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert isinstance(data["model_version"], str)
        assert len(data["model_version"]) > 0

    def test_processing_time_non_negative(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert data["processing_time_ms"] >= 0

    def test_statistics_has_required_keys(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        for key in ("sharpness", "brightness", "contrast", "noise_estimate",
                     "entropy", "saturation"):
            assert key in data["statistics"]

    def test_explanations_is_list(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert isinstance(data["explanations"], list)

    def test_summary_is_string(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert isinstance(data["summary"], str)
        assert len(data["summary"]) > 0

    def test_recommendation_is_string(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert isinstance(data["recommendation"], str)
        assert len(data["recommendation"]) > 0


# ===================================================================
# 3. Context parameter variations
# ===================================================================

class TestContextParameter:
    @pytest.mark.parametrize("ctx", [
        "CCTV Surveillance", "Traffic Monitoring", "Crowd Monitoring",
        "Drone Imagery", "Infrastructure Inspection", "Smart Campus",
    ])
    def test_valid_context_accepted(self, client, ctx):
        resp = _upload(client, context=ctx)
        assert resp.status_code == 201
        assert resp.json()["context"] == ctx

    def test_default_context(self, client, sample_jpeg):
        data = _upload(client, sample_jpeg).json()
        assert data["context"] == "CCTV Surveillance"

    def test_invalid_context_returns_400(self, client, sample_jpeg):
        resp = _upload(client, sample_jpeg, context="Invalid")
        assert resp.status_code == 400


# ===================================================================
# 4. Oversized image
# ===================================================================

class TestOversizedImage:
    def test_oversized_returns_413(self, client):
        # Create an image larger than max_upload_size_mb
        big = b"\xff" * (21 * 1024 * 1024)  # 21 MB > 20 MB limit
        resp = client.post(
            "/api/v1/analyze",
            files={"file": ("big.jpg", io.BytesIO(big), "image/jpeg")},
        )
        assert resp.status_code == 413


# ===================================================================
# 5. Missing model graceful degradation
# ===================================================================

class TestMissingModel:
    def test_upload_works_without_model(self, client, sample_jpeg):
        """The pipeline should fall back to rule-based scoring if no model is loaded."""
        with patch("apps.ml.model_loader._model", None), \
             patch("apps.ml.model_loader._loaded", True):
            resp = _upload(client, sample_jpeg)
            assert resp.status_code == 201
            data = resp.json()
            assert 0 <= data["quality_score"] <= 100
            assert data["quality_label"] in ("Excellent", "Good", "Fair", "Poor", "Critical")

    def test_model_info_shows_not_loaded(self, client):
        with patch("apps.ml.model_loader._model", None), \
             patch("apps.ml.model_loader._loaded", True):
            data = client.get("/api/v1/model-info").json()
            assert data["model_loaded"] is False


# ===================================================================
# 6. Inference / processing failure handling
# ===================================================================

class TestInferenceFailure:
    def test_cv2_imread_failure_returns_400(self, client, sample_jpeg):
        """If cv2.imread returns None (corrupted-but-valid-extension), expect 400."""
        with patch("apps.services.analysis_service.cv2.imread", return_value=None):
            resp = _upload(client, sample_jpeg)
            assert resp.status_code == 400

    def test_unexpected_exception_returns_500(self, client, sample_jpeg):
        """If run_analysis throws an unexpected error, expect 500 not a traceback."""
        with patch("apps.api.routes.analysis.run_analysis",
                    side_effect=RuntimeError("boom")):
            resp = _upload(client, sample_jpeg)
            assert resp.status_code == 500
            assert resp.json()["detail"] == "Image analysis failed."


# ===================================================================
# 7. Database persistence round-trip
# ===================================================================

class TestDatabasePersistence:
    def test_persist_and_retrieve(self, client, sample_jpeg):
        upload = _upload(client, sample_jpeg)
        assert upload.status_code == 201
        analysis_id = upload.json()["analysis_id"]

        # GET single
        single = client.get(f"/api/v1/analyses/{analysis_id}")
        assert single.status_code == 200
        data = single.json()
        assert data["analysis_id"] == analysis_id
        assert "quality_score" in data
        assert "analytics_readiness_score" in data
        assert "issue_explanations" in data
        assert "context" in data

    def test_persist_in_list(self, client, sample_jpeg):
        upload = _upload(client, sample_jpeg)
        assert upload.status_code == 201

        lst = client.get("/api/v1/analyses?limit=1")
        assert lst.status_code == 200
        items = lst.json()
        assert len(items) == 1
        assert items[0]["analysis_id"] == upload.json()["analysis_id"]

    def test_missing_analysis_returns_404(self, client):
        resp = client.get("/api/v1/analyses/nonexistent")
        assert resp.status_code == 404


# ===================================================================
# 8. Phase 2 issue contract
# ===================================================================

class TestPhase2Contract:
    def test_issue_fields(self, client, sample_jpeg):
        issues = _upload(client, sample_jpeg).json()["issues"]
        for issue in issues:
            assert "type" in issue
            assert "severity" in issue
            assert "metric" in issue
            assert "value" in issue
            assert "threshold" in issue
            assert "impact" in issue
            assert "confidence" in issue

    def test_severity_enum(self, client, sample_jpeg):
        issues = _upload(client, sample_jpeg).json()["issues"]
        valid = {"low", "moderate", "high", "critical"}
        for issue in issues:
            assert issue["severity"] in valid

    def test_numeric_values(self, client, sample_jpeg):
        issues = _upload(client, sample_jpeg).json()["issues"]
        for issue in issues:
            assert isinstance(issue["value"], (int, float))
            assert isinstance(issue["threshold"], (int, float))
            assert 0 <= issue["confidence"] <= 1


# ===================================================================
# 9. Feature extraction direct tests
# ===================================================================

class TestFeatureExtractionDirect:
    def test_extract_model_features(self):
        from apps.ml.feature_extractor import extract_model_features
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = [128, 128, 128]
        features = extract_model_features(img)
        assert set(features.keys()) == {"brightness", "contrast", "sharpness", "saturation", "edge_density"}
        assert all(isinstance(v, float) for v in features.values())

    def test_extract_all_features(self):
        from apps.ml.feature_extractor import extract_all_features
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = [128, 128, 128]
        features = extract_all_features(img)
        for key in ("sharpness", "brightness", "contrast", "noise_estimate", "entropy", "saturation"):
            assert key in features
        assert all(isinstance(v, float) for v in features.values())
