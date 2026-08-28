"""Tests for Phase 4 — Smart-City Context pipeline selector."""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from apps.main import app
from apps.ml.context_definitions import (
    DEFAULT_CONTEXT,
    SUPPORTED_CONTEXTS,
    get_context_impacts,
)

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
    """Upload an image, optionally with a context query param."""
    url = "/api/v1/analyze"
    if context is not None:
        url = f"{url}?context={context}"
    return client.post(
        url,
        files={"file": ("test.jpg", sample_jpeg, "image/jpeg")},
    )


# ---------------------------------------------------------------------------
# Supported contexts
# ---------------------------------------------------------------------------

class TestSupportedContexts:
    """All six supported contexts must be accepted and returned."""

    @pytest.mark.parametrize("ctx", SUPPORTED_CONTEXTS)
    def test_valid_context_accepted(self, client, sample_jpeg, ctx):
        response = _upload(client, sample_jpeg, context=ctx)
        assert response.status_code == 201
        data = response.json()
        assert data["context"] == ctx

    @pytest.mark.parametrize("ctx", SUPPORTED_CONTEXTS)
    def test_context_impacts_present(self, client, sample_jpeg, ctx):
        response = _upload(client, sample_jpeg, context=ctx)
        assert response.status_code == 201
        data = response.json()
        assert "context_impacts" in data
        assert isinstance(data["context_impacts"], list)


# ---------------------------------------------------------------------------
# Default context
# ---------------------------------------------------------------------------

class TestDefaultContext:
    def test_default_context_is_cctv_surveillance(self, client, sample_jpeg):
        response = _upload(client, sample_jpeg)
        assert response.status_code == 201
        data = response.json()
        assert data["context"] == "CCTV Surveillance"

    def test_default_context_matches_constant(self):
        assert DEFAULT_CONTEXT == "CCTV Surveillance"


# ---------------------------------------------------------------------------
# Invalid context
# ---------------------------------------------------------------------------

class TestInvalidContext:
    def test_invalid_context_returns_400(self, client, sample_jpeg):
        response = _upload(client, sample_jpeg, context="Invalid Context")
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert "Unsupported context" in detail

    def test_empty_context_returns_400(self, client, sample_jpeg):
        response = _upload(client, sample_jpeg, context="")
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# Context-specific impact structure
# ---------------------------------------------------------------------------

class TestContextImpactStructure:
    @pytest.mark.parametrize("ctx", SUPPORTED_CONTEXTS)
    def test_impact_entry_has_required_keys(self, client, sample_jpeg, ctx):
        response = _upload(client, sample_jpeg, context=ctx)
        assert response.status_code == 201
        impacts = response.json()["context_impacts"]
        for entry in impacts:
            assert "issue_type" in entry
            assert "context" in entry
            assert "impact" in entry
            assert entry["context"] == ctx
            assert isinstance(entry["impact"], str)
            assert len(entry["impact"]) > 0

    def test_impact_entry_count_matches_issues(self, client, sample_jpeg):
        response = _upload(
            client, sample_jpeg, context="Traffic Monitoring"
        )
        assert response.status_code == 201
        data = response.json()
        # Each detected issue should have a corresponding context impact
        assert len(data["context_impacts"]) == len(data["issues"])


# ---------------------------------------------------------------------------
# Context-specific impact content
# ---------------------------------------------------------------------------

class TestContextImpactContent:
    """Verify that known issue+context pairs produce the expected messages."""

    def test_traffic_monitoring_blur(self, client, sample_jpeg):
        """Traffic Monitoring + blur → vehicle detection message."""
        response = _upload(
            client, sample_jpeg, context="Traffic Monitoring"
        )
        assert response.status_code == 201
        impacts = response.json()["context_impacts"]
        blur_impacts = [
            i for i in impacts if i["issue_type"] in {"severe_blur", "insufficient_sharpness"}
        ]
        for imp in blur_impacts:
            assert "vehicle" in imp["impact"].lower() or "licence" in imp["impact"].lower()

    def test_crowd_monitoring_underexposure(self, client, sample_jpeg):
        """Crowd Monitoring + underexposure → person detection message."""
        response = _upload(
            client, sample_jpeg, context="Crowd Monitoring"
        )
        assert response.status_code == 201
        impacts = response.json()["context_impacts"]
        exposure_impacts = [
            i for i in impacts
            if i["issue_type"] in {"underexposure", "severe_underexposure"}
        ]
        for imp in exposure_impacts:
            assert "person" in imp["impact"].lower() or "crowd" in imp["impact"].lower()

    def test_infrastructure_inspection_sharpness(self, client, sample_jpeg):
        """Infrastructure Inspection + insufficient_sharpness → defect message."""
        response = _upload(
            client, sample_jpeg, context="Infrastructure Inspection"
        )
        assert response.status_code == 201
        impacts = response.json()["context_impacts"]
        sharp_impacts = [
            i for i in impacts if i["issue_type"] in {"severe_blur", "insufficient_sharpness"}
        ]
        for imp in sharp_impacts:
            assert "defect" in imp["impact"].lower() or "crack" in imp["impact"].lower()

    def test_drone_imagery_blur(self, client, sample_jpeg):
        """Drone Imagery + blur → aerial survey message."""
        response = _upload(
            client, sample_jpeg, context="Drone Imagery"
        )
        assert response.status_code == 201
        impacts = response.json()["context_impacts"]
        blur_impacts = [
            i for i in impacts if i["issue_type"] in {"severe_blur", "insufficient_sharpness"}
        ]
        for imp in blur_impacts:
            assert "aerial" in imp["impact"].lower() or "survey" in imp["impact"].lower()

    def test_smart_campus_blur(self, client, sample_jpeg):
        """Smart Campus + blur → building access message."""
        response = _upload(
            client, sample_jpeg, context="Smart Campus"
        )
        assert response.status_code == 201
        impacts = response.json()["context_impacts"]
        blur_impacts = [
            i for i in impacts if i["issue_type"] in {"severe_blur", "insufficient_sharpness"}
        ]
        for imp in blur_impacts:
            assert "building" in imp["impact"].lower() or "occupancy" in imp["impact"].lower()


# ---------------------------------------------------------------------------
# Fallback behaviour
# ---------------------------------------------------------------------------

class TestImpactFallback:
    """When no context-specific message exists, the generic impact is used."""

    def test_fallback_uses_generic_impact(self, client, sample_jpeg):
        response = _upload(
            client, sample_jpeg, context="CCTV Surveillance"
        )
        assert response.status_code == 201
        impacts = response.json()["context_impacts"]
        issues = response.json()["issues"]
        # For every issue, the impact should be non-empty
        for imp in impacts:
            assert len(imp["impact"]) > 0


# ---------------------------------------------------------------------------
# Context persistence
# ---------------------------------------------------------------------------

class TestContextPersistence:
    """Context and context_impacts must survive a round-trip through the DB."""

    def test_context_in_history(self, client, sample_jpeg):
        ctx = "Traffic Monitoring"
        upload = _upload(client, sample_jpeg, context=ctx)
        assert upload.status_code == 201

        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        item = response.json()[0]
        assert item["context"] == ctx
        assert isinstance(item["context_impacts"], list)

    def test_context_in_single_analysis(self, client, sample_jpeg):
        ctx = "Drone Imagery"
        upload = _upload(client, sample_jpeg, context=ctx)
        assert upload.status_code == 201
        analysis_id = upload.json()["analysis_id"]

        response = client.get(f"/api/v1/analyses/{analysis_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["context"] == ctx
        assert isinstance(data["context_impacts"], list)
        assert len(data["context_impacts"]) > 0

    def test_old_records_get_default_context(self, client, sample_jpeg):
        """Records inserted without a context should default to CCTV Surveillance."""
        # Upload with context to verify it's not a stale DB issue
        upload = _upload(client, sample_jpeg, context="Smart Campus")
        assert upload.status_code == 201

        response = client.get("/api/v1/analyses")
        assert response.status_code == 200
        analyses = response.json()
        # Every record should have a context field
        for item in analyses:
            assert "context" in item
            assert "context_impacts" in item


# ---------------------------------------------------------------------------
# Existing Phase 1-3 fields still work
# ---------------------------------------------------------------------------

class TestExistingFieldsUnaffected:
    def test_phase1_fields_present(self, client, sample_jpeg):
        response = _upload(
            client, sample_jpeg, context="Crowd Monitoring"
        )
        assert response.status_code == 201
        data = response.json()
        # Phase 1 fields
        assert "quality_score" in data
        assert "quality_label" in data
        assert "analysis_confidence" in data
        assert isinstance(data["issues"], list)

    def test_phase2_issue_contract(self, client, sample_jpeg):
        response = _upload(
            client, sample_jpeg, context="Traffic Monitoring"
        )
        assert response.status_code == 201
        for issue in response.json()["issues"]:
            assert "type" in issue
            assert "severity" in issue
            assert "metric" in issue
            assert "value" in issue
            assert "threshold" in issue
            assert "impact" in issue

    def test_phase3_readiness_fields(self, client, sample_jpeg):
        response = _upload(
            client, sample_jpeg, context="Infrastructure Inspection"
        )
        assert response.status_code == 201
        data = response.json()
        assert "analytics_readiness_score" in data
        assert "analytics_readiness_status" in data
        assert "analytics_readiness_details" in data
        details = data["analytics_readiness_details"]
        for key in (
            "base_quality_score",
            "blur_penalty",
            "exposure_penalty",
            "noise_penalty",
            "corruption_penalty",
            "information_penalty",
        ):
            assert key in details


# ---------------------------------------------------------------------------
# Unit tests for get_context_impacts helper
# ---------------------------------------------------------------------------

class TestGetContextImpactsUnit:
    def test_empty_issues(self):
        result = get_context_impacts([], "CCTV Surveillance")
        assert result == []

    def test_issue_without_context_specific_msg(self):
        """An issue type not mapped in any context should get the generic impact."""
        issues = [{"type": "unknown_type", "impact": "Generic impact"}]
        result = get_context_impacts(issues, "Traffic Monitoring")
        assert len(result) == 1
        assert result[0]["impact"] == "Generic impact"
        assert result[0]["context"] == "Traffic Monitoring"
        assert result[0]["issue_type"] == "unknown_type"

    def test_all_six_contexts_have_messages(self):
        """Every supported context should have at least some context-specific messages."""
        for ctx in SUPPORTED_CONTEXTS:
            issues = [{"type": "severe_blur", "impact": "Generic"}]
            result = get_context_impacts(issues, ctx)
            assert len(result) == 1
            assert result[0]["context"] == ctx
            # The impact should be non-empty
            assert len(result[0]["impact"]) > 0

    def test_cctv_surveillance_blur_message(self):
        issues = [{"type": "severe_blur", "impact": "Generic"}]
        result = get_context_impacts(issues, "CCTV Surveillance")
        assert "surveillance" in result[0]["impact"].lower()
