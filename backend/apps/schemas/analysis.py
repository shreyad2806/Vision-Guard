"""Pydantic schemas for VisionGuard analysis API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Issue severity
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Normalized severity levels returned by the quality detection pipeline."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Detected issue
# ---------------------------------------------------------------------------

class Issue(BaseModel):
    """Structured, explainable image-quality issue."""

    type: str = Field(
        ...,
        description="Detected issue category.",
    )

    severity: Severity = Field(
        ...,
        description=(
            "Normalized issue severity: "
            "low, moderate, high, or critical."
        ),
    )

    metric: str = Field(
        ...,
        description="Feature or measurement used to detect the issue.",
    )

    value: float = Field(
        ...,
        description="Measured metric value.",
    )

    threshold: float = Field(
        ...,
        description="Detection threshold for the metric.",
    )

    impact: str = Field(
        ...,
        description=(
            "Expected impact on downstream "
            "smart-city analytics."
        ),
    )

    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Confidence in the issue detection.",
    )

    description: str | None = Field(
        default=None,
        description="Optional human-readable explanation.",
    )


# ---------------------------------------------------------------------------
# Quality assessment
# ---------------------------------------------------------------------------

class QualityAssessment(BaseModel):
    """Human-readable quality classification."""

    label: str
    status: str

class AnalyticsReadinessDetails(BaseModel):
    """Explainable breakdown of analytics readiness scoring."""

    base_quality_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    blur_penalty: float = Field(
        ...,
        ge=0,
    )

    exposure_penalty: float = Field(
        ...,
        ge=0,
    )

    noise_penalty: float = Field(
        ...,
        ge=0,
    )

    corruption_penalty: float = Field(
        ...,
        ge=0,
    )

    information_penalty: float = Field(
        ...,
        ge=0,
    )


# ---------------------------------------------------------------------------
# Context impact
# ---------------------------------------------------------------------------

class ContextImpact(BaseModel):
    """Context-specific impact explanation for a detected issue."""

    issue_type: str = Field(
        ...,
        description="Detected issue category this impact relates to.",
    )
    context: str = Field(
        ...,
        description="Pipeline context this impact was generated for.",
    )
    impact: str = Field(
        ...,
        description="Context-specific impact description.",
    )


# ---------------------------------------------------------------------------
# Analysis response
# ---------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Complete response returned by VisionGuard image analysis."""

    analysis_id: str

    filename: str
    image_path: str | None = None
    image_url: str | None = None

    quality_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Overall visual quality score from 0 to 100.",
    )

    quality_label: str

    analysis_confidence: float = Field(
        ...,
        ge=0,
        le=100,
    )

    issues: list[Issue] = Field(
        default_factory=list,
    )

    statistics: dict[str, Any] = Field(
        default_factory=dict,
    )

    summary: str
    recommendation: str

    created_at: datetime

    processing_time_ms: int

    model_version: str

    explanations: list[Any] = Field(
        default_factory=list,
    )

    quality_assessment: dict[str, Any] | None = None

    raw_prediction: float | None = None
    
    analytics_readiness_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Suitability of the image for downstream smart-city computer vision analytics.",
    )

    analytics_readiness_status: str = Field(
        ...,
        description="HIGHLY READY, READY, LIMITED READINESS, NOT READY, or CRITICAL / REJECT.",
    )

    analytics_readiness_details: AnalyticsReadinessDetails

    context: str = Field(
        default="CCTV Surveillance",
        description="Pipeline context used for this analysis.",
    )

    context_impacts: list[ContextImpact] = Field(
        default_factory=list,
        description="Context-specific impact explanations for detected issues.",
    )


# ---------------------------------------------------------------------------
# Health response
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """Backend health and model availability response."""

    status: str = Field(
        ...,
        description="Backend service health status.",
    )

    model_loaded: bool = Field(
        ...,
        description=(
            "Whether the ML quality model "
            "is loaded and ready."
        ),
    )

    version: str = Field(
        ...,
        description=(
            "VisionGuard backend or model version."
        ),
    )