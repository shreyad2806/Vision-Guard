"""Pydantic schemas for API request and response bodies."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QualityLabel(str, Enum):
    Excellent = "Excellent"
    Good = "Good"
    Fair = "Fair"
    Poor = "Poor"
    Critical = "Critical"


class Severity(str, Enum):
    low = "low"
    moderate = "moderate"
    high = "high"
    critical = "critical"


class Issue(BaseModel):
    type: str = Field(..., description="Detected issue type, e.g. 'low_brightness'")
    severity: Severity = Field(..., description="low, moderate, high, or critical")
    metric: str = Field(..., description="The feature name or metric key")
    value: float = Field(..., description="The observed value of the metric")
    threshold: float = Field(..., description="The warning/critical threshold value")
    impact: str = Field(..., description="The impact description")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    description: str | None = Field(default=None, description="Detailed description of the issue")
    explanation: str | None = Field(default=None, description="Human-readable explanation")


class QualityAssessment(BaseModel):
    label: str = Field(..., description="Quality label: Excellent, Good, Fair, Poor, Critical")
    status: str = Field(..., description="Lowercase status string")


class ImageStatistics(BaseModel):
    # Core metrics (displayed in UI)
    sharpness: float = Field(..., description="Variance of Laplacian")
    brightness: float = Field(..., description="Mean grayscale intensity")
    contrast: float = Field(..., description="Grayscale standard deviation")
    noise_estimate: float = Field(..., description="Estimated noise level")
    entropy: float = Field(..., description="Grayscale Shannon entropy")
    saturation: float = Field(..., description="Mean HSV saturation")
    # Extended ML features
    underexposure_pct: float = Field(default=0.0, description="Percentage of very dark pixels")
    overexposure_pct: float = Field(default=0.0, description="Percentage of very bright pixels")
    edge_density: float = Field(default=0.0, description="Canny edge pixel percentage")
    dynamic_range: float = Field(default=0.0, description="Intensity range (p95 - p5)")
    colorfulness: float = Field(default=0.0, description="Color diversity metric")
    texture_complexity: float = Field(default=0.0, description="High-frequency energy ratio")


class AnalysisResponse(BaseModel):
    analysis_id: str
    filename: str
    image_url: str = Field(default="", description="Relative URL to access the uploaded image")
    quality_score: int = Field(..., ge=0, le=100)
    quality_label: str = Field(
        ...,
        description="Quality label: Excellent, Good, Fair, Poor, Critical",
    )
    quality_assessment: QualityAssessment = Field(
        default_factory=lambda: QualityAssessment(label="Fair", status="fair"),
        description="Structured quality assessment with label and status",
    )
    analysis_confidence: float = Field(
        ..., ge=0.0, le=100.0, description="Overall confidence in the assessment"
    )
    raw_prediction: float | None = Field(
        default=None, description="Raw DMOS prediction from the model"
    )
    issues: list[Issue]
    statistics: ImageStatistics
    explanations: list[str] = Field(
        default_factory=list, description="Per-feature explanations"
    )
    summary: str
    recommendation: str = Field(default="", description="Recommended action")
    created_at: str = Field(..., description="ISO 8601 datetime string")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(default="visionguard-iqa-v1.1", description="ML model version identifier")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = Field(default="1.0.0")
