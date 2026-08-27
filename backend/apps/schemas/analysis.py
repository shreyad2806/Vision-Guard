"""Pydantic schemas for API request and response bodies."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QualityLabel(str, Enum):
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    DEFECTIVE = "DEFECTIVE"


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Issue(BaseModel):
    type: str = Field(..., description="Detected issue type, e.g. 'Blur', 'Noise'")
    severity: Severity = Field(..., description="LOW, MEDIUM, or HIGH")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0–1")
    explanation: str = Field(..., description="Human-readable explanation")


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
    quality_label: QualityLabel
    analysis_confidence: float = Field(..., ge=0.0, le=100.0, description="Overall confidence in the assessment")
    issues: list[Issue]
    statistics: ImageStatistics
    summary: str
    recommendation: str = Field(default="", description="Recommended action")
    created_at: str = Field(..., description="ISO 8601 datetime string")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(default="feature-baseline-v1", description="ML model version identifier")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = Field(default="1.0.0")
