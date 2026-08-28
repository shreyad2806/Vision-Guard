"""SQLAlchemy model for analysis results."""

import json
from datetime import datetime, timezone

from sqlalchemy import String, Float, Integer, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from apps.db.database import Base


class AnalysisRecord(Base):
    """Persisted analysis result."""

    __tablename__ = "analyses"

    analysis_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(String(512))
    image_url: Mapped[str] = mapped_column(String(512), default="")
    quality_score: Mapped[int] = mapped_column(Integer)
    quality_label: Mapped[str] = mapped_column(String(20))
    analysis_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    issues_json: Mapped[str] = mapped_column(Text, default="[]")
    statistics_json: Mapped[str] = mapped_column(Text, default="{}")
    summary: Mapped[str] = mapped_column(Text, default="")
    recommendation: Mapped[str] = mapped_column(Text, default="")
    explanations_json: Mapped[str] = mapped_column(Text, default="[]")
    quality_assessment_json: Mapped[str] = mapped_column(
        Text, default='{"label":"Fair","status":"fair"}'
    )
    analytics_readiness_score: Mapped[float] = mapped_column(Float, default=0.0)
    analytics_readiness_status: Mapped[str] = mapped_column(String(50), default="")
    analytics_readiness_details_json: Mapped[str] = mapped_column(Text, default="{}")
    context: Mapped[str] = mapped_column(String(50), default="CCTV Surveillance")
    context_impacts_json: Mapped[str] = mapped_column(Text, default="[]")
    issue_explanations_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    processing_time_ms: Mapped[int] = mapped_column(Integer, default=0)
    model_version: Mapped[str] = mapped_column(String(100), default="visionguard-iqa-v1.1")

    # ----- convenience helpers -----

    @property
    def issues(self) -> list[dict]:  # type: ignore[override]
        return json.loads(self.issues_json)

    @issues.setter
    def issues(self, value: list[dict]) -> None:  # type: ignore[override]
        self.issues_json = json.dumps(value)

    @property
    def statistics(self) -> dict:  # type: ignore[override]
        return json.loads(self.statistics_json)

    @statistics.setter
    def statistics(self, value: dict) -> None:  # type: ignore[override]
        self.statistics_json = json.dumps(value)

    @property
    def explanations(self) -> list[str]:
        return json.loads(self.explanations_json)

    @explanations.setter
    def explanations(self, value: list[str]) -> None:
        self.explanations_json = json.dumps(value)

    @property
    def quality_assessment(self) -> dict:
        return json.loads(self.quality_assessment_json)

    @quality_assessment.setter
    def quality_assessment(self, value: dict) -> None:
        self.quality_assessment_json = json.dumps(value)

    @property
    def analytics_readiness_details(self) -> dict:
        return json.loads(self.analytics_readiness_details_json)

    @analytics_readiness_details.setter
    def analytics_readiness_details(self, value: dict) -> None:
        self.analytics_readiness_details_json = json.dumps(value)

    @property
    def context_impacts(self) -> list[dict]:
        return json.loads(self.context_impacts_json)

    @context_impacts.setter
    def context_impacts(self, value: list[dict]) -> None:
        self.context_impacts_json = json.dumps(value)

    @property
    def issue_explanations(self) -> list[dict]:
        return json.loads(self.issue_explanations_json)

    @issue_explanations.setter
    def issue_explanations(self, value: list[dict]) -> None:
        self.issue_explanations_json = json.dumps(value)

    def to_dict(self) -> dict:
        """Serialise to the shape expected by the API response."""
        return {
            "analysis_id": self.analysis_id,
            "filename": self.filename,
            "image_url": self.image_url,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "quality_assessment": self.quality_assessment,
            "analysis_confidence": self.analysis_confidence,
            "issues": self.issues,
            "statistics": self.statistics,
            "explanations": self.explanations,
            "summary": self.summary,
            "recommendation": self.recommendation,
            "created_at": self.created_at.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "model_version": self.model_version,
            "analytics_readiness_score": self.analytics_readiness_score,
            "analytics_readiness_status": self.analytics_readiness_status,
            "analytics_readiness_details": self.analytics_readiness_details,
            "context": self.context,
            "context_impacts": self.context_impacts,
            "issue_explanations": self.issue_explanations,
        }
