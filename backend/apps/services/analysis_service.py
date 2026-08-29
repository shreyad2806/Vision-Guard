"""Orchestrate the full image analysis pipeline.

Pipeline:  image -> feature extraction -> model inference -> quality decision -> persist
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from apps.core.config import settings
from apps.models.analysis import AnalysisRecord
from apps.ml.feature_extractor import extract_model_features, extract_all_features
from apps.ml.inference import predict_quality
from apps.ml.context_definitions import DEFAULT_CONTEXT


def run_analysis(db: Session, filename: str, image_path: str, context: str = DEFAULT_CONTEXT) -> dict:
    """Execute the full analysis pipeline and persist the result.

    Returns the serialised analysis dict ready for the API response.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image file.")

    # 1. Extract the 5 model features (matches ml/train.py preprocessing)
    model_features = extract_model_features(img)

    # 2. Extract all 12 statistics for UI display
    all_features = extract_all_features(img)

    # 3. Run calibrated inference (model prediction + calibration + issue detection)
    result = predict_quality(model_features, all_features, context=context)

    # 4. Build image URL from stored filename
    stored_name = Path(image_path).name
    image_url = f"{settings.upload_url_base}/{stored_name}"

    # 5. Persist
    analysis_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc)

    record = AnalysisRecord(
        analysis_id=analysis_id,
        filename=filename,
        image_path=image_path,
        image_url=image_url,
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        analysis_confidence=result["analysis_confidence"],
        issues=result["issues"],
        statistics=result["statistics"],
        summary=result["summary"],
        recommendation=result["recommendation"],
        created_at=created_at,
        processing_time_ms=result["processing_time_ms"],
        model_version=settings.model_version,
    )

    # Store extra fields
    explanations = result.get("explanations", [])
    quality_assessment = result.get("quality_assessment", {"label": "Fair", "status": "fair"})
    record.explanations_json = json.dumps(explanations)
    record.quality_assessment_json = json.dumps(quality_assessment)

    # Store Phase 3 analytics readiness fields
    record.analytics_readiness_score = result.get("analytics_readiness_score", 0.0)
    record.analytics_readiness_status = result.get("analytics_readiness_status", "")
    record.analytics_readiness_details_json = json.dumps(result.get("analytics_readiness_details", {}))

    # Store Phase 4 context fields
    record.context = result.get("context", DEFAULT_CONTEXT)
    record.context_impacts_json = json.dumps(result.get("context_impacts", []))

    # Store Phase 6 explainability fields
    record.issue_explanations_json = json.dumps(result.get("issue_explanations", []))

    # Store downstream analytics fields
    record.downstream_analytics_json = json.dumps(result.get("downstream_analytics", []))

    db.add(record)
    db.commit()
    db.refresh(record)

    # Build response dict
    response = record.to_dict()
    response["raw_prediction"] = result.get("raw_prediction")
    response["downstream_analytics"] = result.get("downstream_analytics", [])
    return response
