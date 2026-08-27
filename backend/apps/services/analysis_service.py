"""Orchestrate the full image analysis pipeline.

Pipeline:  image → feature extraction → model inference → quality decision → persist
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from sqlalchemy.orm import Session

from apps.core.config import settings
from apps.models.analysis import AnalysisRecord
from apps.ml.feature_extractor import extract_features
from apps.ml.inference import predict_quality


def run_analysis(db: Session, filename: str, image_path: str) -> dict:
    """Execute the full analysis pipeline and persist the result.

    Returns the serialised analysis dict ready for the API response.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image file.")

    # 1. Extract real image features
    features = extract_features(img)

    # 2. Run inference (real ML model or temporary fallback)
    result = predict_quality(features)

    # 3. Build image URL from stored filename
    stored_name = Path(image_path).name
    image_url = f"{settings.upload_url_base}/{stored_name}"

    # 4. Build response payload
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
    db.add(record)
    db.commit()
    db.refresh(record)

    return record.to_dict()
