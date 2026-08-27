"""Analysis endpoints: upload, history, single analysis."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from apps.core.config import settings
from apps.db.database import get_db
from apps.models.analysis import AnalysisRecord
from apps.schemas.analysis import AnalysisResponse
from apps.services.image_service import (
    ALLOWED_EXTENSIONS,
    save_upload,
    validate_extension,
    validate_size,
)
from apps.services.analysis_service import run_analysis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])


def _record_to_response(record: AnalysisRecord) -> dict:
    """Convert an ORM record to the API response shape."""
    return record.to_dict()


# ---------- POST /analyze ----------


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyse an uploaded image",
    description="Accepts a multipart image upload, runs quality analysis, and returns structured results.",
    status_code=201,
)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyse"),
    db: Session = Depends(get_db),
):
    # --- Validate file extension ---
    filename = file.filename or "unknown"
    if not validate_extension(filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed extensions: {allowed}",
        )

    # --- Read and validate size ---
    data = await file.read()
    if not validate_size(data):
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum upload size of {settings.max_upload_size_mb} MB.",
        )

    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # --- Save to disk ---
    try:
        stored_name, image_path = save_upload(data, filename)
    except Exception:
        logger.exception("Failed to save upload")
        raise HTTPException(status_code=500, detail="Failed to store uploaded file.")

    # --- Run analysis ---
    try:
        result = run_analysis(db, filename, image_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Analysis failed for %s", filename)
        raise HTTPException(status_code=500, detail="Image analysis failed.")

    return result


# ---------- GET /analyses ----------

# Valid quality labels for filtering
_VALID_LABELS = {"Excellent", "Good", "Fair", "Poor", "Critical"}


@router.get(
    "/analyses",
    response_model=list[AnalysisResponse],
    summary="List past analyses",
    description="Returns previous analyses, optionally filtered by quality label.",
)
def get_analyses(
    limit: int = Query(50, ge=1, le=200, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    quality_label: str | None = Query(
        None,
        description="Filter by quality label (Excellent, Good, Fair, Poor, Critical)",
    ),
    db: Session = Depends(get_db),
):
    q = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
    if quality_label:
        q = q.filter(AnalysisRecord.quality_label == quality_label.capitalize())
    records = q.offset(offset).limit(limit).all()
    return [_record_to_response(r) for r in records]


# ---------- GET /analyses/{analysis_id} ----------


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get a single analysis",
    description="Return the full stored analysis for a given ID.",
)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)):
    record = (
        db.query(AnalysisRecord)
        .filter(AnalysisRecord.analysis_id == analysis_id)
        .first()
    )
    if record is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return _record_to_response(record)
