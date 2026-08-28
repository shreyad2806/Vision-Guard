"""Health and model-info endpoints."""

from fastapi import APIRouter

from apps.core.config import settings
from apps.ml.model_loader import (
    is_model_loaded,
    get_model_version,
    get_feature_names,
    get_calibrator,
    get_metadata,
)
from apps.schemas.analysis import HealthResponse, ModelInfoResponse
from apps.ml.context_definitions import SUPPORTED_CONTEXTS
from apps.services.image_service import ALLOWED_EXTENSIONS

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status and whether a trained model is loaded.",
)
def health_check() -> HealthResponse:
    loaded = is_model_loaded()
    return HealthResponse(
        status="healthy",
        model_loaded=loaded,
        version=get_model_version(),
    )


@router.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
    summary="Model and pipeline information",
    description="Returns detailed information about the loaded ML model and supported pipeline configuration.",
)
def model_info() -> ModelInfoResponse:
    loaded = is_model_loaded()
    return ModelInfoResponse(
        model_loaded=loaded,
        model_version=get_model_version(),
        feature_names=get_feature_names(),
        calibrator_loaded=get_calibrator() is not None,
        metadata_loaded=get_metadata() is not None,
        supported_contexts=list(SUPPORTED_CONTEXTS),
        supported_extensions=sorted(ALLOWED_EXTENSIONS),
        max_upload_size_mb=settings.max_upload_size_mb,
    )
