"""Health check endpoint."""

from fastapi import APIRouter

from apps.core.config import settings
from apps.ml.model_loader import is_model_loaded, get_model_version
from apps.schemas.analysis import HealthResponse

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
