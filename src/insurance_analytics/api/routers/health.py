from __future__ import annotations

from fastapi import APIRouter, Depends

from insurance_analytics import __version__
from insurance_analytics.api.deps import get_registry
from insurance_analytics.api.schemas import HealthResponse
from insurance_analytics.ml.registry import ModelRegistry
from insurance_analytics.ml.train import PROBABILITY_MODEL_NAME, SEVERITY_MODEL_NAME

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    models_loaded = registry.has_model(SEVERITY_MODEL_NAME) and registry.has_model(PROBABILITY_MODEL_NAME)
    return HealthResponse(status="ok", version=__version__, models_loaded=models_loaded)
