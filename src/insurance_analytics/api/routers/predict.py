from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from insurance_analytics.api.deps import get_pricing_service, require_api_key
from insurance_analytics.api.schemas import PolicyRequest, PricingResponse
from insurance_analytics.logging_config import get_logger
from insurance_analytics.ml.predict import PricingService

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"], dependencies=[Depends(require_api_key)])


@router.post("/premium", response_model=PricingResponse)
def predict_premium(policy: PolicyRequest, service: PricingService = Depends(get_pricing_service)) -> PricingResponse:
    """Score a single policy: predicted claim probability, expected severity, and risk-based premium."""
    try:
        df = pd.DataFrame([policy.model_dump()])
        prediction = service.predict(df)[0]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not available: {exc}") from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    return PricingResponse(
        claim_probability=prediction.claim_probability,
        claim_severity=prediction.claim_severity,
        predicted_premium=prediction.predicted_premium,
        model_versions=service.model_versions,
    )
