from __future__ import annotations

from fastapi import APIRouter, Depends

from insurance_analytics.ai.insight_generator import InsightGenerator, InsightReport
from insurance_analytics.api.deps import require_api_key
from insurance_analytics.api.schemas import InsightRequest
from insurance_analytics.config import get_settings

router = APIRouter(prefix="/insights", tags=["insights"], dependencies=[Depends(require_api_key)])


@router.post("/generate", response_model=InsightReport)
def generate_insight(request: InsightRequest) -> InsightReport:
    """Turn a pre-computed analytics summary (see /analysis/eda) into a narrative report.

    Falls back to a deterministic template if no ANTHROPIC_API_KEY is configured -- this
    endpoint never fails purely because AI is unavailable.
    """
    settings = get_settings()
    generator = InsightGenerator(api_key=settings.anthropic_api_key, model=settings.ai_model)
    return generator.generate(request.stats)
