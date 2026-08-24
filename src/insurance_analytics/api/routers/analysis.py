from __future__ import annotations

from fastapi import APIRouter, HTTPException

from insurance_analytics.analysis.eda import compute_eda_summary
from insurance_analytics.config import get_settings
from insurance_analytics.data.ingest import DataIngestionError, load_policies

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get("/eda")
def get_eda_summary() -> dict:
    """Compute the EDA summary against the configured cleaned dataset (or sample data if
    the full dataset isn't present locally), so the endpoint is demoable without the
    529MB production file."""
    settings = get_settings()
    path = settings.cleaned_data_path if settings.cleaned_data_path.exists() else settings.sample_data_path
    try:
        df = load_policies(path, sep="|" if path == settings.cleaned_data_path else ",", validate=False)
    except DataIngestionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return compute_eda_summary(df)
