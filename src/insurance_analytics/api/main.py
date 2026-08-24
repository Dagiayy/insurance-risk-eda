"""FastAPI application entrypoint.

Run with: ``uvicorn insurance_analytics.api.main:app --reload``
"""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from insurance_analytics import __version__
from insurance_analytics.api.routers import analysis, health, insights, predict
from insurance_analytics.config import get_settings
from insurance_analytics.logging_config import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.api_title,
    version=__version__,
    description=(
        "Risk-based pricing and analytics API for ACIS motor insurance policies. "
        "Serves claim-probability, claim-severity, and premium predictions from registered "
        "models, plus EDA and AI-narrated insight endpoints."
    ),
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(insights.router)
app.include_router(analysis.router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = (time.monotonic() - start) * 1000
    logger.info("%s %s -> %d (%.1fms)", request.method, request.url.path, response.status_code, duration_ms)
    return response


@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("Model not available: %s", exc)
    return JSONResponse(status_code=503, content={"detail": f"Model not available: {exc}"})
