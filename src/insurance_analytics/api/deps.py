"""FastAPI dependency providers: settings, model registry, pricing service, auth."""

from __future__ import annotations

from functools import lru_cache

from fastapi import Header, HTTPException, status

from insurance_analytics.config import Settings, get_settings
from insurance_analytics.ml.predict import PricingService
from insurance_analytics.ml.registry import ModelRegistry


@lru_cache
def get_registry() -> ModelRegistry:
    return ModelRegistry(get_settings().models_dir)


@lru_cache
def get_pricing_service() -> PricingService:
    settings = get_settings()
    return PricingService(get_registry(), settings.pricing, settings.current_year)


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """No-op when API_KEY is unset (local/dev); enforced once an operator configures one."""
    settings: Settings = get_settings()
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key")
