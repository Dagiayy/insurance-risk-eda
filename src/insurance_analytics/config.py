"""Centralized, environment-driven configuration.

All paths, pricing constants, and third-party credentials live here instead of being
hard-coded across scripts (the original project scattered ``2015 - RegistrationYear``,
premium-loading percentages, and relative file paths through five different files).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PricingConfig(BaseSettings):
    """Premium-formula constants, previously hard-coded magic numbers in modeling.py."""

    expense_loading_pct: float = Field(0.10, description="Fraction of severity added as expense loading")
    profit_margin_pct: float = Field(0.15, description="Fraction of expected loss added as profit margin")

    model_config = SettingsConfigDict(env_prefix="PRICING_", extra="ignore")

    def calculate_premium(self, claim_probability: float, claim_severity: float) -> float:
        """Risk-based premium = expected loss + expense loading + profit margin.

        expected_loss = probability * severity
        """
        expected_loss = claim_probability * claim_severity
        return expected_loss + self.expense_loading_pct * claim_severity + self.profit_margin_pct * expected_loss


class Settings(BaseSettings):
    """Application-wide settings, overridable via environment variables or a ``.env`` file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- paths -------------------------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_path: Path = PROJECT_ROOT / "data" / "MachineLearningRating_v3.txt"
    cleaned_data_path: Path = PROJECT_ROOT / "data" / "MachineLearningRating_v3_cleaned.txt"
    sample_data_path: Path = PROJECT_ROOT / "data" / "sample" / "sample_policies.csv"
    docs_dir: Path = PROJECT_ROOT / "docs"
    results_dir: Path = PROJECT_ROOT / "results"
    plots_dir: Path = PROJECT_ROOT / "notebooks" / "plots"
    models_dir: Path = PROJECT_ROOT / "models"

    # --- data source formatting ----------------------------------------------------
    raw_data_sep: str = "|"

    # --- ML -------------------------------------------------------------------------
    random_state: int = 42
    test_size: float = 0.2
    current_year: int = 2015  # policy book snapshot year; overridable, not hard-coded per-file

    # --- API --------------------------------------------------------------------
    api_title: str = "ACIS Insurance Risk & Pricing API"
    api_key: str | None = Field(default=None, description="Optional shared-secret for write/predict endpoints")

    # --- AI (optional) ------------------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    ai_model: str = "claude-sonnet-5"

    # --- logging --------------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    pricing: PricingConfig = Field(default_factory=PricingConfig)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton (safe to call repeatedly across modules)."""
    return Settings()
