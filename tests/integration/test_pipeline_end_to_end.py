"""End-to-end: ingest -> validate -> clean -> engineer -> train -> register -> predict.

Exercises the full pipeline described in the architecture doc against the small synthetic
sample dataset, without touching the 529MB production file.
"""

from __future__ import annotations

from insurance_analytics.config import PricingConfig
from insurance_analytics.data.clean import clean_policies
from insurance_analytics.data.ingest import load_policies
from insurance_analytics.ml.predict import PricingService
from insurance_analytics.ml.registry import ModelRegistry
from insurance_analytics.ml.train import train_and_register


def test_full_pipeline_runs_end_to_end(sample_data_path, tmp_path):
    df = load_policies(sample_data_path, sep=",", validate=True)
    cleaned, quality_report = clean_policies(df)
    assert quality_report.output_rows > 0

    registry = ModelRegistry(tmp_path / "models")
    result = train_and_register(cleaned, reference_year=2015, registry=registry, cv_folds=2)

    assert result.best_probability_model, "probability model must train on the full population"
    assert "random_forest" in result.probability_metrics or "xgboost" in result.probability_metrics

    service = PricingService(registry, PricingConfig(), reference_year=2015)
    prediction = service.predict(cleaned.head(3))[0]

    assert 0.0 <= prediction.claim_probability <= 1.0
    assert prediction.claim_severity >= 0.0
    assert prediction.predicted_premium >= 0.0


def test_pipeline_is_reproducible_with_fixed_random_state(sample_data_path, tmp_path):
    df = load_policies(sample_data_path, sep=",", validate=True)
    cleaned, _ = clean_policies(df)

    registry_a = ModelRegistry(tmp_path / "a")
    registry_b = ModelRegistry(tmp_path / "b")
    result_a = train_and_register(cleaned, reference_year=2015, registry=registry_a, random_state=42, cv_folds=2)
    result_b = train_and_register(cleaned, reference_year=2015, registry=registry_b, random_state=42, cv_folds=2)

    assert result_a.probability_metrics == result_b.probability_metrics
