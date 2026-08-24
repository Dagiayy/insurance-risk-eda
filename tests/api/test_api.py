from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from insurance_analytics.api import deps
from insurance_analytics.api.main import app
from insurance_analytics.config import PricingConfig
from insurance_analytics.data.clean import clean_policies
from insurance_analytics.data.ingest import load_policies
from insurance_analytics.ml.predict import PricingService
from insurance_analytics.ml.registry import ModelRegistry
from insurance_analytics.ml.train import train_and_register


@pytest.fixture
def trained_registry(sample_data_path, tmp_path):
    df = load_policies(sample_data_path, sep=",", validate=True)
    cleaned, _ = clean_policies(df)
    registry = ModelRegistry(tmp_path / "models")
    train_and_register(cleaned, reference_year=2015, registry=registry, cv_folds=2)
    return registry


@pytest.fixture
def client(trained_registry):
    service = PricingService(trained_registry, PricingConfig(), reference_year=2015)
    app.dependency_overrides[deps.get_registry] = lambda: trained_registry
    app.dependency_overrides[deps.get_pricing_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health_endpoint_reports_models_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_predict_premium_returns_valid_response(client):
    payload = {
        "Province": "Gauteng",
        "PostalCode": "2000",
        "Gender": "Male",
        "VehicleType": "Passenger Vehicle",
        "CoverType": "Comprehensive",
        "RegistrationYear": 2012,
        "SumInsured": 200000.0,
        "CustomValueEstimate": 190000.0,
        "TotalPremium": 1500.0,
    }
    response = client.post("/predict/premium", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["claim_probability"] <= 1.0
    assert body["claim_severity"] >= 0.0
    assert body["predicted_premium"] >= 0.0
    assert "severity" in body["model_versions"]


def test_predict_premium_rejects_invalid_payload(client):
    response = client.post("/predict/premium", json={"Province": "Gauteng"})  # missing required fields
    assert response.status_code == 422


def test_insights_endpoint_falls_back_to_template_without_api_key(client, monkeypatch):
    from insurance_analytics.config import get_settings

    get_settings.cache_clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.post("/insights/generate", json={"stats": {"overall_loss_ratio": 0.5, "claim_frequency": 0.04}})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "template"
    get_settings.cache_clear()


def test_eda_endpoint_returns_summary_from_sample_data(client):
    response = client.get("/analysis/eda")
    assert response.status_code == 200
    body = response.json()
    assert "n_rows" in body
    assert body["n_rows"] > 0
