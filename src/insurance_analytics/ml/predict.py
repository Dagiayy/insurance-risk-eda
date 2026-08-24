"""Inference: load registered models and score new policies, end-to-end.

This is the piece the original project entirely lacked -- ``modeling.py`` computed
``docs/predicted_premiums.csv`` once, in-process, on the test split, and there was no way
to score a brand-new policy record. :class:`PricingService` is what the API layer and any
batch-scoring script use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from insurance_analytics.config import PricingConfig
from insurance_analytics.ml.pipeline import prepare_features
from insurance_analytics.ml.registry import ModelRegistry
from insurance_analytics.ml.train import PROBABILITY_MODEL_NAME, SEVERITY_MODEL_NAME


@dataclass
class PricingPrediction:
    claim_probability: float
    claim_severity: float
    predicted_premium: float


class PricingService:
    """Loads the latest severity + probability models once and scores policy rows."""

    def __init__(self, registry: ModelRegistry, pricing: PricingConfig, reference_year: int):
        self.registry = registry
        self.pricing = pricing
        self.reference_year = reference_year
        self._severity_model, self._severity_meta = registry.load(SEVERITY_MODEL_NAME)
        self._probability_model, self._probability_meta = registry.load(PROBABILITY_MODEL_NAME)

    @property
    def model_versions(self) -> dict[str, str]:
        return {"severity": self._severity_meta.version, "probability": self._probability_meta.version}

    def predict(self, policy: pd.DataFrame) -> list[PricingPrediction]:
        features = prepare_features(
            policy, reference_year=self.reference_year, fitted_columns=self._probability_meta.feature_columns
        )
        proba = self._probability_model.predict_proba(features)[:, 1]

        severity_features = features.reindex(columns=self._severity_meta.feature_columns, fill_value=0)
        log_severity = self._severity_model.predict(severity_features)
        severity = np.expm1(log_severity)
        severity = np.clip(severity, a_min=0, a_max=None)

        predictions = []
        for p, s in zip(proba, severity, strict=True):
            premium = self.pricing.calculate_premium(float(p), float(s))
            predictions.append(PricingPrediction(claim_probability=float(p), claim_severity=float(s), predicted_premium=premium))
        return predictions
