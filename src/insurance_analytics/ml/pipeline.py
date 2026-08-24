"""Shared feature-preparation logic used identically at train time and inference time.

The original ``modeling.py`` built one-hot columns with ``pd.get_dummies`` and never
saved the resulting column list, so a model trained on that matrix could never be
correctly applied to new data (a new batch missing a rare category would produce a
different, misaligned column set). ``prepare_features`` is the single place both
training and inference call, and it's the only thing allowed to know about
:data:`DEFAULT_CATEGORICAL_COLUMNS` / :data:`DROP_COLUMNS`.
"""

from __future__ import annotations

import pandas as pd

from insurance_analytics.features.engineering import DEFAULT_CATEGORICAL_COLUMNS, encode_categoricals, engineer_features

# Columns that must never enter the feature matrix: they leak the targets directly
# (TotalClaims/HasClaim), are pure identifiers with no predictive meaning (they were
# included as "features" in the original script, which is why PolicyID/UnderwrittenCoverID
# showed up as top SHAP features in docs/modeling_report.md -- an identifier "predicting"
# the target is a leakage smell, not a genuine risk driver), or are non-numeric dates.
LEAKAGE_AND_ID_COLUMNS = [
    "TotalClaims",
    "HasClaim",
    "TotalPremium",
    "NewVehicle",
    "TransactionMonth",
    "LossRatio",
    "Margin",
    "PolicyID",
    "UnderwrittenCoverID",
    "PostalCode",  # raw code is a nominal identifier; PostalCodeFrequency (engineered) is used instead
]


def prepare_features(
    df: pd.DataFrame,
    *,
    reference_year: int,
    fitted_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Engineer + encode features. Pass ``fitted_columns`` (from training) at inference time
    so the output matrix is guaranteed to match what the model was trained on."""
    engineered = engineer_features(df, reference_year=reference_year)
    encoded = encode_categoricals(engineered, DEFAULT_CATEGORICAL_COLUMNS, fitted_columns=fitted_columns)

    if fitted_columns is not None:
        return encoded[fitted_columns]

    feature_cols = [c for c in encoded.columns if c not in LEAKAGE_AND_ID_COLUMNS]
    matrix = encoded[feature_cols].apply(pd.to_numeric, errors="coerce")
    return matrix.dropna(axis=1, how="all")
