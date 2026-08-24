"""Feature engineering, extracted from ``eda/modeling.py`` and made testable/configurable.

Key fixes versus the original:
  * ``VehicleAge`` used a hard-coded ``2015`` (the year the notebook was written in) instead
    of a configurable reference year, so the code silently became wrong the moment it was
    reused in 2016+. Now takes ``reference_year`` explicitly (defaults to
    ``Settings.current_year``).
  * ``PostalCode`` was label-encoded (0..N integers), imposing a fake ordinal relationship
    on a nominal, high-cardinality field. Replaced with frequency encoding, which is
    monotonic in a meaningful quantity (exposure volume) rather than arbitrary.
  * Rare-category grouping threshold is a named constant instead of a magic ``0.01`` buried
    in a loop.
"""

from __future__ import annotations

import pandas as pd

RARE_CATEGORY_THRESHOLD = 0.01  # categories representing < 1% of rows are grouped into "Other"

DEFAULT_CATEGORICAL_COLUMNS = ["Province", "VehicleType", "CoverType", "Gender", "MaritalStatus"]


def engineer_features(df: pd.DataFrame, *, reference_year: int) -> pd.DataFrame:
    """Derive modeling features from cleaned policy data. Pure function: does not mutate input."""
    df = df.copy()

    if "RegistrationYear" in df.columns:
        df["VehicleAge"] = (reference_year - df["RegistrationYear"]).clip(lower=0)
    if "NewVehicle" in df.columns:
        df["IsNewVehicle"] = (df["NewVehicle"] == "New").astype(int)
    if {"TotalPremium", "SumInsured"}.issubset(df.columns):
        df["PremiumPerSumInsured"] = df["TotalPremium"] / df["SumInsured"].clip(lower=1)
    if "TotalClaims" in df.columns:
        df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    if "PostalCode" in df.columns:
        freq = df["PostalCode"].value_counts(normalize=True)
        df["PostalCodeFrequency"] = df["PostalCode"].map(freq).fillna(0.0)

    return df


def _group_rare_categories(series: pd.Series, threshold: float) -> pd.Series:
    counts = series.value_counts(normalize=True)
    rare = counts[counts < threshold].index
    return series.where(~series.isin(rare), "Other")


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    *,
    rare_threshold: float = RARE_CATEGORY_THRESHOLD,
    fitted_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Group rare categories and one-hot encode. If ``fitted_columns`` is given (inference
    time), the output is reindexed to match the training-time column set exactly -- this is
    what the original ``modeling.py`` was missing, so a saved model could never actually be
    reused on new data (its one-hot columns would never line up).
    """
    df = df.copy()
    categorical_cols = categorical_cols or DEFAULT_CATEGORICAL_COLUMNS
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = "Unknown"
            continue
        df[col] = df[col].fillna("Unknown")
        df[col] = _group_rare_categories(df[col], rare_threshold)

    df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)

    if fitted_columns is not None:
        df = df.reindex(columns=fitted_columns, fill_value=0)

    return df


def to_numeric_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Coerce the selected feature columns to numeric, dropping any that remain non-numeric."""
    matrix = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    return matrix.dropna(axis=1, how="all")
