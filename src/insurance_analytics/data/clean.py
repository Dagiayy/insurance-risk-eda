"""Cleaning / data-quality stage: Transform step of the ETL pipeline.

Consolidates the imputation logic that was previously duplicated (and drifting out of
sync) across ``eda/eda_summary.py``, ``eda/preprocess_data.py`` and
``src/process/pre_process.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from insurance_analytics.logging_config import get_logger

logger = get_logger(__name__)

# Columns imputed with a categorical placeholder rather than dropped, because in this
# domain "unknown bank" or "unknown marital status" is informative, not missing-at-random.
CATEGORICAL_FILL_UNKNOWN = ["Gender", "Province", "Bank", "AccountType", "MaritalStatus", "NewVehicle"]


@dataclass
class DataQualityReport:
    """Machine-readable summary of what `clean_policies` found and did, for observability."""

    input_rows: int
    output_rows: int
    duplicate_rows_dropped: int
    missing_values_before: dict[str, int] = field(default_factory=dict)
    missing_values_after: dict[str, int] = field(default_factory=dict)
    outliers_flagged: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "duplicate_rows_dropped": self.duplicate_rows_dropped,
            "missing_values_before": self.missing_values_before,
            "missing_values_after": self.missing_values_after,
            "outliers_flagged": self.outliers_flagged,
        }


def profile_data_quality(df: pd.DataFrame, numeric_cols: list[str] | None = None) -> dict:
    """Compute a data-quality snapshot (nulls, duplicates, IQR outliers) without mutating df."""
    numeric_cols = numeric_cols or df.select_dtypes("number").columns.tolist()
    outliers = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[col] = int(((series < lower) | (series > upper)).sum())

    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "outliers_iqr": outliers,
    }


def clean_policies(df: pd.DataFrame, *, dedupe_on: str | None = "UnderwrittenCoverID") -> tuple[pd.DataFrame, DataQualityReport]:
    """Deduplicate and impute a raw (but schema-normalized) policy DataFrame.

    Returns the cleaned frame alongside a :class:`DataQualityReport` so callers (CLI
    scripts, the API, tests) can log or persist exactly what changed instead of scraping
    stdout, which is what the original scripts required.
    """
    input_rows = len(df)
    missing_before = df.isnull().sum().to_dict()

    df = df.copy()

    duplicate_count = 0
    if dedupe_on and dedupe_on in df.columns:
        duplicate_count = int(df.duplicated(subset=[dedupe_on]).sum())
        df = df.drop_duplicates(subset=[dedupe_on])
    else:
        duplicate_count = int(df.duplicated().sum())
        df = df.drop_duplicates()

    if "CustomValueEstimate" in df.columns:
        df["CustomValueEstimate"] = df["CustomValueEstimate"].fillna(df["CustomValueEstimate"].median())
    if "TotalClaims" in df.columns:
        # A missing claim amount means "no claim was filed", not "unknown" -- 0 is the correct impute, not the median.
        df["TotalClaims"] = df["TotalClaims"].fillna(0)
    if "TotalPremium" in df.columns:
        df["TotalPremium"] = df["TotalPremium"].fillna(df["TotalPremium"].median())

    for col in CATEGORICAL_FILL_UNKNOWN:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    outlier_report = profile_data_quality(df, numeric_cols=["TotalPremium", "TotalClaims", "CustomValueEstimate"])

    report = DataQualityReport(
        input_rows=input_rows,
        output_rows=len(df),
        duplicate_rows_dropped=duplicate_count,
        missing_values_before=missing_before,
        missing_values_after=df.isnull().sum().to_dict(),
        outliers_flagged=outlier_report["outliers_iqr"],
    )
    logger.info(
        "Cleaned data: %d -> %d rows (%d duplicates dropped)", input_rows, len(df), duplicate_count
    )
    return df, report
