"""Reusable, headless EDA computations.

The original ``eda/eda_summary.py`` mixed data loading, plotting, and analysis in one
365-line top-level script with no functions -- impossible to unit test or reuse (e.g.
from the API or a notebook) without re-running the entire file, including all the
``plt.savefig`` calls. Here the *numbers* are pure functions; plotting is a thin,
separate layer (see :mod:`insurance_analytics.analysis.plots`) that callers opt into.
"""

from __future__ import annotations

import pandas as pd

NUMERIC_COLUMNS = ["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate"]


def compute_eda_summary(df: pd.DataFrame) -> dict:
    """Return a JSON-serializable summary: dtypes, descriptive stats, missingness, loss ratios."""
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]

    summary: dict = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "descriptive_stats": df[numeric_cols].describe().to_dict() if numeric_cols else {},
    }

    if {"TotalClaims", "TotalPremium"}.issubset(df.columns):
        loss_ratio = df["TotalClaims"] / df["TotalPremium"].replace(0, pd.NA)
        summary["overall_loss_ratio"] = float(loss_ratio.mean(skipna=True))
        for group_col in ("Province", "VehicleType", "Gender", "CoverType"):
            if group_col in df.columns:
                grouped = (df["TotalClaims"] / df["TotalPremium"].replace(0, pd.NA)).groupby(df[group_col]).mean()
                summary[f"loss_ratio_by_{group_col.lower()}"] = grouped.sort_values(ascending=False).round(4).to_dict()

    if "TotalClaims" in df.columns:
        summary["claim_frequency"] = float((df["TotalClaims"] > 0).mean())
        claimed = df.loc[df["TotalClaims"] > 0, "TotalClaims"]
        summary["claim_severity_mean"] = float(claimed.mean()) if not claimed.empty else None

    if "TransactionMonth" in df.columns:
        month = pd.to_datetime(df["TransactionMonth"], errors="coerce")
        monthly_premium = df.groupby(month.dt.to_period("M"))["TotalPremium"].mean() if "TotalPremium" in df.columns else None
        if monthly_premium is not None:
            summary["monthly_avg_premium"] = {str(k): round(float(v), 2) for k, v in monthly_premium.items()}

    return summary


def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or [c for c in NUMERIC_COLUMNS if c in df.columns]
    return df[columns].corr()


def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.Series:
    q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return df[(df[column] < lower) | (df[column] > upper)][column]
