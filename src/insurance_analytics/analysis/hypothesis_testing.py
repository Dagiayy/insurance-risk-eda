"""A/B hypothesis testing for risk drivers, refactored from ``eda/hypothesis_testing.py``.

Behavior is preserved (chi-squared for claim frequency, Welch's t-test for severity and
margin) but each test is now a pure function returning a typed result instead of printing
to stdout, and the four hard-coded province/zip/gender comparisons are data (a list of
:class:`GroupComparison`) instead of copy-pasted branches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind


@dataclass(frozen=True)
class GroupComparison:
    group_col: str
    group_a: object
    group_b: object
    label: str
    metrics: tuple[str, ...] = ("claim_frequency", "claim_severity")


DEFAULT_COMPARISONS: list[GroupComparison] = [
    GroupComparison("Province", "Gauteng", "Western Cape", "Provinces"),
    GroupComparison("PostalCode", 2000, 8000, "Zip Codes (Risk)"),
    GroupComparison("PostalCode", 2000, 8000, "Zip Codes (Margin)", metrics=("margin",)),
    GroupComparison("Gender", "Male", "Female", "Gender"),
]

EQUIVALENCE_ATTRIBUTES = ["VehicleType", "RegistrationYear", "CoverType", "SumInsured"]

SIGNIFICANCE_LEVEL = 0.05


def _claim_frequency_test(df: pd.DataFrame, comparison: GroupComparison) -> dict:
    subset = df[df[comparison.group_col].isin([comparison.group_a, comparison.group_b])]
    contingency = pd.crosstab(subset[comparison.group_col], subset["HasClaim"])
    chi2, p, _, _ = chi2_contingency(contingency)
    a_freq = df.loc[df[comparison.group_col] == comparison.group_a, "HasClaim"].mean()
    b_freq = df.loc[df[comparison.group_col] == comparison.group_b, "HasClaim"].mean()
    return _result("Claim Frequency", comparison, p, chi2, a_freq, b_freq)


def _claim_severity_test(df: pd.DataFrame, comparison: GroupComparison) -> dict:
    a = df.loc[(df[comparison.group_col] == comparison.group_a) & df["HasClaim"].astype(bool), "TotalClaims"]
    b = df.loc[(df[comparison.group_col] == comparison.group_b) & df["HasClaim"].astype(bool), "TotalClaims"]
    if len(a) > 1 and len(b) > 1:
        t_stat, p = ttest_ind(a, b, equal_var=False)
    else:
        t_stat, p = np.nan, np.nan
    return _result("Claim Severity", comparison, p, t_stat, a.mean(), b.mean())


def _margin_test(df: pd.DataFrame, comparison: GroupComparison) -> dict:
    a = df.loc[df[comparison.group_col] == comparison.group_a, "Margin"]
    b = df.loc[df[comparison.group_col] == comparison.group_b, "Margin"]
    t_stat, p = ttest_ind(a, b, equal_var=False)
    return _result("Margin", comparison, p, t_stat, a.mean(), b.mean())


def _result(metric: str, comparison: GroupComparison, p: float, statistic: float, a_val: float, b_val: float) -> dict:
    return {
        "metric": metric,
        "group": comparison.label,
        "p_value": float(p) if p == p else None,  # NaN check
        "statistic": float(statistic) if statistic == statistic else None,
        "group_a_value": float(a_val) if a_val == a_val else None,
        "group_b_value": float(b_val) if b_val == b_val else None,
        "significant": bool(p < SIGNIFICANCE_LEVEL) if p == p else None,
    }


_TEST_FUNCS = {
    "claim_frequency": _claim_frequency_test,
    "claim_severity": _claim_severity_test,
    "margin": _margin_test,
}


def check_group_equivalence(df: pd.DataFrame, comparison: GroupComparison, attributes: list[str] | None = None) -> pd.DataFrame:
    """Check whether the two compared groups differ on confounding attributes.

    A significant difference here means a rejected null hypothesis on the metric of
    interest could be confounded rather than causal -- exactly the caveat the original
    report's "Limitation" section called out but never automated.
    """
    attributes = attributes or EQUIVALENCE_ATTRIBUTES
    rows = []
    for attr in attributes:
        if attr not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[attr]):
            subset = df[df[comparison.group_col].isin([comparison.group_a, comparison.group_b])]
            table = pd.crosstab(subset[comparison.group_col], subset[attr])
            if table.shape[0] == 2 and table.shape[1] > 0:
                chi2, p, _, _ = chi2_contingency(table)
                rows.append({"attribute": attr, "test": "chi-squared", "p_value": p, "group": comparison.label})
        else:
            a = df.loc[df[comparison.group_col] == comparison.group_a, attr].dropna()
            b = df.loc[df[comparison.group_col] == comparison.group_b, attr].dropna()
            if len(a) > 1 and len(b) > 1:
                t_stat, p = ttest_ind(a, b, equal_var=False)
                rows.append({"attribute": attr, "test": "t-test", "p_value": p, "group": comparison.label})
    return pd.DataFrame(rows)


def run_all_hypothesis_tests(
    df: pd.DataFrame, comparisons: list[GroupComparison] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every configured group comparison. Returns (results_df, equivalence_df)."""
    comparisons = comparisons or DEFAULT_COMPARISONS
    df = df.copy()
    df["HasClaim"] = df["TotalClaims"] > 0
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    results, equivalence = [], []
    for comparison in comparisons:
        equivalence.append(check_group_equivalence(df, comparison))
        for metric in comparison.metrics:
            results.append(_TEST_FUNCS[metric](df, comparison))

    return pd.DataFrame(results), pd.concat(equivalence, ignore_index=True) if equivalence else pd.DataFrame()
