"""Lightweight data-quality / drift monitoring.

Not a full Evidently/whylogs integration -- the project has one batch dataset and no
streaming ingestion, so a scheduled service would be pure ceremony. This gives the same
signal (has the incoming data distribution shifted enough to distrust the model?) as a
plain function you can call from a scheduled job or the API, using the Kolmogorov-Smirnov
test that's already a `scipy` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.stats import ks_2samp

DRIFT_P_VALUE_THRESHOLD = 0.05


@dataclass
class DriftResult:
    column: str
    p_value: float
    drifted: bool
    reference_mean: float
    current_mean: float


def detect_drift(reference: pd.DataFrame, current: pd.DataFrame, numeric_columns: list[str]) -> list[DriftResult]:
    """Two-sample KS test per numeric column: is `current` drawn from the same distribution as `reference`?"""
    results = []
    for col in numeric_columns:
        if col not in reference.columns or col not in current.columns:
            continue
        ref_series = pd.to_numeric(reference[col], errors="coerce").dropna()
        cur_series = pd.to_numeric(current[col], errors="coerce").dropna()
        if len(ref_series) < 2 or len(cur_series) < 2:
            continue
        statistic, p_value = ks_2samp(ref_series, cur_series)
        results.append(
            DriftResult(
                column=col,
                p_value=float(p_value),
                drifted=bool(p_value < DRIFT_P_VALUE_THRESHOLD),
                reference_mean=float(ref_series.mean()),
                current_mean=float(cur_series.mean()),
            )
        )
    return results
