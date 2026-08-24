import numpy as np
import pandas as pd

from insurance_analytics.monitoring.data_drift import detect_drift


def test_detect_drift_flags_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"TotalPremium": rng.normal(1000, 100, size=500)})
    current = pd.DataFrame({"TotalPremium": rng.normal(1800, 100, size=500)})  # clearly shifted

    results = detect_drift(reference, current, numeric_columns=["TotalPremium"])
    assert len(results) == 1
    assert results[0].drifted is True
    assert results[0].p_value < 0.05


def test_detect_drift_does_not_flag_identical_distribution():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"TotalPremium": rng.normal(1000, 100, size=500)})
    current = reference.copy()

    results = detect_drift(reference, current, numeric_columns=["TotalPremium"])
    assert results[0].drifted is False
    assert results[0].p_value == 1.0
