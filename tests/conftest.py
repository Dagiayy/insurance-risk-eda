from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SAMPLE_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_policies.csv"


@pytest.fixture
def raw_policy_df() -> pd.DataFrame:
    """A tiny, deterministic synthetic policy DataFrame -- fast, no I/O, exercises every field."""
    rng = np.random.default_rng(7)
    n = 200
    has_claim = rng.random(n) < 0.15
    return pd.DataFrame(
        {
            "UnderwrittenCoverID": np.arange(n),
            "PolicyID": np.arange(1000, 1000 + n),
            "TransactionMonth": pd.date_range("2014-02-01", periods=n, freq="D").astype(str),
            "Gender": rng.choice(["Male", "Female", None], size=n, p=[0.45, 0.45, 0.10]),
            "Province": rng.choice(["Gauteng", "Western Cape", "KwaZulu-Natal"], size=n),
            "PostalCode": rng.choice([2000, 8000, 4000], size=n),
            "VehicleType": rng.choice(["Passenger Vehicle", "Bus"], size=n, p=[0.9, 0.1]),
            "make": rng.choice(["Toyota", "BMW", "Ford"], size=n),  # lowercase alias, exercises normalization
            "RegistrationYear": rng.integers(2000, 2016, size=n),
            "cubiccapacity": rng.integers(1000, 3000, size=n),  # lowercase alias
            "kilowatts": rng.integers(60, 200, size=n),
            "NewVehicle": rng.choice(["New", "Used", None], size=n, p=[0.1, 0.8, 0.1]),
            "CoverType": rng.choice(["Comprehensive", "Third Party"], size=n),
            "SumInsured": rng.uniform(50_000, 400_000, size=n).round(2),
            "CustomValueEstimate": rng.uniform(40_000, 380_000, size=n).round(2),
            "MaritalStatus": rng.choice(["Married", "Single", None], size=n),
            "TotalPremium": rng.uniform(300, 3000, size=n).round(2),
            "TotalClaims": np.where(has_claim, rng.uniform(1000, 60_000, size=n).round(2), 0.0),
        }
    )


@pytest.fixture
def sample_data_path() -> Path:
    if not SAMPLE_DATA_PATH.exists():
        pytest.skip("Sample data not generated. Run: python scripts/generate_sample_data.py")
    return SAMPLE_DATA_PATH
