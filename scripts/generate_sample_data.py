"""Generate a small, schema-correct synthetic sample dataset.

The real dataset (``data/MachineLearningRating_v3.txt``) is DVC-tracked and ~530MB; it is
not committed to the repo and won't be present on a fresh checkout without ``dvc pull``
against a configured remote. That made the entire project impossible to run, test, or
demo out of the box. This script produces a small, statistically plausible stand-in with
the exact canonical schema so tests, the API, and local demos work with zero setup.

Usage: python scripts/generate_sample_data.py [--rows 800] [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROVINCES = ["Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape", "Free State", "Limpopo"]
VEHICLE_TYPES = ["Passenger Vehicle", "Light Commercial Vehicle", "Bus", "Medium Commercial Vehicle"]
COVER_TYPES = ["Comprehensive", "Third Party", "Third Party Fire and Theft"]
MAKES = ["Toyota", "Volkswagen", "Ford", "BMW", "Nissan", "Mercedes-Benz", "Hyundai"]
GENDERS = ["Male", "Female", "Not specified"]
MARITAL = ["Married", "Single", "Unknown"]
BANKS = ["ABSA", "Standard Bank", "FNB", "Nedbank", "Unknown"]


def generate(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    registration_year = rng.integers(1998, 2016, size=n_rows)
    sum_insured = rng.lognormal(mean=11.5, sigma=0.6, size=n_rows).round(2)
    custom_value = (sum_insured * rng.uniform(0.85, 1.05, size=n_rows)).round(2)
    total_premium = (sum_insured * rng.uniform(0.003, 0.012, size=n_rows)).round(2)

    # ~5% claim rate: rare-event but frequent enough that a synthetic 800-row sample still
    # has 30-40 claim rows for the severity model to train on in tests/demos.
    has_claim = rng.random(n_rows) < 0.05
    claim_amount = np.where(
        has_claim,
        rng.lognormal(mean=9.2, sigma=0.9, size=n_rows).round(2),
        0.0,
    )

    months = pd.date_range("2014-02-01", "2015-08-01", freq="MS")

    df = pd.DataFrame(
        {
            "UnderwrittenCoverID": np.arange(1, n_rows + 1),
            "PolicyID": rng.integers(10_000, 99_999, size=n_rows),
            "TransactionMonth": rng.choice(months, size=n_rows).astype(str),
            "Gender": rng.choice(GENDERS, size=n_rows, p=[0.55, 0.35, 0.10]),
            "Province": rng.choice(PROVINCES, size=n_rows),
            "PostalCode": rng.choice([2000, 8000, 4000, 1000, 6000, 9000], size=n_rows),
            "MainCrestaZone": rng.choice(["Zone A", "Zone B", "Zone C"], size=n_rows),
            "Bank": rng.choice(BANKS, size=n_rows),
            "AccountType": rng.choice(["Cheque", "Savings", "Unknown"], size=n_rows),
            "MaritalStatus": rng.choice(MARITAL, size=n_rows),
            "VehicleType": rng.choice(VEHICLE_TYPES, size=n_rows, p=[0.7, 0.15, 0.05, 0.10]),
            "Make": rng.choice(MAKES, size=n_rows),
            "Model": [f"Model-{i}" for i in rng.integers(1, 20, size=n_rows)],
            "RegistrationYear": registration_year,
            "Cylinders": rng.choice([3, 4, 6, 8], size=n_rows, p=[0.1, 0.6, 0.25, 0.05]),
            "cubiccapacity": rng.integers(1000, 4000, size=n_rows),
            "kilowatts": rng.integers(50, 250, size=n_rows),
            "NumberOfDoors": rng.choice([2, 4, 5], size=n_rows, p=[0.15, 0.55, 0.30]),
            "NewVehicle": rng.choice(["New", "Used", "Unknown"], size=n_rows, p=[0.1, 0.85, 0.05]),
            "CoverType": rng.choice(COVER_TYPES, size=n_rows, p=[0.6, 0.25, 0.15]),
            "SumInsured": sum_insured,
            "CalculatedPremiumPerTerm": (total_premium * 12).round(2),
            "CapitalOutstanding": (sum_insured * rng.uniform(0.0, 0.8, size=n_rows)).round(2),
            "CustomValueEstimate": custom_value,
            "TotalPremium": total_premium,
            "TotalClaims": claim_amount,
        }
    )

    # sprinkle a small amount of realistic missingness, mirroring the real dataset
    for col, frac in [("Gender", 0.05), ("CustomValueEstimate", 0.08), ("Bank", 0.1)]:
        mask = rng.random(n_rows) < frac
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "sample" / "sample_policies.csv")
    args = parser.parse_args()

    df = generate(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic policy rows to {args.out}")


if __name__ == "__main__":
    main()
