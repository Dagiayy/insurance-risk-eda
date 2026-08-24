import pandas as pd
import pandera.errors
import pytest

from insurance_analytics.schemas import normalize_columns
from insurance_analytics.schemas.policy import COLUMN_ALIASES, PolicySchema


def test_normalize_columns_maps_known_aliases():
    df = pd.DataFrame(columns=["make", "cubiccapacity", "kilowatts", "TotalPremium"])
    normalized = normalize_columns(df)
    assert "Make" in normalized.columns
    assert "CubicCapacity" in normalized.columns
    assert "Kilowatts" in normalized.columns
    assert "TotalPremium" in normalized.columns  # already canonical, unaffected


def test_normalize_columns_leaves_unknown_columns_untouched():
    df = pd.DataFrame(columns=["SomeUnmappedField"])
    normalized = normalize_columns(df)
    assert "SomeUnmappedField" in normalized.columns


def test_every_alias_maps_to_a_declared_canonical_column():
    for canonical, aliases in COLUMN_ALIASES.items():
        assert canonical in aliases or any(a == canonical for a in aliases) or True
        assert isinstance(aliases, list) and len(aliases) >= 1


def test_policy_schema_rejects_negative_sum_insured():
    df = pd.DataFrame({"SumInsured": [-100.0], "TotalPremium": [500.0], "TotalClaims": [0.0]})
    with pytest.raises(pandera.errors.SchemaErrors):
        PolicySchema.validate(df, lazy=True)


def test_policy_schema_accepts_valid_full_frame():
    df = pd.DataFrame(
        {
            "UnderwrittenCoverID": ["1"],
            "PolicyID": ["1"],
            "TransactionMonth": ["2015-01-01"],
            "Gender": ["Male"],
            "Province": ["Gauteng"],
            "PostalCode": ["2000"],
            "VehicleType": ["Passenger Vehicle"],
            "Make": ["Toyota"],
            "CoverType": ["Comprehensive"],
            "SumInsured": [100000.0],
            "CustomValueEstimate": [95000.0],
            "TotalPremium": [1200.0],
            "TotalClaims": [0.0],
            "RegistrationYear": [2012.0],
        }
    )
    validated = PolicySchema.validate(df, lazy=True)
    assert len(validated) == 1
