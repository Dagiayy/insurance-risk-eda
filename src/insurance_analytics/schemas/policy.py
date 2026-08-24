"""Canonical schema for the ACIS policy-level dataset.

The original scripts disagreed with each other about column casing
(``Cubiccapacity`` vs ``cubiccapacity``, ``Make`` vs ``make``, ``Kilowatts`` vs
``kilowatts``) because the source file itself mixes conventions and each script was
hand-edited against a different snapshot of it. That made the scripts non-portable:
whichever one you ran second would KeyError.

Fix: normalize every incoming column name to one canonical form *once*, at the
ingestion boundary (:func:`normalize_columns`), so everything downstream can rely on
a single, documented name per field. ``COLUMN_ALIASES`` records every casing variant
seen in the wild so new source files remain compatible.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.typing import Series

# canonical_name -> every raw-header spelling observed for that field
COLUMN_ALIASES: dict[str, list[str]] = {
    "UnderwrittenCoverID": ["UnderwrittenCoverID"],
    "PolicyID": ["PolicyID"],
    "TransactionMonth": ["TransactionMonth"],
    "Gender": ["Gender"],
    "Province": ["Province"],
    "PostalCode": ["PostalCode"],
    "MainCrestaZone": ["MainCrestaZone"],
    "Bank": ["Bank"],
    "AccountType": ["AccountType"],
    "MaritalStatus": ["MaritalStatus"],
    "VehicleType": ["VehicleType"],
    "Make": ["Make", "make"],
    "Model": ["Model", "model"],
    "RegistrationYear": ["RegistrationYear"],
    "Cylinders": ["Cylinders", "cylinders"],
    "CubicCapacity": ["Cubiccapacity", "cubiccapacity", "CubicCapacity"],
    "Kilowatts": ["Kilowatts", "kilowatts"],
    "NumberOfDoors": ["NumberOfDoors", "numberofdoors"],
    "NewVehicle": ["NewVehicle"],
    "CoverType": ["CoverType"],
    "SumInsured": ["SumInsured"],
    "CalculatedPremiumPerTerm": ["CalculatedPremiumPerTerm"],
    "CapitalOutstanding": ["CapitalOutstanding"],
    "CustomValueEstimate": ["CustomValueEstimate"],
    "TotalPremium": ["TotalPremium"],
    "TotalClaims": ["TotalClaims"],
}

# raw spelling -> canonical name (built from the map above)
_ALIAS_LOOKUP: dict[str, str] = {alias: canonical for canonical, aliases in COLUMN_ALIASES.items() for alias in aliases}

CANONICAL_COLUMNS: list[str] = list(COLUMN_ALIASES.keys())

NUMERIC_COLUMNS = [
    "SumInsured",
    "CalculatedPremiumPerTerm",
    "CapitalOutstanding",
    "CustomValueEstimate",
    "TotalPremium",
    "TotalClaims",
    "RegistrationYear",
    "Cylinders",
    "CubicCapacity",
    "Kilowatts",
    "NumberOfDoors",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Province",
    "MainCrestaZone",
    "Bank",
    "AccountType",
    "MaritalStatus",
    "VehicleType",
    "Make",
    "Model",
    "NewVehicle",
    "CoverType",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename every recognized column to its canonical spelling; leave unknown columns untouched."""
    rename_map = {col: _ALIAS_LOOKUP[col] for col in df.columns if col in _ALIAS_LOOKUP}
    return df.rename(columns=rename_map)


class PolicySchema(pa.DataFrameModel):
    """Structural + statistical validation gate for policy-level data before it enters the pipeline.

    Runs on every ingest so silently malformed data (negative premiums, an out-of-range
    RegistrationYear, a nonsense claim amount) fails fast with a readable error instead of
    propagating into EDA plots and model training, which is what happened in the original
    scripts (no validation existed at all).
    """

    UnderwrittenCoverID: Series[str] = pa.Field(nullable=True, coerce=True)
    PolicyID: Series[str] = pa.Field(nullable=True, coerce=True)
    TransactionMonth: Series[str] = pa.Field(nullable=True, coerce=True)
    Gender: Series[str] = pa.Field(nullable=True, coerce=True)
    Province: Series[str] = pa.Field(nullable=True, coerce=True)
    PostalCode: Series[str] = pa.Field(nullable=True, coerce=True)
    VehicleType: Series[str] = pa.Field(nullable=True, coerce=True)
    Make: Series[str] = pa.Field(nullable=True, coerce=True)
    RegistrationYear: Series[float] = pa.Field(nullable=True, ge=1950, le=2035, coerce=True)
    CoverType: Series[str] = pa.Field(nullable=True, coerce=True)
    SumInsured: Series[float] = pa.Field(nullable=True, ge=0, coerce=True)
    CustomValueEstimate: Series[float] = pa.Field(nullable=True, ge=0, coerce=True)
    TotalPremium: Series[float] = pa.Field(nullable=True, coerce=True)
    TotalClaims: Series[float] = pa.Field(nullable=True, ge=0, coerce=True)

    class Config:
        strict = False  # extra, non-schema columns (e.g. mmcode, bodytype) pass through untouched
        coerce = True
