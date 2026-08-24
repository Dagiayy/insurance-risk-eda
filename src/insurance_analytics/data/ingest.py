"""Data ingestion: Extract + Validate stages of the Extract -> Validate -> Transform -> Load -> Verify pipeline.

Replaces the copy-pasted ``pd.read_csv(...)`` + ``exit(1)`` pattern that was duplicated,
with slightly different (and sometimes wrong) paths, across four different scripts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandera.errors

from insurance_analytics.logging_config import get_logger
from insurance_analytics.schemas import PolicySchema, normalize_columns

logger = get_logger(__name__)


class DataIngestionError(RuntimeError):
    """Raised when the source file is missing, unreadable, or fails schema validation."""


def load_policies(
    path: str | Path,
    *,
    sep: str = "|",
    validate: bool = True,
    lenient: bool = False,
) -> pd.DataFrame:
    """Load a policy-level dataset, normalize column names, and (optionally) validate its schema.

    Parameters
    ----------
    path: location of the pipe-delimited (or CSV, if sep=",") source file.
    validate: run the :class:`PolicySchema` structural/statistical checks.
    lenient: if True, invalid rows are dropped and logged instead of raising. Useful for
        exploratory scripts; production ingestion should keep this False so bad data is
        never silently discarded.
    """
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(
            f"Source file not found: {path}. If this is DVC-tracked data, run `dvc pull` first, "
            f"or point INSURANCE_ANALYTICS_RAW_DATA_PATH at a local copy."
        )

    logger.info("Loading policies from %s (sep=%r)", path, sep)
    try:
        df = pd.read_csv(path, sep=sep, low_memory=False, engine="python" if sep != "," else "c")
    except Exception as exc:  # pragma: no cover - defensive, exercised via ingest error tests
        raise DataIngestionError(f"Failed to parse {path}: {exc}") from exc

    if df.empty:
        raise DataIngestionError(f"{path} parsed to an empty DataFrame; check the delimiter ({sep!r}).")

    df = normalize_columns(df)
    logger.info("Loaded %d rows x %d columns", *df.shape)

    if validate:
        try:
            df = PolicySchema.validate(df, lazy=True)
        except pandera.errors.SchemaErrors as exc:
            if not lenient:
                raise DataIngestionError(f"Schema validation failed:\n{exc.failure_cases.head(20)}") from exc
            bad_index = exc.failure_cases["index"].dropna().unique()
            logger.warning("Dropping %d rows that failed schema validation", len(bad_index))
            df = df.drop(index=bad_index, errors="ignore")

    return df
