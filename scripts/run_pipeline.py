"""ETL entrypoint: Extract -> Validate -> Transform (clean) -> Load -> Verify.

Replaces ``eda/preprocess_data.py``. Produces the cleaned dataset plus a JSON data-quality
report (previously this information only ever went to stdout).

Usage:
    python scripts/run_pipeline.py [--input PATH] [--output PATH] [--sep "|"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from insurance_analytics.config import get_settings
from insurance_analytics.data.clean import clean_policies
from insurance_analytics.data.ingest import DataIngestionError, load_policies
from insurance_analytics.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.raw_data_path)
    parser.add_argument("--output", type=Path, default=settings.cleaned_data_path)
    parser.add_argument("--sep", default=settings.raw_data_sep)
    parser.add_argument("--lenient", action="store_true", help="Drop invalid rows instead of failing")
    args = parser.parse_args()

    try:
        df = load_policies(args.input, sep=args.sep, validate=True, lenient=args.lenient)
    except DataIngestionError as exc:
        logger.error(str(exc))
        return 1

    cleaned, report = clean_policies(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, sep=args.sep, index=False)

    report_path = settings.results_dir / "data_quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))

    logger.info("Wrote cleaned dataset to %s", args.output)
    logger.info("Wrote data-quality report to %s", report_path)

    # Verify: reload what we just wrote and assert row-count integrity (Extract->...->Verify).
    verify_df = load_policies(args.output, sep=args.sep, validate=False)
    assert len(verify_df) == report.output_rows, "Verification failed: written row count mismatch"
    logger.info("Verification passed: %d rows written and confirmed readable", len(verify_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
