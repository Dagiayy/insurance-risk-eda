"""Hypothesis-testing entrypoint. Replaces ``eda/hypothesis_testing.py``.

Usage:
    python scripts/run_hypothesis_tests.py [--input PATH] [--sep "|"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insurance_analytics.analysis.hypothesis_testing import run_all_hypothesis_tests
from insurance_analytics.config import get_settings
from insurance_analytics.data.ingest import DataIngestionError, load_policies
from insurance_analytics.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.cleaned_data_path)
    parser.add_argument("--sep", default=settings.raw_data_sep)
    args = parser.parse_args()

    try:
        df = load_policies(args.input, sep=args.sep, validate=False)
    except DataIngestionError as exc:
        logger.error(str(exc))
        return 1

    results, equivalence = run_all_hypothesis_tests(df)

    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(settings.docs_dir / "hypothesis_test_results.csv", index=False)
    equivalence.to_csv(settings.docs_dir / "equivalence_check_results.csv", index=False)

    logger.info("Wrote hypothesis test results to %s", settings.docs_dir)
    print(results.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
