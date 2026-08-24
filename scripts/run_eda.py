"""EDA entrypoint: computes the analytics summary and (optionally) renders plots.

Replaces ``eda/eda_summary.py``. Same computations, but they now live in
``insurance_analytics.analysis`` where they're unit tested, and results are written as
structured JSON instead of only being printed.

Usage:
    python scripts/run_eda.py [--input PATH] [--sep "|"] [--no-plots]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from insurance_analytics.analysis.eda import compute_eda_summary, correlation_matrix
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
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    try:
        df = load_policies(args.input, sep=args.sep, validate=False)
    except DataIngestionError as exc:
        logger.error(str(exc))
        return 1

    summary = compute_eda_summary(df)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = settings.results_dir / "eda_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Wrote EDA summary to %s", out_path)

    if not args.no_plots:
        from insurance_analytics.analysis.plots import save_correlation_heatmap, save_histogram

        for col in ["TotalPremium", "TotalClaims", "SumInsured", "CustomValueEstimate"]:
            if col in df.columns:
                save_histogram(df, col, settings.plots_dir)
        corr = correlation_matrix(df)
        save_correlation_heatmap(corr, settings.plots_dir)
        logger.info("Wrote plots to %s", settings.plots_dir)

    print(json.dumps(summary, indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
