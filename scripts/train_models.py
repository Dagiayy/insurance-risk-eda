"""Model training entrypoint. Replaces ``eda/modeling.py``.

Trains + registers the severity and probability models via
:func:`insurance_analytics.ml.train.train_and_register`, then writes a metrics report.

Usage:
    python scripts/train_models.py [--input PATH] [--sep "|"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from insurance_analytics.config import get_settings
from insurance_analytics.data.ingest import DataIngestionError, load_policies
from insurance_analytics.logging_config import configure_logging, get_logger
from insurance_analytics.ml.registry import ModelRegistry
from insurance_analytics.ml.train import train_and_register

logger = get_logger(__name__)


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.cleaned_data_path)
    parser.add_argument("--sep", default=settings.raw_data_sep)
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()

    try:
        df = load_policies(args.input, sep=args.sep, validate=False)
    except DataIngestionError as exc:
        logger.error(str(exc))
        return 1

    registry = ModelRegistry(settings.models_dir)
    result = train_and_register(
        df,
        reference_year=settings.current_year,
        registry=registry,
        test_size=settings.test_size,
        random_state=settings.random_state,
        cv_folds=args.cv_folds,
    )

    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    (settings.docs_dir / "severity_model_results.json").write_text(json.dumps(result.severity_metrics, indent=2))
    (settings.docs_dir / "probability_model_results.json").write_text(json.dumps(result.probability_metrics, indent=2))

    logger.info("Best severity model: %s", result.best_severity_model)
    logger.info("Best probability model: %s", result.best_probability_model)
    logger.info("Models registered under %s", settings.models_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
