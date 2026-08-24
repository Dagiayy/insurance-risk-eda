# Architecture

## Data flow

```
                         ┌─────────────────────────┐
                         │  data/*.txt (DVC, 530MB) │  or  data/sample/*.csv (committed, synthetic)
                         └────────────┬─────────────┘
                                      │
                          insurance_analytics.data.ingest
                       (read, normalize column aliases, pandera-validate)
                                      │
                          insurance_analytics.data.clean
                    (dedupe, targeted imputation, data-quality report)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                     │
     insurance_analytics.analysis          insurance_analytics.features.engineering
     (EDA summary, hypothesis tests)        (VehicleAge, frequency-encode PostalCode,
                    │                        rare-category grouping, one-hot encode)
                    │                                     │
             docs/, results/                    insurance_analytics.ml.train
                                                (severity + probability models,
                                                 GridSearchCV, registered via
                                                 insurance_analytics.ml.registry)
                                                          │
                                                    models/<name>/<version>/
                                                          │
                                          insurance_analytics.ml.predict.PricingService
                                                          │
                                             insurance_analytics.api (FastAPI)
                                          /health /predict/premium /analysis/eda
                                                   /insights/generate
                                                          │
                                    insurance_analytics.ai.insight_generator
                                 (LLM narration of the pre-computed stats above,
                                  template fallback with no API key)
```

## Why these layers, and not others

The mandate for this upgrade explicitly warns against overengineering. Here's the
reasoning behind every non-trivial architectural choice:

- **No Airflow/Dagster/Prefect.** There is one batch dataset and four sequential steps
  (ingest -> clean -> analyze -> train). A DAG orchestrator earns its keep when there are
  many interdependent, scheduled jobs; here it would be pure ceremony. `scripts/*.py` are
  plain, composable CLI steps a cron job or CI workflow can call directly.
- **No Spark/Kafka.** The dataset is ~1M rows / 530MB -- comfortably in pandas' working
  set on a single machine. Distributed compute would add an operations burden with no
  performance benefit at this scale. If the data grew 100x-1000x, the first move would be
  chunked/columnar reads (Parquet) before reaching for a cluster.
- **No MLflow server / model-registry service.** `insurance_analytics.ml.registry.ModelRegistry`
  gives every trained model a versioned home (`models/<name>/<version>/model.joblib` +
  `metadata.json` + a `latest` pointer) with the same information MLflow's registry would
  store. Running an MLflow tracking server for a single-contributor, single-node project
  is infrastructure the project doesn't need yet; the registry's interface is narrow
  enough to swap in a hosted tracker later without touching training code.
- **No vector database / RAG.** The AI feature here is narration over already-computed,
  already-validated statistics -- there's no corpus to retrieve from. A vector DB would
  solve a problem this project doesn't have.
- **FastAPI, not a full web app.** The real-world consumer of this system is another
  service or an internal tool (pricing engine, ops dashboard) calling an API, not an
  end-user browsing pages. A JSON API is the right surface.
- **pandera for validation, not a hand-rolled `if` chain.** The original code had zero
  schema validation, so malformed rows (negative premiums, nonsense years) reached
  plotting and modeling code silently. `PolicySchema` fails fast, with a readable report
  of exactly which rows/columns are wrong.

## Key design decisions worth calling out

- **Column-name normalization at the ingestion boundary.** The source data mixes casing
  (`make`/`Make`, `cubiccapacity`/`Cubiccapacity`). Every downstream module works with one
  canonical name per field (`insurance_analytics.schemas.policy.COLUMN_ALIASES`), so this
  is handled once, not five times inconsistently.
- **Shared `prepare_features` for train and inference.** The original code one-hot
  encoded at train time and never saved which columns resulted, so a trained model could
  never be correctly re-applied to new data. `insurance_analytics.ml.pipeline.prepare_features`
  is the single function both call, with `fitted_columns` guaranteeing inference-time
  alignment.
- **No `imbalanced-learn`/SMOTE**, despite it being a declared (and never-used) dependency
  in the original `requirements.txt`. Class weighting (`class_weight="balanced"`,
  `scale_pos_weight`) is used instead -- see `insurance_analytics/ml/train.py` docstring
  for the reasoning (synthesizing interpolated claim amounts for a regulated pricing
  model is hard to justify).
- **AI insights are narration-only, not generation-from-raw-data.** `InsightGenerator`
  is only ever given a JSON blob of statistics that were already computed and validated
  by the analysis layer. This bounds hallucination risk: the model can misdescribe a
  number, but it cannot invent one that wasn't in its input.

## Scaling notes (10x / 100x / 1000x)

- **10x (~10M rows):** unchanged architecture; pandas with `low_memory=False` and
  explicit dtypes still fits comfortably in memory on a modern machine.
- **100x:** switch `data/` storage to partitioned Parquet (column pruning + predicate
  pushdown matter once you're not reading the whole file every run); ingestion would read
  only the columns/partitions a given script needs.
- **1000x:** the batch pandas pipeline stops being the right tool. At that point,
  introducing a proper distributed engine (Spark/DuckDB-on-Parquet) for the
  ingest/clean/feature stages is justified -- but the ML training step would still
  subsample or aggregate before fitting scikit-learn/XGBoost models, since those aren't
  distributed learners.
