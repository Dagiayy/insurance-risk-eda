# ACIS Insurance Risk & Pricing Analytics

Data engineering, EDA, risk-based pricing models, and an AI insight layer for AlphaCare
Insurance Solutions (ACIS), built on eighteen months (Feb 2014 - Aug 2015) of South
African motor insurance policy data.

## 1. What this does and why

ACIS wants to know two things: **which segments are under- or over-priced today**, and
**what should a new policy's premium be, risk-adjusted**. This repo answers both:

- **Analytics** — loss-ratio and claim-frequency/severity breakdowns by province, vehicle
  type, cover type and gender, plus formal A/B hypothesis tests on the risk drivers
  (`docs/hypothesis_testing_report.md`).
- **Pricing models** — a two-stage model (claim probability x claim severity) that
  produces a risk-based premium per policy, served over an HTTP API.
- **AI insight layer** — turns the computed statistics into a plain-English narrative for
  non-technical stakeholders, via an LLM constrained to the pre-computed numbers.

## 2. Architecture

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full breakdown. In short:

```
Raw data (DVC) --> Ingest+Validate (pandera) --> Clean --> Feature engineering
    --> [Analysis: EDA, hypothesis tests]  --> docs/, results/
    --> [ML: train -> registry -> predict] --> models/
    --> FastAPI (predict / insights / analysis / health)
```

Everything under `src/insurance_analytics/` is an installable, unit-tested Python
package; `scripts/` holds thin CLI entrypoints that call into it; `src/insurance_analytics/api/`
serves it over HTTP.

## 3. What changed from the original project

This started as a four-task academic EDA/modeling exercise (see `legacy/` for the
original scripts, kept for reference, not imported by anything). It has been rebuilt into
a packaged, tested, served, and documented system. Full details, including every bug
found and fixed, are in [`UPGRADE_REPORT.md`](UPGRADE_REPORT.md).

## 4. Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev,ai]"

# No 529MB dataset required to try this out -- generates a small synthetic, schema-correct sample:
python scripts/generate_sample_data.py

# Run the full local pipeline against the sample data:
python scripts/run_pipeline.py --input data/sample/sample_policies.csv --sep ","
python scripts/run_eda.py --input data/sample/sample_policies_cleaned.csv --sep ","
python scripts/run_hypothesis_tests.py --input data/sample/sample_policies_cleaned.csv --sep ","
python scripts/train_models.py --input data/sample/sample_policies_cleaned.csv --sep ","

# Serve the trained models:
uvicorn insurance_analytics.api.main:app --reload
# -> open http://127.0.0.1:8000/docs for interactive API docs
```

Or with Docker:

```bash
docker compose up --build
curl http://localhost:8000/health
```

### Using the real dataset

The production dataset is DVC-tracked and not committed (it's ~530MB):

```bash
pip install dvc
dvc remote add -d localstorage ~/dvc_storage/insurance-risk-eda   # or your own remote
dvc pull
python scripts/run_pipeline.py    # defaults to data/MachineLearningRating_v3.txt, sep="|"
python scripts/train_models.py
```

## 5. Configuration

All configuration is environment-driven (`insurance_analytics.config.Settings`, pydantic-settings).
Copy `.env.example` to `.env` and adjust as needed:

| Variable | Purpose |
|---|---|
| `PRICING_EXPENSE_LOADING_PCT`, `PRICING_PROFIT_MARGIN_PCT` | Premium formula constants |
| `API_KEY` | Optional shared secret required on `/predict`, `/insights` (unset = open, for local dev) |
| `ANTHROPIC_API_KEY` | Enables AI-narrated `/insights/generate`; falls back to a deterministic template without it |
| `LOG_LEVEL`, `LOG_JSON` | Logging verbosity / structured (JSON) output |
| `RANDOM_STATE`, `TEST_SIZE`, `CURRENT_YEAR` | ML reproducibility and feature-engineering reference year |

## 6. Running tests

```bash
python scripts/generate_sample_data.py   # tests use this, not the 530MB production file
pytest                                   # 39 tests: unit, integration, API
pytest --cov=insurance_analytics --cov-report=term-missing
ruff check src scripts tests
```

## 7. API usage

Interactive docs at `/docs` once the server is running. Key endpoints:

- `GET /health` — service + model-registry status.
- `POST /predict/premium` — score a policy: `{claim_probability, claim_severity, predicted_premium, model_versions}`.
- `GET /analysis/eda` — computed EDA summary (loss ratios, claim frequency, correlations) as JSON.
- `POST /insights/generate` — turn a stats JSON blob into a narrative report (AI or template fallback).

Example:

```bash
curl -X POST http://localhost:8000/predict/premium \
  -H "Content-Type: application/json" \
  -d '{"Province":"Gauteng","PostalCode":"2000","Gender":"Male","VehicleType":"Passenger Vehicle",
       "CoverType":"Comprehensive","RegistrationYear":2012,"SumInsured":200000,
       "CustomValueEstimate":190000,"TotalPremium":1500}'
```

## 8. Example workflows (who uses this and how)

- **Pricing actuary**: calls `/predict/premium` from an underwriting tool to get a
  risk-based quote for a new policy.
- **Data/ML engineer**: runs `scripts/run_pipeline.py` and `scripts/train_models.py` on a
  schedule (or CI) to refresh the cleaned dataset and retrain models against new data;
  `insurance_analytics.monitoring.data_drift.detect_drift` flags when a new batch's
  distribution has shifted enough to warrant a retrain.
- **Ops / underwriting manager**: reads `docs/hypothesis_testing_report.md` and
  `docs/modeling_report.md`, or calls `/insights/generate` for a live narrative version.
- **Engineer onboarding**: reads `ARCHITECTURE.md`, runs the Quickstart above with the
  synthetic sample data, no production data access required.

## 9. Project structure

```
src/insurance_analytics/
  config.py, logging_config.py     # environment-driven settings, structured logging
  schemas/                         # canonical column names + pandera validation schema
  data/                            # ingest (Extract+Validate), clean (Transform)
  features/                        # feature engineering (age, encoding, frequency encoding)
  analysis/                        # EDA summary + hypothesis testing (pure functions) + plots
  ml/                              # pipeline, train, evaluate, registry, predict
  monitoring/                      # KS-test based data drift detection
  ai/                              # LLM-powered insight narration, with template fallback
  api/                             # FastAPI app: health, predict, insights, analysis routers
scripts/            # CLI entrypoints wrapping the package above
tests/               # unit / integration / api tests (39 tests, no production data needed)
data/sample/         # synthetic, schema-correct dataset for tests, demos, CI
models/              # filesystem model registry (gitignored artifacts; regenerate via `make train`)
legacy/              # original pre-upgrade scripts, kept for reference only
docs/, results/      # analysis & modeling reports
notebooks/           # original exploratory notebooks (unchanged; not part of the production path)
```

## 10. Limitations

- The severity model's R² is modest (~0.3 on the original full dataset) -- claim amounts
  are noisy and the available features explain limited variance. Treat premium output as
  a risk-ranking signal, not a guaranteed-accurate quote.
- Claim probability is a rare-event problem (~0.3% claim rate on the real book); F1 will
  look low even for a well-calibrated model. Precision/recall and ROC-AUC, not accuracy,
  are the metrics that matter here.
- The model registry is filesystem-based, single-node. That's the right amount of
  infrastructure for this project's scale (see `UPGRADE_REPORT.md` for the reasoning);
  swapping in a hosted tracking server is a registry-interface change, not a rewrite.
- `/insights/generate` never invents numbers (it's only ever given pre-computed stats),
  but LLM-generated prose should still be reviewed before it reaches a regulator or a
  customer-facing document.

## 11. Future improvements

- Batch drift monitoring wired into a scheduled job, alerting when `detect_drift` fires.
- Threshold tuning / cost-sensitive learning for the claim-probability model instead of
  the default 0.5 cutoff, given the class imbalance.
- A hosted experiment tracker (MLflow/W&B) if/when multiple people are training in
  parallel and need to compare runs beyond the local registry's JSON metadata.
