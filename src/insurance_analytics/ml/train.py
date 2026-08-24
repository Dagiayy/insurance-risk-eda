"""Training pipeline for the two-stage severity/probability pricing model.

Refactor of ``eda/modeling.py`` with the following substantive fixes:

  * Models are now persisted (:mod:`insurance_analytics.ml.registry`) instead of being
    fit-and-discarded in a single script run.
  * The engineered feature matrix (with its exact one-hot column set) is saved alongside
    the model so inference can reproduce it exactly -- previously impossible.
  * ``imbalanced-learn`` was a declared dependency but never imported anywhere in the
    original code. The severe class imbalance (~0.3% claim rate) is handled the way the
    code actually already tried to (``class_weight="balanced"`` / ``scale_pos_weight``),
    which is the safer choice for a rare, monetary event: SMOTE would synthesize
    interpolated claim amounts between real policies, which is hard to justify for a
    regulated pricing model. That dependency is removed; the decision is documented here
    instead of silently ignored.
  * Hyperparameter search space, CV folds, and test split are configurable, not hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from insurance_analytics.logging_config import get_logger
from insurance_analytics.ml.evaluate import evaluate_classification, evaluate_regression
from insurance_analytics.ml.pipeline import prepare_features
from insurance_analytics.ml.registry import ModelMetadata, ModelRegistry, new_version_string

logger = get_logger(__name__)

SEVERITY_MODEL_NAME = "claim_severity"
PROBABILITY_MODEL_NAME = "claim_probability"

HYPERPARAM_GRID = {"n_estimators": [50, 100], "max_depth": [10, 20]}


@dataclass
class TrainingResult:
    severity_metrics: dict[str, dict] = field(default_factory=dict)
    probability_metrics: dict[str, dict] = field(default_factory=dict)
    best_severity_model: str = ""
    best_probability_model: str = ""
    feature_columns: list[str] = field(default_factory=list)
    n_rows_severity: int = 0
    n_rows_probability: int = 0


def _fit_with_search(model, param_grid, X_train, y_train, scoring, cv):
    search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    search.fit(X_train, y_train)
    logger.info("Best params: %s (score=%.4f)", search.best_params_, search.best_score_)
    return search.best_estimator_


def train_and_register(
    df: pd.DataFrame,
    *,
    reference_year: int,
    registry: ModelRegistry,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 3,
) -> TrainingResult:
    df = df.copy()
    for col in ["TotalClaims", "TotalPremium", "SumInsured", "RegistrationYear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[c for c in ["TotalClaims", "TotalPremium", "SumInsured", "RegistrationYear"] if c in df.columns])

    df["HasClaim"] = (df["TotalClaims"] > 0).astype(int)

    features = prepare_features(df, reference_year=reference_year)
    feature_columns = list(features.columns)

    has_claim = df.loc[features.index, "HasClaim"]
    total_claims = df.loc[features.index, "TotalClaims"]

    result = TrainingResult(feature_columns=feature_columns)
    version = new_version_string()

    # --- Claim severity (regression, claims-only subset) ---------------------------------
    claim_mask = has_claim == 1
    X_sev, y_sev = features[claim_mask], np.log1p(total_claims[claim_mask])
    result.n_rows_severity = len(X_sev)

    if len(X_sev) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(X_sev, y_sev, test_size=test_size, random_state=random_state)
        severity_models = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(random_state=random_state),
            "xgboost": XGBRegressor(random_state=random_state),
        }
        best_name, best_model, best_rmse = None, None, np.inf
        for name, model in severity_models.items():
            try:
                if name != "linear_regression":
                    model = _fit_with_search(model, HYPERPARAM_GRID, X_train, y_train, "neg_root_mean_squared_error", cv_folds)
                else:
                    model.fit(X_train, y_train)
                metrics = evaluate_regression(y_test, model.predict(X_test))
                result.severity_metrics[name] = metrics
                if metrics["rmse"] < best_rmse:
                    best_name, best_model, best_rmse = name, model, metrics["rmse"]
            except Exception:
                logger.exception("Severity model %s failed to train", name)

        if best_model is not None:
            result.best_severity_model = best_name
            registry.register(
                SEVERITY_MODEL_NAME,
                best_model,
                ModelMetadata(
                    model_name=SEVERITY_MODEL_NAME,
                    version=version,
                    created_at=version,
                    metrics=result.severity_metrics[best_name],
                    params={"algorithm": best_name, "target": "log1p(TotalClaims)"},
                    feature_columns=feature_columns,
                    notes="Trained on claims-only subset (TotalClaims > 0).",
                ),
            )

    # --- Claim probability (classification, full population) -----------------------------
    X_prob, y_prob = features, has_claim
    result.n_rows_probability = len(X_prob)
    X_train, X_test, y_train, y_test = train_test_split(
        X_prob, y_prob, test_size=test_size, random_state=random_state, stratify=y_prob
    )

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    probability_models = {
        "random_forest": RandomForestClassifier(random_state=random_state, class_weight="balanced"),
        "xgboost": XGBClassifier(random_state=random_state, eval_metric="logloss", scale_pos_weight=pos_weight),
    }

    best_name, best_model, best_f1 = None, None, -1.0
    for name, model in probability_models.items():
        try:
            model = _fit_with_search(model, HYPERPARAM_GRID, X_train, y_train, "f1", cv_folds)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_classification(y_test, y_pred, y_proba)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring="f1")
            metrics["cv_f1_mean"] = float(cv_scores.mean())
            metrics["cv_f1_std"] = float(cv_scores.std())
            result.probability_metrics[name] = metrics
            if metrics["f1"] > best_f1:
                best_name, best_model, best_f1 = name, model, metrics["f1"]
        except Exception:
            logger.exception("Probability model %s failed to train", name)

    if best_model is not None:
        result.best_probability_model = best_name
        registry.register(
            PROBABILITY_MODEL_NAME,
            best_model,
            ModelMetadata(
                model_name=PROBABILITY_MODEL_NAME,
                version=version,
                created_at=version,
                metrics=result.probability_metrics[best_name],
                params={"algorithm": best_name, "target": "HasClaim", "class_imbalance_strategy": "class_weight/scale_pos_weight"},
                feature_columns=feature_columns,
                notes="Full population; severe class imbalance -- see F1/precision, not accuracy.",
            ),
        )

    return result
