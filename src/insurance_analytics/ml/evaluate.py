"""Model evaluation metrics, isolated so both training and post-hoc audits can reuse them."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_regression(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)))),
    }


def evaluate_classification(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    return metrics


def shap_feature_importance(model, X: pd.DataFrame, max_features: int = 10) -> pd.DataFrame:
    """Mean |SHAP value| per feature, for model interpretability / error analysis."""
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = pd.DataFrame(
        {"feature": X.columns, "importance": np.abs(shap_values).mean(axis=0)}
    ).sort_values("importance", ascending=False)
    return importance.head(max_features).reset_index(drop=True)
