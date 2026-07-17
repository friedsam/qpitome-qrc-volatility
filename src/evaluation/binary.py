"""Probability-aware binary classification metrics shared across experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


EPS = 1e-10


def binary_metric_row(
    model: str,
    y_true: np.ndarray,
    probability_positive: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability_positive, dtype=float), EPS, 1.0 - EPS)
    if y.ndim != 1 or probability.ndim != 1 or len(y) != len(probability):
        raise ValueError("y_true and probability_positive must be same-length 1D arrays")
    if len(y) == 0:
        raise ValueError("at least one observation is required")

    prediction = probability >= threshold
    has_positive = np.any(y == 1)
    has_negative = np.any(y == 0)
    positive_recall = float(np.mean(prediction[y == 1])) if has_positive else np.nan
    negative_recall = float(np.mean(~prediction[y == 0])) if has_negative else np.nan
    balanced_accuracy = (
        float((positive_recall + negative_recall) / 2)
        if has_positive and has_negative
        else np.nan
    )

    return {
        "model": model,
        "n_predictions": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "positive_prevalence": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probability)) if has_positive and has_negative else np.nan,
        "average_precision": float(average_precision_score(y, probability)) if has_positive else np.nan,
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
        "brier": float(np.mean((y - probability) ** 2)),
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": balanced_accuracy,
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "mean_probability": float(probability.mean()),
        "threshold": float(threshold),
    }


def grouped_binary_metrics(
    predictions: pd.DataFrame,
    *,
    model_col: str = "model",
    target_col: str = "y_true",
    probability_col: str = "probability_positive",
    subgroup_col: str | None = None,
    minimum_subgroup_size: int = 1,
    threshold: float = 0.5,
) -> pd.DataFrame:
    required = {model_col, target_col, probability_col}
    if subgroup_col is not None:
        required.add(subgroup_col)
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")

    rows: list[dict[str, Any]] = []
    for model, model_frame in predictions.groupby(model_col, sort=True):
        rows.append({
            "subgroup": "all",
            **binary_metric_row(
                str(model),
                model_frame[target_col].to_numpy(int),
                model_frame[probability_col].to_numpy(float),
                threshold=threshold,
            ),
        })
        if subgroup_col is None:
            continue
        for subgroup, part in model_frame.groupby(subgroup_col, sort=True):
            if len(part) < minimum_subgroup_size:
                continue
            rows.append({
                "subgroup": str(subgroup),
                **binary_metric_row(
                    str(model),
                    part[target_col].to_numpy(int),
                    part[probability_col].to_numpy(float),
                    threshold=threshold,
                ),
            })
    return pd.DataFrame(rows).sort_values(["subgroup", "log_loss", "model"]).reset_index(drop=True)
