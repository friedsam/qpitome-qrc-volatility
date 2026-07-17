"""Outcome-maturity-aware prequential evaluation for destination forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.destination_logistic import fit_predict_probability, fit_ridge_logistic, sigmoid, standardize_train_test
from evaluation.binary import grouped_binary_metrics

MODEL_FEATURES = {
    "historical_prior": [],
    "momentum_13w": ["momentum_13w"],
    "compact_directional": ["momentum_13w", "return_1w", "vol_13w", "bull_probability"],
    "compact_transition": ["momentum_13w", "return_1w", "vol_13w", "bull_probability", "long_regime_uncertainty", "one_week_probability_motion"],
}


def predict_logistic_row(train: pd.DataFrame, row: pd.Series, features: list[str], penalty: float) -> tuple[float, dict[str, float]]:
    if not features:
        return float(train["y_positive"].mean()), {}
    train_x = train[features].to_numpy(float)
    test_x = row[features].to_numpy(float)
    standardized_train, standardized_test, _, _ = standardize_train_test(train_x, test_x)
    beta = fit_ridge_logistic(standardized_train, train["y_positive"].to_numpy(float), penalty=penalty)
    probability = float(sigmoid(np.array([np.r_[1.0, standardized_test] @ beta]))[0])
    coefficients = {"intercept": float(beta[0]), **{feature: float(value) for feature, value in zip(features, beta[1:])}}
    return probability, coefficients


def evaluate_classical(frame: pd.DataFrame, split_date: pd.Timestamp, penalty: float, minimum_train: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows, coefficient_rows = [], []
    for _, row in frame[frame["date"] >= split_date].iterrows():
        train = frame[frame["outcome_available_date"] < row["date"]]
        if len(train) < minimum_train or train["y_positive"].nunique() < 2:
            continue
        for model, features in MODEL_FEATURES.items():
            probability, coefficients = predict_logistic_row(train, row, features, penalty)
            prediction_rows.append({
                "date": row["date"], "model": model, "probability_positive": probability,
                "y_true": int(row["y_positive"]), "future_return_pct": float(row["future_return_pct"]),
                "origin_regime": row["origin_regime"], "train_n": len(train),
                "train_positive_fraction": float(train["y_positive"].mean()),
                "latest_training_outcome_date": train["outcome_available_date"].max(),
            })
            if coefficients:
                coefficient_rows.append({"date": row["date"], "model": model, "train_n": len(train), **coefficients})
    return pd.DataFrame(prediction_rows), pd.DataFrame(coefficient_rows)


def momentum_offset(train: pd.DataFrame, row: pd.Series, penalty: float = 1.0) -> tuple[float, np.ndarray, float]:
    train_x = train[["momentum_13w_at_episode"]].to_numpy(float)
    test_x = row[["momentum_13w_at_episode"]].to_numpy(float)
    standardized_train, standardized_test, _, _ = standardize_train_test(train_x, test_x)
    beta = fit_ridge_logistic(standardized_train, train["y_positive"].to_numpy(float), penalty=penalty)
    train_logit = np.column_stack([np.ones(len(standardized_train)), standardized_train]) @ beta
    test_logit = float(np.r_[1.0, standardized_test] @ beta)
    return float(sigmoid(np.array([test_logit]))[0]), train_logit, test_logit


def evaluate_reservoir_features(
    features_by_name: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    split_date: pd.Timestamp,
    minimum_train: int,
    readout_penalty: float,
    momentum_penalty: float,
    model_prefix: str,
) -> pd.DataFrame:
    rows = []
    for name, features in features_by_name.items():
        if len(features) != len(metadata):
            raise ValueError(f"Feature/metadata length mismatch for {name}")
        for test_index, row in metadata[metadata["date"] >= split_date].iterrows():
            train_index = metadata.index[metadata["outcome_available_date"] < row["date"]].to_numpy(int)
            if len(train_index) < minimum_train:
                continue
            train = metadata.loc[train_index]
            y_train = train["y_positive"].to_numpy(float)
            p_momentum, train_offset, test_offset = momentum_offset(train, row, momentum_penalty)
            p_direct = fit_predict_probability(features[train_index], y_train, features[test_index], penalty=readout_penalty)
            p_offset = fit_predict_probability(features[train_index], y_train, features[test_index], penalty=readout_penalty, train_offset=train_offset, test_offset=test_offset)
            for model, probability in (
                ("momentum_13w", p_momentum),
                (f"{model_prefix}_{name}_direct", p_direct),
                (f"momentum_plus_{model_prefix}_{name}", p_offset),
            ):
                rows.append({"date": row["date"], "model": model, "encoding": name, "probability_positive": probability, "y_true": int(row["y_positive"]), "origin_regime": row["origin_regime"], "train_n": len(train_index)})
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame, subgroup: bool = False) -> pd.DataFrame:
    return grouped_binary_metrics(predictions, subgroup_col="origin_regime" if subgroup else None, minimum_subgroup_size=8)
