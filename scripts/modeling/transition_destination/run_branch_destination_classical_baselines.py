#!/usr/bin/env python3
"""Compact leakage-safe classical baselines for Task B.

The target is the provisional 8-week positive-versus-negative destination after
an HMM-uncertainty episode, excluding outcomes within +/-1%. Evaluation is
prequential on the post-1999 holdout: before predicting an episode, training uses
only episodes whose future outcome window has already completed.

This is intentionally a small baseline ladder, not a hyperparameter search.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-10
MODEL_FEATURES = {
    "historical_prior": [],
    "momentum_13w": ["momentum_13w"],
    "compact_directional": [
        "momentum_13w",
        "return_1w",
        "vol_13w",
        "bull_probability",
    ],
    "compact_transition": [
        "momentum_13w",
        "return_1w",
        "vol_13w",
        "bull_probability",
        "long_regime_uncertainty",
        "one_week_probability_motion",
    ],
}


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float,
    max_iter: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Newton solver with an unpenalized intercept and fixed L2 penalty."""
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1], dtype=float)
    prior = np.clip(float(np.mean(y)), 1e-5, 1.0 - 1e-5)
    beta[0] = np.log(prior / (1.0 - prior))
    penalty_matrix = np.eye(design.shape[1]) * penalty
    penalty_matrix[0, 0] = 0.0

    for _ in range(max_iter):
        probability = sigmoid(design @ beta)
        weight = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = design.T @ (probability - y) + penalty_matrix @ beta
        hessian = design.T @ (weight[:, None] * design) + penalty_matrix
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta - step
        if float(np.max(np.abs(beta_new - beta))) < tolerance:
            beta = beta_new
            break
        beta = beta_new
    return beta


def predict_ridge_logistic(
    train: pd.DataFrame,
    row: pd.Series,
    features: list[str],
    penalty: float,
) -> tuple[float, dict[str, float]]:
    if not features:
        probability = float(train["y_positive"].mean())
        return probability, {}

    x_train = train[features].to_numpy(float)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_scaled = (x_train - mean) / scale
    x_row = (row[features].to_numpy(float) - mean) / scale
    beta = fit_ridge_logistic(
        x_scaled,
        train["y_positive"].to_numpy(float),
        penalty=penalty,
    )
    probability = float(sigmoid(np.array([np.r_[1.0, x_row] @ beta]))[0])
    coefficients = {"intercept": float(beta[0])}
    coefficients.update({feature: float(value) for feature, value in zip(features, beta[1:])})
    return probability, coefficients


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    rank_sum = float(ranks[y == 1].sum())
    return float((rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    ordered = y[order].astype(int)
    n_positive = int(ordered.sum())
    if n_positive == 0:
        return float("nan")
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / n_positive)


def metric_row(model: str, frame: pd.DataFrame) -> dict[str, float | int | str]:
    y = frame["y_true"].to_numpy(int)
    probability = np.clip(frame["probability_positive"].to_numpy(float), EPS, 1.0 - EPS)
    prediction = probability >= 0.5
    positive_recall = float(np.mean(prediction[y == 1])) if np.any(y == 1) else np.nan
    negative_recall = float(np.mean(~prediction[y == 0])) if np.any(y == 0) else np.nan
    return {
        "model": model,
        "n_predictions": int(len(frame)),
        "n_positive": int(y.sum()),
        "n_negative": int(len(y) - y.sum()),
        "positive_prevalence": float(y.mean()),
        "roc_auc": roc_auc(y, probability),
        "average_precision": average_precision(y, probability),
        "log_loss": float(-np.mean(y * np.log(probability) + (1 - y) * np.log(1 - probability))),
        "brier": float(np.mean((y - probability) ** 2)),
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": float((positive_recall + negative_recall) / 2),
        "positive_recall": positive_recall,
        "negative_recall": negative_recall,
        "mean_probability": float(probability.mean()),
    }


def prepare(
    path: Path,
    horizon: int,
    neutral_zone: float,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"])
    target = f"future_return_{horizon}w"
    if target not in frame:
        raise ValueError(f"Missing target column {target!r}")
    frame = frame[frame[target].notna()].copy()
    frame = frame[np.abs(frame[target]) > neutral_zone].copy()
    frame["y_positive"] = (frame[target] > neutral_zone).astype(int)
    frame["outcome_available_date"] = frame["date"] + pd.to_timedelta(horizon, unit="W")
    frame["origin_regime"] = np.where(frame["bull_probability"] >= 0.5, "bull_side", "bear_side")
    return frame.sort_values("date").reset_index(drop=True)


def evaluate(
    frame: pd.DataFrame,
    split_date: pd.Timestamp,
    penalty: float,
    minimum_train: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = frame[frame["date"] >= split_date].copy()
    prediction_rows: list[dict] = []
    coefficient_rows: list[dict] = []

    for _, row in test.iterrows():
        train = frame[frame["outcome_available_date"] < row["date"]].copy()
        if len(train) < minimum_train or train["y_positive"].nunique() < 2:
            continue
        for model, features in MODEL_FEATURES.items():
            probability, coefficients = predict_ridge_logistic(
                train,
                row,
                features=features,
                penalty=penalty,
            )
            prediction_rows.append({
                "date": row["date"],
                "model": model,
                "probability_positive": probability,
                "y_true": int(row["y_positive"]),
                "future_return_pct": float(row.filter(like="future_return_").iloc[0]),
                "origin_regime": row["origin_regime"],
                "train_n": int(len(train)),
                "train_positive_fraction": float(train["y_positive"].mean()),
                "latest_training_outcome_date": train["outcome_available_date"].max(),
            })
            if coefficients:
                coefficient_rows.append({
                    "date": row["date"],
                    "model": model,
                    "train_n": int(len(train)),
                    **coefficients,
                })

    return pd.DataFrame(prediction_rows), pd.DataFrame(coefficient_rows)


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, model_frame in predictions.groupby("model"):
        rows.append({"subgroup": "all", **metric_row(model, model_frame)})
        for origin, part in model_frame.groupby("origin_regime"):
            if len(part) >= 8 and part["y_true"].nunique() == 2:
                rows.append({"subgroup": origin, **metric_row(model, part)})
    return pd.DataFrame(rows).sort_values(["subgroup", "log_loss"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_target_audit/episodes.csv"),
    )
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--neutral-zone", type=float, default=1.0)
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--minimum-train", type=int, default=60)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_classical_baselines"),
    )
    args = parser.parse_args()

    frame = prepare(args.episodes, args.horizon, args.neutral_zone)
    predictions, coefficients = evaluate(
        frame,
        split_date=pd.Timestamp(args.split_date),
        penalty=args.penalty,
        minimum_train=args.minimum_train,
    )
    if predictions.empty:
        raise RuntimeError("No predictions produced")
    metrics = subgroup_metrics(predictions)
    manifest = pd.DataFrame([{
        "horizon_weeks": args.horizon,
        "neutral_zone_pct": args.neutral_zone,
        "split_date": args.split_date,
        "ridge_penalty": args.penalty,
        "minimum_train": args.minimum_train,
        "n_binary_episodes": len(frame),
        "n_holdout_episodes": int((frame["date"] >= pd.Timestamp(args.split_date)).sum()),
        "evaluation": "prequential; train only on outcomes completed before prediction date",
    }])

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.outdir / "manifest.csv", index=False)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    coefficients.to_csv(args.outdir / "coefficient_history.csv", index=False)
    metrics.to_csv(args.outdir / "metrics.csv", index=False)

    print("Task B compact classical baseline manifest")
    print(manifest.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPrequential holdout metrics")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPrimary safeguards")
    print("- ordinary accuracy is not a selection metric")
    print("- negative_recall is deterioration recall")
    print("- historical_prior is the mandatory class-drift baseline")
    print("- no episode is trained on before its future outcome window has completed")


if __name__ == "__main__":
    main()
