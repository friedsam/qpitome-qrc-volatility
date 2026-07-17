#!/usr/bin/env python3
"""Matched classical ESN baseline for Task B.

Uses the exact causal episode paths prepared for the later Rydberg reservoir.
Evaluation is prequential and outcome-maturity aware. The primary comparison is
an ESN correction on top of the frozen 13-week-momentum baseline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-10
SEED = 20260712


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def make_reservoir(n_inputs: int, n_units: int, spectral_radius: float, input_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    w_in = rng.normal(0.0, input_scale, size=(n_units, n_inputs))
    w = rng.normal(0.0, 1.0 / np.sqrt(n_units), size=(n_units, n_units))
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    w *= spectral_radius / max(radius, 1e-8)
    bias = rng.normal(0.0, 0.1, size=n_units)
    return w_in, w, bias


def reservoir_features(paths: np.ndarray, w_in: np.ndarray, w: np.ndarray, bias: np.ndarray, leak: float) -> np.ndarray:
    rows = []
    for path in paths:
        state = np.zeros(w.shape[0], dtype=float)
        states = []
        for u in path:
            proposal = np.tanh(w_in @ u + w @ state + bias)
            state = (1.0 - leak) * state + leak * proposal
            states.append(state.copy())
        states = np.asarray(states)
        rows.append(np.r_[states[-1], states.mean(axis=0)])
    return np.asarray(rows)


def fit_logistic(x: np.ndarray, y: np.ndarray, penalty: float, offset: np.ndarray | None = None) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1])
    reg = np.eye(design.shape[1]) * penalty
    reg[0, 0] = 0.0
    fixed = np.zeros(len(y)) if offset is None else offset
    for _ in range(100):
        p = sigmoid(fixed + design @ beta)
        weight = np.clip(p * (1.0 - p), 1e-7, None)
        gradient = design.T @ (p - y) + reg @ beta
        hessian = design.T @ (weight[:, None] * design) + reg
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new
    return beta


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, penalty: float, train_offset: np.ndarray | None = None, test_offset: float | None = None) -> float:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    beta = fit_logistic(x_train, train_y, penalty, offset=train_offset)
    fixed = 0.0 if test_offset is None else test_offset
    return float(sigmoid(np.array([fixed + np.r_[1.0, x_test] @ beta]))[0])


def momentum_probability(train: pd.DataFrame, row: pd.Series, penalty: float = 1.0) -> tuple[float, np.ndarray, float]:
    x_train = train[["momentum_13w_at_episode"]].to_numpy(float)
    x_test = row[["momentum_13w_at_episode"]].to_numpy(float)
    mean = x_train.mean(axis=0)
    scale = np.where(x_train.std(axis=0) < 1e-8, 1.0, x_train.std(axis=0))
    z_train = (x_train - mean) / scale
    z_test = (x_test - mean) / scale
    beta = fit_logistic(z_train, train["y_positive"].to_numpy(float), penalty)
    train_logit = np.column_stack([np.ones(len(z_train)), z_train]) @ beta
    test_logit = float(np.r_[1.0, z_test] @ beta)
    return float(sigmoid(np.array([test_logit]))[0]), train_logit, test_logit


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    yy = y[order].astype(int)
    if yy.sum() == 0:
        return float("nan")
    precision = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float(np.sum(precision * yy) / yy.sum())


def metrics(model: str, part: pd.DataFrame) -> dict:
    y = part["y_true"].to_numpy(int)
    p = np.clip(part["probability_positive"].to_numpy(float), EPS, 1.0 - EPS)
    pred = p >= 0.5
    pos_recall = float(np.mean(pred[y == 1])) if np.any(y == 1) else np.nan
    neg_recall = float(np.mean(~pred[y == 0])) if np.any(y == 0) else np.nan
    return {
        "model": model,
        "n_predictions": len(part),
        "roc_auc": roc_auc(y, p),
        "average_precision": average_precision(y, p),
        "log_loss": float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p))),
        "brier": float(np.mean((y-p)**2)),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float((pos_recall + neg_recall) / 2),
        "positive_recall": pos_recall,
        "negative_recall": neg_recall,
        "mean_probability": float(p.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pathdir",
        type=Path,
        default=Path("results/modeling/transition_destination"),
    )
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--n-units", type=int, default=24)
    parser.add_argument("--spectral-radius", type=float, default=0.9)
    parser.add_argument("--input-scale", type=float, default=0.35)
    parser.add_argument("--leak", type=float, default=0.5)
    parser.add_argument("--penalty", type=float, default=10.0)
    parser.add_argument("--minimum-train", type=int, default=60)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_destination"),
    )
    args = parser.parse_args()

    paths = np.load(args.pathdir / "branch_destination_paths__paths.npy")
    metadata = pd.read_csv(args.pathdir / "branch_destination_paths__episode_metadata.csv")
    metadata["date"] = pd.to_datetime(metadata["date"])
    metadata["outcome_available_date"] = pd.to_datetime(metadata["outcome_available_date"])
    if len(paths) != len(metadata):
        raise ValueError("paths.npy and episode_metadata.csv length mismatch")

    w_in, w, bias = make_reservoir(paths.shape[2], args.n_units, args.spectral_radius, args.input_scale)
    features = reservoir_features(paths, w_in, w, bias, args.leak)
    split_date = pd.Timestamp(args.split_date)
    rows = []

    for test_idx, row in metadata[metadata["date"] >= split_date].iterrows():
        train_idx = metadata.index[metadata["outcome_available_date"] < row["date"]].to_numpy(int)
        if len(train_idx) < args.minimum_train:
            continue
        train = metadata.loc[train_idx]
        y_train = train["y_positive"].to_numpy(float)

        p_momentum, train_offset, test_offset = momentum_probability(train, row)
        p_esn = fit_predict(features[train_idx], y_train, features[test_idx], args.penalty)
        p_offset = fit_predict(
            features[train_idx], y_train, features[test_idx], args.penalty,
            train_offset=train_offset, test_offset=test_offset,
        )
        for model, probability in [
            ("momentum_13w", p_momentum),
            ("esn_direct", p_esn),
            ("momentum_plus_esn_offset", p_offset),
        ]:
            rows.append({
                "date": row["date"],
                "model": model,
                "probability_positive": probability,
                "y_true": int(row["y_positive"]),
                "origin_regime": row["origin_regime"],
                "train_n": len(train_idx),
            })

    predictions = pd.DataFrame(rows)
    summary = pd.DataFrame([
        metrics(model, part) for model, part in predictions.groupby("model")
    ]).sort_values("log_loss")

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        args.outdir / "branch_destination_esn_baseline__predictions.csv",
        index=False,
    )
    summary.to_csv(
        args.outdir / "branch_destination_esn_baseline__metrics.csv",
        index=False,
    )
    pd.DataFrame([{
        "n_units": args.n_units,
        "feature_dimension": 2 * args.n_units,
        "spectral_radius": args.spectral_radius,
        "input_scale": args.input_scale,
        "leak": args.leak,
        "ridge_penalty": args.penalty,
        "seed": SEED,
        "evaluation": "prequential; outcome-maturity aware; same paths reserved for Rydberg",
    }]).to_csv(
        args.outdir / "branch_destination_esn_baseline__manifest.csv",
        index=False,
    )

    print("Task B matched classical reservoir metrics")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nInterpretation rule")
    print("Keep ESN only if the protected-offset model improves probability quality or deterioration recall without collapsing positive recall.")


if __name__ == "__main__":
    main()
