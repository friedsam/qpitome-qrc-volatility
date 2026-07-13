#!/usr/bin/env python3
"""Small exact-statevector temporal Rydberg reservoir for Task B.

This is the first usable Rydberg test on the frozen Task B protocol. It consumes
hardware-friendly, label-independent encodings and compares a protected
momentum offset against reservoir readout features.

The simulator uses a six-atom chain with global Rabi drive, input-modulated
detuning, and nearest-neighbor van der Waals interactions. Each encoded market
value is applied as one short analog segment. Features are final and temporal-
mean occupations plus nearest-neighbor pair occupations.

The implementation is intentionally compact and fixed-parameter. It is not a
hardware-calibrated Pulser emulation and does not support quantum-advantage
claims. Its role is to determine whether the Rydberg dynamics make Task B more
usable than the matched ESN control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

EPS = 1e-10


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def kron_all(operators: list[np.ndarray]) -> np.ndarray:
    result = operators[0]
    for operator in operators[1:]:
        result = np.kron(result, operator)
    return result


def local_operator(operator: np.ndarray, site: int, n_atoms: int) -> np.ndarray:
    identity = np.eye(2)
    return kron_all([operator if index == site else identity for index in range(n_atoms)])


def build_operators(n_atoms: int) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    n = np.array([[0.0, 0.0], [0.0, 1.0]])
    x_ops = [local_operator(x, site, n_atoms) for site in range(n_atoms)]
    n_ops = [local_operator(n, site, n_atoms) for site in range(n_atoms)]
    drive = sum(x_ops)
    number = sum(n_ops)
    pairs = [n_ops[index] @ n_ops[index + 1] for index in range(n_atoms - 1)]
    interaction = sum(pairs)
    return drive, number, n_ops, pairs, interaction


def precompute_unitaries(
    values: np.ndarray,
    drive: np.ndarray,
    number: np.ndarray,
    interaction: np.ndarray,
    omega: float,
    detuning_base: float,
    detuning_scale: float,
    interaction_strength: float,
    segment_time: float,
    quantization_levels: int,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    grid = np.linspace(-1.0, 1.0, quantization_levels)
    unique_indices = np.unique(np.argmin(np.abs(values[..., None] - grid), axis=-1))
    unitaries: dict[int, np.ndarray] = {}
    for index in unique_indices:
        input_value = float(grid[index])
        detuning = detuning_base + detuning_scale * input_value
        hamiltonian = (
            0.5 * omega * drive
            - detuning * number
            + interaction_strength * interaction
        )
        unitaries[int(index)] = expm(-1j * segment_time * hamiltonian)
    return unitaries, grid


def expectation(state: np.ndarray, operator: np.ndarray) -> float:
    return float(np.real(np.vdot(state, operator @ state)))


def reservoir_features(
    encoded: np.ndarray,
    n_ops: list[np.ndarray],
    pair_ops: list[np.ndarray],
    unitaries: dict[int, np.ndarray],
    grid: np.ndarray,
) -> np.ndarray:
    n_episodes, n_weeks, n_channels = encoded.shape
    dimension = n_ops[0].shape[0]
    initial = np.zeros(dimension, dtype=complex)
    initial[0] = 1.0
    rows = []

    for episode in range(n_episodes):
        state = initial.copy()
        snapshots = []
        for week in range(n_weeks):
            for channel in range(n_channels):
                value = encoded[episode, week, channel]
                index = int(np.argmin(np.abs(grid - value)))
                state = unitaries[index] @ state
                state /= np.linalg.norm(state)
            observables = [expectation(state, operator) for operator in n_ops]
            observables += [expectation(state, operator) for operator in pair_ops]
            snapshots.append(observables)
        snapshots = np.asarray(snapshots)
        rows.append(np.r_[snapshots[-1], snapshots.mean(axis=0)])
    return np.asarray(rows)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    penalty: float,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(design.shape[1])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    fixed = np.zeros(len(y)) if offset is None else offset
    for _ in range(100):
        probability = sigmoid(fixed + design @ beta)
        weight = np.clip(probability * (1.0 - probability), 1e-7, None)
        gradient = design.T @ (probability - y) + regularizer @ beta
        hessian = design.T @ (weight[:, None] * design) + regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient
        updated = beta - step
        if np.max(np.abs(updated - beta)) < 1e-8:
            beta = updated
            break
        beta = updated
    return beta


def standardized_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    penalty: float,
    train_offset: np.ndarray | None = None,
    test_offset: float | None = None,
) -> float:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    x_train = (train_x - mean) / scale
    x_test = (test_x - mean) / scale
    beta = fit_logistic(x_train, train_y, penalty, offset=train_offset)
    fixed = 0.0 if test_offset is None else test_offset
    return float(sigmoid(np.array([fixed + np.r_[1.0, x_test] @ beta]))[0])


def momentum_offset(train: pd.DataFrame, row: pd.Series, penalty: float) -> tuple[float, np.ndarray, float]:
    train_x = train[["momentum_13w_at_episode"]].to_numpy(float)
    test_x = row[["momentum_13w_at_episode"]].to_numpy(float)
    mean = train_x.mean(axis=0)
    scale = np.where(train_x.std(axis=0) < 1e-8, 1.0, train_x.std(axis=0))
    z_train = (train_x - mean) / scale
    z_test = (test_x - mean) / scale
    beta = fit_logistic(z_train, train["y_positive"].to_numpy(float), penalty)
    train_logit = np.column_stack([np.ones(len(z_train)), z_train]) @ beta
    test_logit = float(np.r_[1.0, z_test] @ beta)
    probability = float(sigmoid(np.array([test_logit]))[0])
    return probability, train_logit, test_logit


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    ordered = y[order].astype(int)
    if ordered.sum() == 0:
        return float("nan")
    precision = np.cumsum(ordered) / np.arange(1, len(ordered) + 1)
    return float(np.sum(precision * ordered) / ordered.sum())


def metric_row(model: str, part: pd.DataFrame) -> dict:
    y = part["y_true"].to_numpy(int)
    probability = np.clip(part["probability_positive"].to_numpy(float), EPS, 1.0 - EPS)
    prediction = probability >= 0.5
    positive_recall = float(np.mean(prediction[y == 1])) if np.any(y == 1) else np.nan
    negative_recall = float(np.mean(~prediction[y == 0])) if np.any(y == 0) else np.nan
    return {
        "model": model,
        "n_predictions": int(len(part)),
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


def evaluate_encoding(
    encoding_name: str,
    features: np.ndarray,
    metadata: pd.DataFrame,
    split_date: pd.Timestamp,
    minimum_train: int,
    readout_penalty: float,
    momentum_penalty: float,
) -> pd.DataFrame:
    rows = []
    for test_index, row in metadata[metadata["date"] >= split_date].iterrows():
        train_index = metadata.index[metadata["outcome_available_date"] < row["date"]].to_numpy(int)
        if len(train_index) < minimum_train:
            continue
        train = metadata.loc[train_index]
        y_train = train["y_positive"].to_numpy(float)
        p_momentum, train_offset, test_offset = momentum_offset(train, row, momentum_penalty)
        p_direct = standardized_predict(features[train_index], y_train, features[test_index], readout_penalty)
        p_offset = standardized_predict(
            features[train_index],
            y_train,
            features[test_index],
            readout_penalty,
            train_offset=train_offset,
            test_offset=test_offset,
        )
        for model, probability in [
            ("momentum_13w", p_momentum),
            (f"rydberg_{encoding_name}_direct", p_direct),
            (f"momentum_plus_rydberg_{encoding_name}", p_offset),
        ]:
            rows.append({
                "date": row["date"],
                "model": model,
                "encoding": encoding_name,
                "probability_positive": probability,
                "y_true": int(row["y_positive"]),
                "origin_regime": row["origin_regime"],
                "train_n": int(len(train_index)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputdir",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_rydberg_inputs"),
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        default=["return_only", "return_uncertainty"],
    )
    parser.add_argument("--split-date", default="1999-12-17")
    parser.add_argument("--minimum-train", type=int, default=60)
    parser.add_argument("--n-atoms", type=int, default=6)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--detuning-base", type=float, default=0.0)
    parser.add_argument("--detuning-scale", type=float, default=1.5)
    parser.add_argument("--interaction-strength", type=float, default=1.2)
    parser.add_argument("--segment-time", type=float, default=0.35)
    parser.add_argument("--quantization-levels", type=int, default=41)
    parser.add_argument("--readout-penalty", type=float, default=10.0)
    parser.add_argument("--momentum-penalty", type=float, default=1.0)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_rydberg_simulator"),
    )
    args = parser.parse_args()

    metadata = pd.read_csv(args.inputdir / "episode_metadata.csv")
    metadata["date"] = pd.to_datetime(metadata["date"])
    metadata["outcome_available_date"] = pd.to_datetime(metadata["outcome_available_date"])
    split_date = pd.Timestamp(args.split_date)

    drive, number, n_ops, pair_ops, interaction = build_operators(args.n_atoms)
    prediction_frames = []
    feature_frames = []

    for encoding_name in args.encodings:
        encoded = np.load(args.inputdir / f"{encoding_name}.npy")
        unitaries, grid = precompute_unitaries(
            encoded,
            drive,
            number,
            interaction,
            omega=args.omega,
            detuning_base=args.detuning_base,
            detuning_scale=args.detuning_scale,
            interaction_strength=args.interaction_strength,
            segment_time=args.segment_time,
            quantization_levels=args.quantization_levels,
        )
        features = reservoir_features(encoded, n_ops, pair_ops, unitaries, grid)
        feature_frames.append(pd.DataFrame(features).assign(encoding=encoding_name, episode_index=np.arange(len(features))))
        prediction_frames.append(
            evaluate_encoding(
                encoding_name,
                features,
                metadata,
                split_date=split_date,
                minimum_train=args.minimum_train,
                readout_penalty=args.readout_penalty,
                momentum_penalty=args.momentum_penalty,
            )
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_table = pd.concat(feature_frames, ignore_index=True)
    metrics = pd.DataFrame([
        metric_row(model, part)
        for model, part in predictions.groupby("model")
    ]).sort_values("log_loss")

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / "predictions.csv", index=False)
    metrics.to_csv(args.outdir / "metrics.csv", index=False)
    feature_table.to_csv(args.outdir / "reservoir_features.csv", index=False)
    pd.DataFrame([{
        "n_atoms": args.n_atoms,
        "hilbert_dimension": 2 ** args.n_atoms,
        "omega": args.omega,
        "detuning_base": args.detuning_base,
        "detuning_scale": args.detuning_scale,
        "interaction_strength": args.interaction_strength,
        "segment_time": args.segment_time,
        "quantization_levels": args.quantization_levels,
        "readout_penalty": args.readout_penalty,
        "encodings": ",".join(args.encodings),
        "feature_definition": "final and temporal-mean occupations plus nearest-neighbor pair occupations",
        "evaluation": "prequential; outcome-maturity aware; protected momentum offset",
        "claim_limit": "exact simulator only; not hardware calibrated; no quantum advantage claim",
    }]).to_csv(args.outdir / "manifest.csv", index=False)

    print("Task B exact-statevector Rydberg reservoir metrics")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPromotion rule")
    print("Promote only if the protected-offset Rydberg model improves log loss or Brier score and does not reduce deterioration recall relative to momentum.")


if __name__ == "__main__":
    main()
