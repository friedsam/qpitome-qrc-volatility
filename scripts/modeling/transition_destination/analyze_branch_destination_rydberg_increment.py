#!/usr/bin/env python3
"""Paired diagnostic for the first Task B Rydberg result.

Compares the protected momentum baseline with the return+uncertainty Rydberg
correction on the exact same 48 holdout episodes. This is diagnostic only: no
model parameters or decision threshold are tuned from holdout outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-10
SEED = 20260712


def per_episode_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(values: np.ndarray, n_bootstrap: int) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    n = len(values)
    samples = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = rng.integers(0, n, size=n)
        samples[index] = float(values[draw].mean())
    return {
        "mean": float(values.mean()),
        "bootstrap_q025": float(np.quantile(samples, 0.025)),
        "bootstrap_q50": float(np.quantile(samples, 0.50)),
        "bootstrap_q975": float(np.quantile(samples, 0.975)),
        "bootstrap_fraction_below_zero": float(np.mean(samples < 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_rydberg_simulator/predictions.csv"),
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/modeling/transition_destination/branch_destination_rydberg_increment"),
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    frame["date"] = pd.to_datetime(frame["date"])
    baseline = (
        frame[frame["model"] == "momentum_13w"]
        .drop_duplicates("date", keep="first")
        [["date", "y_true", "origin_regime", "probability_positive"]]
        .rename(columns={"probability_positive": "p_momentum"})
    )
    rydberg = (
        frame[frame["model"] == "momentum_plus_rydberg_return_uncertainty"]
        [["date", "probability_positive"]]
        .rename(columns={"probability_positive": "p_rydberg"})
    )
    paired = baseline.merge(rydberg, on="date", how="inner", validate="one_to_one")
    if len(paired) != 48:
        raise ValueError(f"Expected 48 paired holdout episodes, found {len(paired)}")

    y = paired["y_true"].to_numpy(int)
    p0 = paired["p_momentum"].to_numpy(float)
    p1 = paired["p_rydberg"].to_numpy(float)
    paired["momentum_log_loss"] = per_episode_loss(y, p0)
    paired["rydberg_log_loss"] = per_episode_loss(y, p1)
    paired["rydberg_minus_momentum_log_loss"] = paired["rydberg_log_loss"] - paired["momentum_log_loss"]
    paired["probability_shift"] = p1 - p0
    paired["momentum_correct_0_5"] = ((p0 >= 0.5).astype(int) == y)
    paired["rydberg_correct_0_5"] = ((p1 >= 0.5).astype(int) == y)

    rows = []
    for subgroup, part in [("all", paired), ("positive", paired[paired["y_true"] == 1]), ("negative", paired[paired["y_true"] == 0])]:
        delta = part["rydberg_minus_momentum_log_loss"].to_numpy(float)
        bootstrap = paired_bootstrap(delta, args.n_bootstrap)
        rows.append({
            "subgroup": subgroup,
            "n": len(part),
            "mean_probability_shift": float(part["probability_shift"].mean()),
            "median_probability_shift": float(part["probability_shift"].median()),
            "episodes_with_lower_rydberg_loss": int((delta < 0).sum()),
            "episodes_with_higher_rydberg_loss": int((delta > 0).sum()),
            **bootstrap,
        })
    paired_summary = pd.DataFrame(rows)

    threshold_rows = []
    for threshold in np.arange(0.35, 0.651, 0.025):
        for model, probability in [("momentum", p0), ("rydberg_uncertainty", p1)]:
            prediction = probability >= threshold
            positive_recall = float(np.mean(prediction[y == 1]))
            negative_recall = float(np.mean(~prediction[y == 0]))
            threshold_rows.append({
                "threshold": float(round(threshold, 3)),
                "model": model,
                "positive_recall": positive_recall,
                "negative_recall": negative_recall,
                "balanced_accuracy": float((positive_recall + negative_recall) / 2),
                "accuracy": float(np.mean(prediction == y)),
            })
    threshold_table = pd.DataFrame(threshold_rows)

    correction_rows = []
    for label, part in [("positive", paired[paired["y_true"] == 1]), ("negative", paired[paired["y_true"] == 0])]:
        correction_rows.append({
            "class": label,
            "n": len(part),
            "mean_momentum_probability": float(part["p_momentum"].mean()),
            "mean_rydberg_probability": float(part["p_rydberg"].mean()),
            "mean_probability_shift": float(part["probability_shift"].mean()),
            "fraction_shifted_down": float((part["probability_shift"] < 0).mean()),
            "fraction_shifted_up": float((part["probability_shift"] > 0).mean()),
        })
    correction_summary = pd.DataFrame(correction_rows)

    args.outdir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.outdir / "paired_episode_comparison.csv", index=False)
    paired_summary.to_csv(args.outdir / "paired_loss_summary.csv", index=False)
    threshold_table.to_csv(args.outdir / "threshold_tradeoff.csv", index=False)
    correction_summary.to_csv(args.outdir / "class_probability_shift.csv", index=False)

    print("Paired Rydberg incremental-loss diagnostic")
    print(paired_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nClass-conditional probability shifts")
    print(correction_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nThreshold tradeoff around 0.5")
    print(threshold_table[threshold_table["threshold"].between(0.45, 0.55)].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nDecision rule")
    print("Treat this as an interesting ranking signal, not a promoted model, unless paired loss is stable enough to justify a causally selected shrinkage/calibration step.")


if __name__ == "__main__":
    main()
