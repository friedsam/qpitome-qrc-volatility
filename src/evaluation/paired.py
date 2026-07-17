"""Paired probability-loss diagnostics for protected-baseline comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-10


def per_episode_log_loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=int)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def paired_bootstrap(values: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    samples = np.array([values[rng.integers(0, len(values), size=len(values))].mean() for _ in range(n_bootstrap)])
    return {
        "mean": float(values.mean()),
        "bootstrap_q025": float(np.quantile(samples, 0.025)),
        "bootstrap_q50": float(np.quantile(samples, 0.50)),
        "bootstrap_q975": float(np.quantile(samples, 0.975)),
        "bootstrap_fraction_below_zero": float(np.mean(samples < 0.0)),
    }


def compare_predictions(frame: pd.DataFrame, baseline_model: str, candidate_model: str, n_bootstrap: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = frame[frame["model"] == baseline_model].drop_duplicates("date")[["date", "y_true", "origin_regime", "probability_positive"]].rename(columns={"probability_positive": "p_baseline"})
    candidate = frame[frame["model"] == candidate_model][["date", "probability_positive"]].rename(columns={"probability_positive": "p_candidate"})
    paired = baseline.merge(candidate, on="date", validate="one_to_one")
    y = paired["y_true"].to_numpy(int)
    p0 = paired["p_baseline"].to_numpy(float)
    p1 = paired["p_candidate"].to_numpy(float)
    paired["baseline_log_loss"] = per_episode_log_loss(y, p0)
    paired["candidate_log_loss"] = per_episode_log_loss(y, p1)
    paired["candidate_minus_baseline_log_loss"] = paired["candidate_log_loss"] - paired["baseline_log_loss"]
    paired["probability_shift"] = p1 - p0
    rows = []
    for subgroup, part in (("all", paired), ("positive", paired[paired["y_true"] == 1]), ("negative", paired[paired["y_true"] == 0])):
        delta = part["candidate_minus_baseline_log_loss"].to_numpy(float)
        rows.append({"subgroup": subgroup, "n": len(part), "mean_probability_shift": float(part["probability_shift"].mean()), "median_probability_shift": float(part["probability_shift"].median()), "episodes_with_lower_candidate_loss": int((delta < 0).sum()), "episodes_with_higher_candidate_loss": int((delta > 0).sum()), **paired_bootstrap(delta, n_bootstrap, seed)})
    threshold_rows = []
    for threshold in np.arange(0.35, 0.651, 0.025):
        for model, probability in ((baseline_model, p0), (candidate_model, p1)):
            prediction = probability >= threshold
            positive_recall = float(np.mean(prediction[y == 1]))
            negative_recall = float(np.mean(~prediction[y == 0]))
            threshold_rows.append({"threshold": float(round(threshold, 3)), "model": model, "positive_recall": positive_recall, "negative_recall": negative_recall, "balanced_accuracy": float((positive_recall + negative_recall) / 2), "accuracy": float(np.mean(prediction == y))})
    correction = pd.DataFrame([{"class": label, "n": len(part), "mean_baseline_probability": float(part["p_baseline"].mean()), "mean_candidate_probability": float(part["p_candidate"].mean()), "mean_probability_shift": float(part["probability_shift"].mean())} for label, part in (("positive", paired[paired["y_true"] == 1]), ("negative", paired[paired["y_true"] == 0]))])
    return paired, pd.DataFrame(rows), pd.DataFrame(threshold_rows), correction
