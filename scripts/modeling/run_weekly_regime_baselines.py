#!/usr/bin/env python3
"""Evaluate simple weekly return regime baselines on one market series.

Models:
- iid Gaussian
- two-state Gaussian HMM
- unrestricted four-state Gaussian HMM
- Maheu-restricted four-state Gaussian HMM

The evaluation is causal. Parameters are estimated on an initial history and
optionally re-estimated at a fixed cadence. Between refits, filtered state
probabilities are updated one observation at a time and one-step predictive
log densities are recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.baselines.gaussian_hmm import (
    fit_gaussian_hmm,
    forward_filter,
    maheu_restricted_mask,
    one_step_predictive_density,
)

EPS = 1e-12
SEED = 20260711


def load_weekly_returns(
    path: Path,
    date_col: str,
    value_col: str,
    input_kind: str,
    market_col: str | None,
    market: str | None,
) -> pd.Series:
    frame = pd.read_csv(path)
    if date_col not in frame or value_col not in frame:
        raise ValueError(f"CSV must contain {date_col!r} and {value_col!r}")
    if market_col is not None:
        if market_col not in frame:
            raise ValueError(f"CSV does not contain market column {market_col!r}")
        if market is None:
            raise ValueError("--market is required when --market-col is used")
        frame = frame[frame[market_col].astype(str) == str(market)].copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    frame = frame.sort_values(date_col).drop_duplicates(date_col, keep="last")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    series = pd.Series(values.to_numpy(float), index=frame[date_col])
    series = series.dropna()

    if input_kind == "price":
        weekly_price = series.resample("W-FRI").last().dropna()
        weekly = 100.0 * np.log(weekly_price).diff()
    elif input_kind == "daily_log_return":
        weekly = series.resample("W-FRI").sum(min_count=1)
    elif input_kind == "daily_simple_return":
        weekly = (1.0 + series / 100.0).resample("W-FRI").prod(min_count=1)
        weekly = 100.0 * (weekly - 1.0)
    elif input_kind == "weekly_return":
        weekly = series
    else:
        raise ValueError(f"Unknown input kind: {input_kind}")
    weekly = weekly.replace([np.inf, -np.inf], np.nan).dropna()
    weekly.name = "return_pct"
    return weekly


def iid_gaussian_log_density(train: np.ndarray, value: float) -> float:
    mean = float(np.mean(train))
    variance = max(float(np.var(train)), 1e-4)
    return float(
        -0.5 * np.log(2.0 * np.pi * variance)
        -0.5 * (value - mean) ** 2 / variance
    )


def model_specs() -> dict[str, dict]:
    return {
        "hmm2": {
            "n_states": 2,
            "mask": np.ones((2, 2), dtype=bool),
            "mean_signs": np.array([-1, 1]),
        },
        "hmm4_unrestricted": {
            "n_states": 4,
            "mask": np.ones((4, 4), dtype=bool),
            "mean_signs": np.array([-1, 1, -1, 1]),
        },
        "hmm4_restricted": {
            "n_states": 4,
            "mask": maheu_restricted_mask(),
            "mean_signs": np.array([-1, 1, -1, 1]),
        },
    }


def evaluate(
    weekly: pd.Series,
    initial_train: int,
    refit_every: int,
    n_starts: int,
    max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if initial_train < 100:
        raise ValueError("initial_train should be at least 100 weeks")
    if len(weekly) <= initial_train:
        raise ValueError("Series is too short for requested initial training window")

    x = weekly.to_numpy(float)
    dates = weekly.index
    specs = model_specs()
    fits = {}
    filtered = {}
    rows = []
    fit_history = []

    for t in range(initial_train, len(x)):
        must_refit = t == initial_train or (refit_every > 0 and (t - initial_train) % refit_every == 0)
        if must_refit:
            train = x[:t]
            for model_name, spec in specs.items():
                fit = fit_gaussian_hmm(
                    train,
                    n_states=spec["n_states"],
                    transition_mask=spec["mask"],
                    mean_signs=spec["mean_signs"],
                    n_starts=n_starts,
                    max_iter=max_iter,
                    random_seed=SEED + t,
                )
                fits[model_name] = fit
                filtered_train, _, _ = forward_filter(
                    train,
                    fit.initial,
                    fit.transition,
                    fit.means,
                    fit.variances,
                )
                filtered[model_name] = filtered_train[-1]
                fit_history.append({
                    "refit_date": dates[t],
                    "train_weeks": int(t),
                    "model": model_name,
                    "train_log_likelihood": fit.log_likelihood,
                    "converged": fit.converged,
                    "n_iter": fit.n_iter,
                    "means": fit.means.tolist(),
                    "stddevs": np.sqrt(fit.variances).tolist(),
                    "transition": fit.transition.tolist(),
                })

        row = {
            "date": dates[t],
            "return_pct": float(x[t]),
            "iid_gaussian_log_density": iid_gaussian_log_density(x[:t], float(x[t])),
        }
        for model_name, fit in fits.items():
            density, updated = one_step_predictive_density(
                float(x[t]), filtered[model_name], fit
            )
            row[f"{model_name}_log_density"] = float(np.log(max(density, EPS)))
            for state in range(len(updated)):
                row[f"{model_name}_filtered_state_{state}"] = float(updated[state])
            filtered[model_name] = updated
        rows.append(row)

    predictions = pd.DataFrame(rows)
    score_rows = []
    density_columns = [c for c in predictions.columns if c.endswith("_log_density")]
    for column in density_columns:
        values = predictions[column].to_numpy(float)
        score_rows.append({
            "model": column.removesuffix("_log_density"),
            "n_forecasts": int(len(values)),
            "mean_log_predictive_density": float(np.mean(values)),
            "total_log_predictive_density": float(np.sum(values)),
            "median_log_predictive_density": float(np.median(values)),
        })
    scores = pd.DataFrame(score_rows).sort_values(
        "mean_log_predictive_density", ascending=False
    ).reset_index(drop=True)
    manifest = {
        "n_weeks": int(len(weekly)),
        "first_date": str(weekly.index.min().date()),
        "last_date": str(weekly.index.max().date()),
        "initial_train": int(initial_train),
        "refit_every": int(refit_every),
        "n_starts": int(n_starts),
        "max_iter": int(max_iter),
        "random_seed": SEED,
        "model_state_order": {
            "hmm2": ["negative", "positive"],
            "hmm4_restricted": ["bear", "bear_rally", "bull_correction", "bull"],
            "hmm4_unrestricted": ["negative_0", "positive_1", "negative_2", "positive_3"],
        },
        "fit_history": fit_history,
    }
    return predictions, scores, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--value-col", default="close")
    parser.add_argument(
        "--input-kind",
        choices=["price", "daily_log_return", "daily_simple_return", "weekly_return"],
        default="price",
    )
    parser.add_argument("--market-col")
    parser.add_argument("--market")
    parser.add_argument("--initial-train", type=int, default=520)
    parser.add_argument("--refit-every", type=int, default=52)
    parser.add_argument("--n-starts", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    weekly = load_weekly_returns(
        args.input,
        args.date_col,
        args.value_col,
        args.input_kind,
        args.market_col,
        args.market,
    )
    predictions, scores, manifest = evaluate(
        weekly,
        initial_train=args.initial_train,
        refit_every=args.refit_every,
        n_starts=args.n_starts,
        max_iter=args.max_iter,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(args.outdir / "weekly_returns.csv", header=True)
    predictions.to_csv(args.outdir / "one_step_predictions.csv", index=False)
    scores.to_csv(args.outdir / "predictive_scores.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Weekly regime baseline predictive scores (higher is better)")
    print(scores.to_string(index=False, float_format=lambda value: f"{value:.5f}"))


if __name__ == "__main__":
    main()
