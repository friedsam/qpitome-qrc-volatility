"""Stability audit for the relapse-conditioned volatility reservoir trail.

This is a diagnostic only. It does not refit any forecasting model.

Questions:
1. Is aggregate MSE skill robust to leaving out any one held-out relapse episode?
2. Which episodes contribute most to aggregate SSE improvement or deterioration?
3. Do model wins survive episode-level median aggregation?
4. How often does each correction improve RMSE, MAE, and QLIKE by episode?
5. Does an episode-block bootstrap support positive aggregate MSE skill?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ("har_plus_esn", "har_plus_tfim", "har_plus_rydberg")
BOOTSTRAP_SEED = 20260709
N_BOOTSTRAP = 20000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    actual = np.maximum(np.asarray(y, dtype=float), 1e-12)
    pred = np.maximum(np.asarray(yhat, dtype=float), 1e-12)
    ratio = actual / pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def episode_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_id, group in frame.groupby("episode_id"):
        y = group["actual_future_rv20"].to_numpy(dtype=float)
        har = group["har"].to_numpy(dtype=float)
        har_sse = float(np.sum((har - y) ** 2))
        har_rmse = float(np.sqrt(np.mean((har - y) ** 2)))
        har_mae = float(np.mean(np.abs(har - y)))
        har_qlike = qlike(y, har)

        for model in MODELS:
            pred = group[model].to_numpy(dtype=float)
            model_sse = float(np.sum((pred - y) ** 2))
            rows.append(
                {
                    "episode_id": int(episode_id),
                    "branch_date": group["branch_date"].iloc[0],
                    "model": model,
                    "n_rows": len(group),
                    "har_sse": har_sse,
                    "model_sse": model_sse,
                    "sse_improvement": har_sse - model_sse,
                    "mse_skill_vs_har": 1.0 - model_sse / har_sse if har_sse > 0 else np.nan,
                    "har_rmse": har_rmse,
                    "model_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
                    "delta_rmse": float(np.sqrt(np.mean((pred - y) ** 2))) - har_rmse,
                    "har_mae": har_mae,
                    "model_mae": float(np.mean(np.abs(pred - y))),
                    "delta_mae": float(np.mean(np.abs(pred - y))) - har_mae,
                    "har_qlike": har_qlike,
                    "model_qlike": qlike(y, pred),
                    "delta_qlike": qlike(y, pred) - har_qlike,
                }
            )
    return pd.DataFrame(rows)


def leave_one_episode_out_audit(stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in stats.groupby("model"):
        total_har_sse = float(group["har_sse"].sum())
        total_model_sse = float(group["model_sse"].sum())
        for left_out in group["episode_id"]:
            keep = group[group["episode_id"] != left_out]
            har_sse = float(keep["har_sse"].sum())
            model_sse = float(keep["model_sse"].sum())
            rows.append(
                {
                    "model": model,
                    "left_out_episode_id": int(left_out),
                    "n_episodes_remaining": len(keep),
                    "mse_skill_vs_har": 1.0 - model_sse / har_sse if har_sse > 0 else np.nan,
                    "full_sample_mse_skill_vs_har": (
                        1.0 - total_model_sse / total_har_sse if total_har_sse > 0 else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def model_summary(stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, group in stats.groupby("model"):
        total_har_sse = float(group["har_sse"].sum())
        total_model_sse = float(group["model_sse"].sum())
        rows.append(
            {
                "model": model,
                "n_episodes": len(group),
                "aggregate_mse_skill_vs_har": (
                    1.0 - total_model_sse / total_har_sse if total_har_sse > 0 else np.nan
                ),
                "median_episode_mse_skill": float(group["mse_skill_vs_har"].median()),
                "mean_episode_mse_skill": float(group["mse_skill_vs_har"].mean()),
                "episodes_positive_mse_skill": int((group["mse_skill_vs_har"] > 0).sum()),
                "episodes_better_rmse": int((group["delta_rmse"] < 0).sum()),
                "episodes_better_mae": int((group["delta_mae"] < 0).sum()),
                "episodes_better_qlike": int((group["delta_qlike"] < 0).sum()),
                "median_delta_rmse": float(group["delta_rmse"].median()),
                "median_delta_mae": float(group["delta_mae"].median()),
                "median_delta_qlike": float(group["delta_qlike"].median()),
                "largest_positive_sse_contribution": float(group["sse_improvement"].max()),
                "largest_negative_sse_contribution": float(group["sse_improvement"].min()),
            }
        )
    return pd.DataFrame(rows).sort_values("aggregate_mse_skill_vs_har", ascending=False)


def bootstrap_episode_blocks(stats: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for model, group in stats.groupby("model"):
        group = group.reset_index(drop=True)
        n = len(group)
        skills = np.empty(N_BOOTSTRAP, dtype=float)
        for draw in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            sample = group.iloc[idx]
            har_sse = float(sample["har_sse"].sum())
            model_sse = float(sample["model_sse"].sum())
            skills[draw] = 1.0 - model_sse / har_sse if har_sse > 0 else np.nan
        finite = skills[np.isfinite(skills)]
        rows.append(
            {
                "model": model,
                "n_bootstrap": len(finite),
                "bootstrap_median_skill": float(np.median(finite)),
                "bootstrap_ci_2p5": float(np.quantile(finite, 0.025)),
                "bootstrap_ci_97p5": float(np.quantile(finite, 0.975)),
                "bootstrap_probability_skill_positive": float(np.mean(finite > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("bootstrap_probability_skill_positive", ascending=False)


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise FileNotFoundError(args.predictions)
    args.output.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.predictions, parse_dates=["branch_date", "row_date"])
    required = {
        "episode_id",
        "actual_future_rv20",
        "har",
        *MODELS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Missing prediction columns: {sorted(missing)}")

    stats = episode_stats(frame)
    summary = model_summary(stats)
    loo = leave_one_episode_out_audit(stats)
    bootstrap = bootstrap_episode_blocks(stats)

    contribution = stats.sort_values(["model", "sse_improvement"], ascending=[True, False])
    stats.to_csv(args.output / "per_episode_stability_metrics.csv", index=False)
    summary.to_csv(args.output / "model_stability_summary.csv", index=False)
    loo.to_csv(args.output / "leave_one_episode_out_skill.csv", index=False)
    bootstrap.to_csv(args.output / "episode_block_bootstrap.csv", index=False)
    contribution.to_csv(args.output / "episode_sse_contributions.csv", index=False)

    print("Model stability summary:")
    print(summary.to_string(index=False))

    print("\nLeave-one-episode-out skill range:")
    loo_range = (
        loo.groupby("model")["mse_skill_vs_har"]
        .agg(["min", "median", "max"])
        .reset_index()
    )
    print(loo_range.to_string(index=False))

    print("\nEpisode-block bootstrap:")
    print(bootstrap.to_string(index=False))

    print("\nPer-episode SSE contributions (positive helps, negative hurts):")
    print(
        contribution[
            [
                "model",
                "episode_id",
                "har_sse",
                "model_sse",
                "sse_improvement",
                "mse_skill_vs_har",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
