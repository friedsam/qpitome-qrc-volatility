"""Audit continuous HAR volatility forecasting on branch episodes.

This closes the logical loop between two claims:

1. HAR may still forecast future volatility reasonably well in branch states.
2. The same HAR forecast may be poor at resolving recovery versus relapse.

HAR is compared against current-RV persistence on the exact same episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_HAR_FEATURES = Path(
    "results/baselines/branch_har_front_v1/har_episode_features.csv"
)
DEFAULT_OUTPUT = Path("results/baselines/branch_har_continuous_audit_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--har-features", type=Path, default=DEFAULT_HAR_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    yhat = np.maximum(np.asarray(yhat, dtype=float), 1e-12)
    ratio = y / yhat
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mz_fit(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(yhat)), yhat])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(coef[0]), float(coef[1])


def metric_row(group: str, model: str, y: np.ndarray, yhat: np.ndarray) -> dict[str, object]:
    err = yhat - y
    intercept, slope = mz_fit(y, yhat)
    return {
        "group": group,
        "model": model,
        "n": len(y),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "qlike": qlike(y, yhat),
        "mz_intercept": intercept,
        "mz_slope": slope,
        "mean_actual": float(np.mean(y)),
        "mean_prediction": float(np.mean(yhat)),
    }


def main() -> None:
    args = parse_args()
    for path in (args.data, args.episodes, args.har_features):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(args.episodes, parse_dates=["branch_date"])
    har = pd.read_csv(args.har_features, parse_dates=["branch_date"])

    required_daily = {"future_rv_20d"}
    required_episode = {"episode_id", "branch_idx", "outcome", "rv_20d", "outcome_complete"}
    required_har = {"episode_id", "har_future_rv20_pred"}
    if missing := required_daily - set(daily.columns):
        raise KeyError(f"Missing daily columns: {sorted(missing)}")
    if missing := required_episode - set(episodes.columns):
        raise KeyError(f"Missing episode columns: {sorted(missing)}")
    if missing := required_har - set(har.columns):
        raise KeyError(f"Missing HAR columns: {sorted(missing)}")

    frame = episodes[episodes["outcome_complete"]].copy()
    frame = frame.merge(
        har[["episode_id", "har_future_rv20_pred"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    frame["actual_future_rv20"] = daily.iloc[
        frame["branch_idx"].astype(int).to_numpy()
    ]["future_rv_20d"].to_numpy(dtype=float)
    frame["persistence_pred"] = frame["rv_20d"].to_numpy(dtype=float)

    if frame[["actual_future_rv20", "har_future_rv20_pred", "persistence_pred"]].isna().any().any():
        raise ValueError("Missing continuous targets or predictions")

    groups = {
        "all_complete": frame,
        "binary": frame[frame["outcome"].isin(["recovery", "relapse"])],
        "mixed": frame[frame["outcome"] == "mixed"],
        "recovery": frame[frame["outcome"] == "recovery"],
        "relapse": frame[frame["outcome"] == "relapse"],
    }

    rows: list[dict[str, object]] = []
    for group_name, group in groups.items():
        y = group["actual_future_rv20"].to_numpy(dtype=float)
        for model, column in (
            ("persistence", "persistence_pred"),
            ("har", "har_future_rv20_pred"),
        ):
            rows.append(
                metric_row(
                    group_name,
                    model,
                    y,
                    group[column].to_numpy(dtype=float),
                )
            )

    metrics = pd.DataFrame(rows)

    skill_rows: list[dict[str, object]] = []
    for group_name, group in groups.items():
        y = group["actual_future_rv20"].to_numpy(dtype=float)
        har_err = group["har_future_rv20_pred"].to_numpy(dtype=float) - y
        persistence_err = group["persistence_pred"].to_numpy(dtype=float) - y
        sse_har = float(np.sum(har_err**2))
        sse_persistence = float(np.sum(persistence_err**2))
        skill_rows.append(
            {
                "group": group_name,
                "n": len(group),
                "innovation_r2_vs_persistence": (
                    1.0 - sse_har / sse_persistence if sse_persistence > 0 else np.nan
                ),
                "rmse_ratio_har_over_persistence": (
                    np.sqrt(sse_har / len(group)) / np.sqrt(sse_persistence / len(group))
                    if sse_persistence > 0
                    else np.nan
                ),
                "qlike_har_minus_persistence": (
                    qlike(y, group["har_future_rv20_pred"].to_numpy(dtype=float))
                    - qlike(y, group["persistence_pred"].to_numpy(dtype=float))
                ),
            }
        )
    skill = pd.DataFrame(skill_rows)

    frame.to_csv(args.output / "episode_predictions.csv", index=False)
    metrics.to_csv(args.output / "continuous_metrics.csv", index=False)
    skill.to_csv(args.output / "har_skill_vs_persistence.csv", index=False)

    manifest = {
        "data": str(args.data),
        "episodes": str(args.episodes),
        "har_features": str(args.har_features),
        "models": {
            "har": "causal episode-specific canonical HAR forecast from branch_har_front_v1",
            "persistence": "current rv_20d at branch point",
        },
        "groups": list(groups.keys()),
        "metrics": [
            "RMSE",
            "MAE",
            "QLIKE",
            "Mincer-Zarnowitz intercept",
            "Mincer-Zarnowitz slope",
            "innovation R2 versus persistence",
        ],
        "status": "diagnostic; small branch-conditioned samples",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Continuous HAR metrics:")
    print(metrics.to_string(index=False))
    print("\nHAR skill versus persistence:")
    print(skill.to_string(index=False))
    print("\nPer-episode forecasts:")
    print(
        frame[
            [
                "episode_id",
                "branch_date",
                "outcome",
                "rv_20d",
                "actual_future_rv20",
                "persistence_pred",
                "har_future_rv20_pred",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
