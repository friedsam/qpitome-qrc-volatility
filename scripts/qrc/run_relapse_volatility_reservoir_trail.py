"""Matched ESN -> TFIM -> Rydberg trail on relapse-conditioned volatility.

This is a retrospective conditional diagnostic. It asks whether the specific
relapse episodes where HAR underperforms persistence contain modelable residual
volatility dynamics.

Unit of evaluation:
- rows from relapse episodes, branch_idx through branch_idx + 20 inclusive;
- 40-day causal input window ending at each forecast row;
- target is future_rv_20d at that row;
- test one relapse episode at a time;
- train corrections only on rows from earlier completed relapse episodes.

Models:
1. causal HAR baseline;
2. HAR + 8-unit ESN residual correction;
3. HAR + 8-observable TFIM residual correction;
4. HAR + 8-observable Rydberg residual correction.

All reservoirs receive the same four frozen task-aligned path channels and the
same 40-day windows. No architecture or regularization search is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.branch_har import HAR_FEATURES, HAR_TARGET, HAR_TARGET_HORIZON_ROWS
from qpitome_qrc.baselines.branch_path_reservoir import reset_esn_features
from qpitome_qrc.qrc.rydberg_temporal import RydbergTemporalConfig, rydberg_temporal_features
from qpitome_qrc.qrc.tfim_transition import TFIMTransitionConfig, tfim_transition_features
from qpitome_qrc.regimes.branch_transition_path import (
    TRANSITION_PATH_COLUMNS,
    add_transition_path_channels,
)

LOOKBACK = 40
RELAPSE_FORWARD_ROWS = 20
MIN_PRIOR_RELAPSE_EPISODES = 5
CORRECTION_ALPHA = 10.0
ESN_SEEDS = (0, 1, 2)
ESN_CONFIG = dict(
    n_reservoir=8,
    spectral_radius=0.9,
    input_scale=0.3,
    leak=0.3,
)
TFIM_CONFIG = TFIMTransitionConfig()
RYDBERG_CONFIG = RydbergTemporalConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    actual = np.maximum(np.asarray(y, dtype=float), 1e-12)
    pred = np.maximum(np.asarray(yhat, dtype=float), 1e-12)
    ratio = actual / pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def fit_causal_har_forecast(
    daily: pd.DataFrame,
    row_idx: int,
    *,
    alpha: float = 1.0,
    min_train_rows: int = 504,
) -> tuple[float, int]:
    latest_train_idx = row_idx - HAR_TARGET_HORIZON_ROWS - 1
    if latest_train_idx < 0:
        raise ValueError(f"No leakage-safe HAR training rows for row_idx={row_idx}")

    x_all = daily.loc[:, HAR_FEATURES].to_numpy(dtype=float)
    y_all = daily[HAR_TARGET].to_numpy(dtype=float)
    train_slice = slice(0, latest_train_idx + 1)
    finite = np.isfinite(y_all[train_slice]) & np.isfinite(x_all[train_slice]).all(axis=1)
    x_train = x_all[train_slice][finite]
    y_train = y_all[train_slice][finite]
    if len(y_train) < min_train_rows:
        raise ValueError(f"Only {len(y_train)} causal HAR training rows for row_idx={row_idx}")

    x_test = x_all[[row_idx]]
    if not np.isfinite(x_test).all():
        raise ValueError(f"Non-finite HAR inputs at row_idx={row_idx}")

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x_train, y_train)
    prediction = max(float(model.predict(x_test)[0]), 1e-8)
    return prediction, len(y_train)


def build_relapse_row_dataset(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    channels = add_transition_path_channels(daily)
    channel_values = channels.loc[:, TRANSITION_PATH_COLUMNS].to_numpy(dtype=float)

    relapse = episodes[
        (episodes["outcome"] == "relapse")
        & episodes["outcome_complete"].astype(bool)
    ].sort_values("branch_date").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    windows: list[np.ndarray] = []
    har_cache: dict[int, tuple[float, int]] = {}

    for relapse_order, episode in enumerate(relapse.itertuples(index=False)):
        branch_idx = int(episode.branch_idx)
        for offset in range(RELAPSE_FORWARD_ROWS + 1):
            row_idx = branch_idx + offset
            start = row_idx - LOOKBACK + 1
            if start < 0 or row_idx >= len(daily):
                continue
            window = channel_values[start : row_idx + 1]
            if len(window) != LOOKBACK or not np.isfinite(window).all():
                continue
            actual = float(daily.iloc[row_idx][HAR_TARGET])
            if not np.isfinite(actual):
                continue
            if row_idx not in har_cache:
                har_cache[row_idx] = fit_causal_har_forecast(daily, row_idx)
            har_pred, n_har_train = har_cache[row_idx]

            rows.append(
                {
                    "episode_id": int(episode.episode_id),
                    "relapse_order": relapse_order,
                    "branch_date": episode.branch_date,
                    "row_idx": row_idx,
                    "row_date": daily.iloc[row_idx]["date"],
                    "offset_from_branch": offset,
                    "actual_future_rv20": actual,
                    "har_prediction": har_pred,
                    "har_residual": actual - har_pred,
                    "har_train_rows": n_har_train,
                }
            )
            windows.append(window)

    return pd.DataFrame(rows), np.asarray(windows, dtype=float)


def fit_residual_correction(
    x_train: np.ndarray,
    residual_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    model = Ridge(alpha=CORRECTION_ALPHA, fit_intercept=False)
    model.fit(x_train, residual_train)
    return model.predict(x_test)


def pooled_metrics(frame: pd.DataFrame, model: str) -> dict[str, object]:
    actual = frame["actual_future_rv20"].to_numpy(dtype=float)
    pred = frame[model].to_numpy(dtype=float)
    har = frame["har_prediction"].to_numpy(dtype=float)
    mse = float(np.mean((pred - actual) ** 2))
    har_mse = float(np.mean((har - actual) ** 2))
    return {
        "weighting": "row_weighted",
        "model": model,
        "n_rows": len(frame),
        "n_episodes": frame["episode_id"].nunique(),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(pred - actual))),
        "qlike": qlike(actual, pred),
        "mse_skill_vs_har": 1.0 - mse / har_mse if har_mse > 0 else np.nan,
    }


def episode_balanced_metrics(frame: pd.DataFrame, model: str) -> dict[str, object]:
    episode_rows = []
    for _, group in frame.groupby("episode_id"):
        actual = group["actual_future_rv20"].to_numpy(dtype=float)
        pred = group[model].to_numpy(dtype=float)
        har = group["har_prediction"].to_numpy(dtype=float)
        episode_rows.append(
            {
                "mse": float(np.mean((pred - actual) ** 2)),
                "har_mse": float(np.mean((har - actual) ** 2)),
                "mae": float(np.mean(np.abs(pred - actual))),
                "qlike": qlike(actual, pred),
            }
        )
    episode_frame = pd.DataFrame(episode_rows)
    mean_mse = float(episode_frame["mse"].mean())
    mean_har_mse = float(episode_frame["har_mse"].mean())
    return {
        "weighting": "episode_balanced",
        "model": model,
        "n_rows": len(frame),
        "n_episodes": frame["episode_id"].nunique(),
        "rmse": float(np.sqrt(mean_mse)),
        "mae": float(episode_frame["mae"].mean()),
        "qlike": float(episode_frame["qlike"].mean()),
        "mse_skill_vs_har": (
            1.0 - mean_mse / mean_har_mse if mean_har_mse > 0 else np.nan
        ),
    }


def main() -> None:
    args = parse_args()
    for path in (args.data, args.episodes):
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)

    daily = (
        pd.read_csv(args.data, parse_dates=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    episodes = (
        pd.read_csv(args.episodes, parse_dates=["branch_date"])
        .sort_values("branch_date")
        .reset_index(drop=True)
    )

    frame, windows = build_relapse_row_dataset(daily, episodes)
    if frame.empty:
        raise RuntimeError("No eligible relapse-conditioned forecast rows")

    print(f"Relapse episodes in row dataset: {frame['episode_id'].nunique()}")
    print(f"Eligible row forecasts: {len(frame)}")
    print(f"Window shape: {windows.shape}")

    esn_features_by_seed = {}
    for seed in ESN_SEEDS:
        full = reset_esn_features(windows, seed=seed, **ESN_CONFIG)
        esn_features_by_seed[seed] = full[:, : ESN_CONFIG["n_reservoir"]]

    print("Computing TFIM features...")
    tfim_features = tfim_transition_features(windows, TFIM_CONFIG)
    print("Computing Rydberg features...")
    rydberg_features = rydberg_temporal_features(windows, RYDBERG_CONFIG)

    predictions: list[pd.DataFrame] = []
    unique_orders = sorted(frame["relapse_order"].unique())
    for test_order in unique_orders:
        if test_order < MIN_PRIOR_RELAPSE_EPISODES:
            continue
        train_mask = frame["relapse_order"] < test_order
        test_mask = frame["relapse_order"] == test_order
        train = frame.loc[train_mask]
        test = frame.loc[test_mask].copy()
        if train["episode_id"].nunique() < MIN_PRIOR_RELAPSE_EPISODES:
            continue

        residual_train = train["har_residual"].to_numpy(dtype=float)
        test["har"] = test["har_prediction"].to_numpy(dtype=float)

        esn_preds = []
        for seed in ESN_SEEDS:
            correction = fit_residual_correction(
                esn_features_by_seed[seed][train_mask.to_numpy()],
                residual_train,
                esn_features_by_seed[seed][test_mask.to_numpy()],
            )
            esn_preds.append(test["har_prediction"].to_numpy(dtype=float) + correction)
        test["har_plus_esn"] = np.mean(np.vstack(esn_preds), axis=0)

        tfim_correction = fit_residual_correction(
            tfim_features[train_mask.to_numpy()],
            residual_train,
            tfim_features[test_mask.to_numpy()],
        )
        test["har_plus_tfim"] = test["har_prediction"].to_numpy(dtype=float) + tfim_correction

        rydberg_correction = fit_residual_correction(
            rydberg_features[train_mask.to_numpy()],
            residual_train,
            rydberg_features[test_mask.to_numpy()],
        )
        test["har_plus_rydberg"] = test["har_prediction"].to_numpy(dtype=float) + rydberg_correction

        for column in ("har", "har_plus_esn", "har_plus_tfim", "har_plus_rydberg"):
            test[column] = test[column].clip(lower=1e-8)
        predictions.append(test)

    if not predictions:
        raise RuntimeError("No episode-blocked OOS predictions produced")
    oos = pd.concat(predictions, ignore_index=True)

    model_columns = ("har", "har_plus_esn", "har_plus_tfim", "har_plus_rydberg")
    metric_rows = []
    for model in model_columns:
        metric_rows.append(pooled_metrics(oos, model))
        metric_rows.append(episode_balanced_metrics(oos, model))
    metrics = pd.DataFrame(metric_rows).sort_values(["weighting", "rmse"])

    episode_metrics = []
    for episode_id, group in oos.groupby("episode_id"):
        for model in model_columns:
            row = pooled_metrics(group, model)
            row["episode_id"] = int(episode_id)
            row["branch_date"] = group["branch_date"].iloc[0]
            episode_metrics.append(row)
    episode_metrics_frame = pd.DataFrame(episode_metrics)

    oos.to_csv(args.output / "oos_row_predictions.csv", index=False)
    metrics.to_csv(args.output / "summary_metrics.csv", index=False)
    episode_metrics_frame.to_csv(args.output / "per_episode_metrics.csv", index=False)

    (args.output / "run_manifest.json").write_text(
        json.dumps(
            {
                "status": "retrospective relapse-conditioned diagnostic; not deployable classification",
                "claim": (
                    "HAR residual volatility inside relapse episodes contains recoverable "
                    "temporal structure for ESN, TFIM, or Rydberg reservoirs"
                ),
                "falsifying_result": (
                    "none of the matched residual corrections improves row-weighted and "
                    "episode-balanced OOS metrics versus causal HAR"
                ),
                "data": str(args.data),
                "episodes": str(args.episodes),
                "row_geometry": {
                    "offsets_from_branch_inclusive": [0, RELAPSE_FORWARD_ROWS],
                    "lookback": LOOKBACK,
                    "target": HAR_TARGET,
                    "test_unit": "entire later relapse episode",
                    "training_unit": "rows from earlier relapse episodes only",
                    "min_prior_relapse_episodes": MIN_PRIOR_RELAPSE_EPISODES,
                },
                "har": {
                    "features": list(HAR_FEATURES),
                    "alpha": 1.0,
                    "target_horizon_rows": HAR_TARGET_HORIZON_ROWS,
                    "causal_rule": "latest HAR training target row = forecast row - 21",
                },
                "shared_reservoir_inputs": list(TRANSITION_PATH_COLUMNS),
                "esn": {**ESN_CONFIG, "seeds": list(ESN_SEEDS), "readout_dimension": 8},
                "tfim": {**TFIM_CONFIG.__dict__, "readout_dimension": 8},
                "rydberg": {**RYDBERG_CONFIG.__dict__, "readout_dimension": 8},
                "residual_correction": {
                    "model": "Ridge(fit_intercept=False)",
                    "alpha": CORRECTION_ALPHA,
                    "target": "actual_future_rv20 - causal_har_prediction",
                },
                "metrics": [
                    "row-weighted RMSE/MAE/QLIKE/MSE skill vs HAR",
                    "episode-balanced RMSE/MAE/QLIKE/MSE skill vs HAR",
                ],
                "tuning": "none",
            },
            indent=2,
        )
        + "\n"
    )

    print(f"OOS relapse episodes: {oos['episode_id'].nunique()}")
    print(f"OOS row forecasts: {len(oos)}")
    print("\nMatched relapse-volatility trail:")
    print(metrics.to_string(index=False))
    print("\nPer-episode MSE skill versus HAR:")
    print(
        episode_metrics_frame.pivot(
            index="episode_id",
            columns="model",
            values="mse_skill_vs_har",
        ).to_string()
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
