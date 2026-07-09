"""Quickshot input-family exploration for branch-conditioned HAR residuals.

Two fixed families are compared under the same prequential protocol, residual
target, reservoir configuration, and readout:

1. volatility_failure: dynamics designed around failed volatility normalization;
2. har_aware: causal HAR forecast path, revisions, and already-observable errors.

This is exploratory mechanism work, not a tuning grid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.branch_har import HAR_FEATURES, HAR_TARGET
from qpitome_qrc.baselines.numpy_esn import esn_states, make_esn_weights
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path("results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv")
DEFAULT_HAR = Path("results/baselines/branch_har_front_v1/har_episode_features.csv")
DEFAULT_OUTPUT = Path("results/baselines/branch_har_residual_input_families_v1")
LOOKBACK = 40
SEEDS = (0, 1, 2)
ESN_CONFIG = dict(n_reservoir=50, spectral_radius=0.9, input_scale=0.3, leak=0.3)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)
HAR_TARGET_HORIZON = 20
HAR_MIN_TRAIN = 504

VOLATILITY_FAILURE_COLUMNS = (
    "log_rv5_over_rv20",
    "log_rv20_over_rv60",
    "d_log_rv5_over_rv20_5d",
    "d_log_rv20_over_rv60_5d",
    "rv5_acceleration_scaled",
    "rv20_change_scaled",
    "rv5_distance_from_20d_peak",
    "rv5_reexpansion_count_10d",
    "rv5_reexpansion_magnitude_10d",
)

HAR_AWARE_COLUMNS = (
    "har_log_future_to_current_daily",
    "har_forecast_revision_5d",
    "har_observable_log_error_20d",
    "har_observable_error_change_5d",
    "log_rv5_over_rv20",
    "log_rv20_over_rv60",
    "rv5_change_scaled",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    p.add_argument("--har-features", type=Path, default=DEFAULT_HAR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.maximum(np.asarray(y, float), 1e-12)
    yhat = np.maximum(np.asarray(yhat, float), 1e-12)
    ratio = y / yhat
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mz_fit(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(yhat)), yhat])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(coef[0]), float(coef[1])


def add_base_ratios(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy().sort_values("date").reset_index(drop=True)
    out["log_rv5_over_rv20"] = np.log(out["rv_5d"] / out["rv_20d"])
    out["log_rv20_over_rv60"] = np.log(out["rv_20d"] / out["rv_60d"])
    out["rv5_change_scaled"] = (out["rv_5d"] - out["rv_5d"].shift(5)) / out["rv_20d"]
    return out


def add_volatility_failure_channels(daily: pd.DataFrame) -> pd.DataFrame:
    out = add_base_ratios(daily)
    out["d_log_rv5_over_rv20_5d"] = out["log_rv5_over_rv20"].diff(5)
    out["d_log_rv20_over_rv60_5d"] = out["log_rv20_over_rv60"].diff(5)
    rv5_change = out["rv_5d"].diff(5)
    out["rv5_acceleration_scaled"] = (rv5_change - rv5_change.shift(5)) / out["rv_20d"]
    out["rv20_change_scaled"] = out["rv_20d"].diff(5) / out["rv_20d"]
    out["rv5_distance_from_20d_peak"] = np.log(
        out["rv_5d"] / out["rv_5d"].rolling(20).max()
    )
    up = (out["rv_5d"].diff() > 0).astype(float)
    pos_change = out["rv_5d"].diff().clip(lower=0.0)
    out["rv5_reexpansion_count_10d"] = up.rolling(10).sum() / 10.0
    out["rv5_reexpansion_magnitude_10d"] = pos_change.rolling(10).sum() / out["rv_20d"]

    bounds = {
        "log_rv5_over_rv20": (-2.0, 2.0),
        "log_rv20_over_rv60": (-1.5, 1.5),
        "d_log_rv5_over_rv20_5d": (-2.0, 2.0),
        "d_log_rv20_over_rv60_5d": (-1.5, 1.5),
        "rv5_acceleration_scaled": (-4.0, 4.0),
        "rv20_change_scaled": (-2.0, 2.0),
        "rv5_distance_from_20d_peak": (-3.0, 0.1),
        "rv5_reexpansion_count_10d": (0.0, 1.0),
        "rv5_reexpansion_magnitude_10d": (0.0, 4.0),
    }
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


def causal_har_predictions_for_rows(
    daily: pd.DataFrame,
    prediction_rows: np.ndarray,
) -> np.ndarray:
    """Compute exact canonical HAR forecasts only at requested daily rows."""
    x_all = daily.loc[:, HAR_FEATURES].to_numpy(float)
    y_all = daily[HAR_TARGET].to_numpy(float)
    preds = np.full(len(daily), np.nan, dtype=float)

    for row_idx in np.unique(prediction_rows.astype(int)):
        latest_train_idx = row_idx - HAR_TARGET_HORIZON - 1
        if latest_train_idx < 0:
            continue
        x_slice = x_all[: latest_train_idx + 1]
        y_slice = y_all[: latest_train_idx + 1]
        finite = np.isfinite(y_slice) & np.isfinite(x_slice).all(axis=1)
        x_train = x_slice[finite]
        y_train = y_slice[finite]
        if len(y_train) < HAR_MIN_TRAIN or not np.isfinite(x_all[[row_idx]]).all():
            continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(x_train, y_train)
        preds[row_idx] = max(float(model.predict(x_all[[row_idx]])[0]), 1e-8)
    return preds


def add_har_aware_channels(daily: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    out = add_base_ratios(daily)
    needed: set[int] = set()
    for branch_idx in episodes["branch_idx"].astype(int):
        start = branch_idx - LOOKBACK + 1
        # Need 20-day lagged forecasts and 5-day changes for every path row.
        needed.update(range(max(0, start - 25), branch_idx + 1))
    needed_rows = np.asarray(sorted(needed), dtype=int)
    har_pred = causal_har_predictions_for_rows(out, needed_rows)
    out["causal_har_pred_daily"] = har_pred
    out["har_log_future_to_current_daily"] = np.log(
        out["causal_har_pred_daily"] / out["rv_20d"]
    )
    out["har_forecast_revision_5d"] = np.log(
        out["causal_har_pred_daily"] / out["causal_har_pred_daily"].shift(5)
    )
    # At row t, the forecast made at t-20 has a fully realized RV20 target now.
    out["har_observable_log_error_20d"] = np.log(
        out["rv_20d"] / out["causal_har_pred_daily"].shift(20)
    )
    out["har_observable_error_change_5d"] = out["har_observable_log_error_20d"].diff(5)

    bounds = {
        "har_log_future_to_current_daily": (-2.0, 2.0),
        "har_forecast_revision_5d": (-1.5, 1.5),
        "har_observable_log_error_20d": (-2.0, 2.0),
        "har_observable_error_change_5d": (-2.0, 2.0),
        "log_rv5_over_rv20": (-2.0, 2.0),
        "log_rv20_over_rv60": (-1.5, 1.5),
        "rv5_change_scaled": (-3.0, 3.0),
    }
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


def extract_windows(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    values = daily.loc[:, columns].to_numpy(float)
    ids: list[int] = []
    windows: list[np.ndarray] = []
    for row in episodes.itertuples(index=False):
        end = int(row.branch_idx) + 1
        start = end - LOOKBACK
        window = values[start:end]
        if len(window) != LOOKBACK or not np.isfinite(window).all():
            raise ValueError(
                f"Non-finite or incomplete {columns[0]} window for episode {row.episode_id}"
            )
        ids.append(int(row.episode_id))
        windows.append(window)
    return np.asarray(ids), np.asarray(windows)


def continuous_states(
    daily: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    seed: int,
) -> np.ndarray:
    x = daily.loc[:, columns].to_numpy(float)
    finite = np.isfinite(x).all(axis=1)
    if not finite.any():
        raise ValueError("No finite rows for continuous ESN")
    first = int(np.flatnonzero(finite)[0])
    if not finite[first:].all():
        bad = int(np.flatnonzero(~finite[first:])[0] + first)
        raise ValueError(f"Non-finite family row after warm-up: {bad}")
    w_in, w = make_esn_weights(
        n_inputs=x.shape[1],
        seed=seed,
        **{k: ESN_CONFIG[k] for k in ("n_reservoir", "spectral_radius", "input_scale")},
    )
    h = np.zeros(ESN_CONFIG["n_reservoir"])
    rows = np.full((len(x), ESN_CONFIG["n_reservoir"] + x.shape[1]), np.nan)
    for idx in range(first, len(x)):
        u_t = x[idx]
        h_new = np.tanh(w_in @ u_t + w @ h)
        h = (1.0 - ESN_CONFIG["leak"]) * h + ESN_CONFIG["leak"] * h_new
        rows[idx] = np.concatenate([h.copy(), u_t])
    return rows


def family_feature_maps(
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    columns: tuple[str, ...],
) -> dict[str, dict[int, np.ndarray]]:
    ids, windows = extract_windows(daily, episodes, columns)
    maps: dict[str, dict[int, np.ndarray]] = {
        "full_path_linear": {int(i): windows[k].reshape(-1) for k, i in enumerate(ids)}
    }
    for seed in SEEDS:
        w_in, w = make_esn_weights(
            n_inputs=windows.shape[2],
            seed=seed,
            **{k: ESN_CONFIG[k] for k in ("n_reservoir", "spectral_radius", "input_scale")},
        )
        reset = esn_states(windows, w_in, w, ESN_CONFIG["leak"])
        maps[f"reset_esn_seed{seed}"] = {int(i): reset[k] for k, i in enumerate(ids)}
        states = continuous_states(daily, columns, seed=seed)
        maps[f"continuous_esn_seed{seed}"] = {
            int(row.episode_id): states[int(row.branch_idx)]
            for row in episodes.itertuples(index=False)
        }
    return maps


def fit_residual(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[float, float]:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])
    model.fit(x_train, y_train)
    return float(model.predict(x_test)[0]), float(model.named_steps["ridge"].alpha_)


def metric_row(family: str, model: str, g: pd.DataFrame) -> dict[str, object]:
    y = g["actual_future_rv20"].to_numpy(float)
    yhat = g["combined_future_rv20_pred"].to_numpy(float)
    r = g["har_log_residual_actual"].to_numpy(float)
    rhat = g["har_log_residual_pred"].to_numpy(float)
    sse = float(np.sum((r - rhat) ** 2))
    sse0 = float(np.sum(r**2))
    intercept, slope = mz_fit(y, yhat)
    return {
        "family": family,
        "model": model,
        "n_oos": len(g),
        "future_rv_rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
        "future_rv_mae": float(np.mean(np.abs(yhat - y))),
        "qlike": qlike(y, yhat),
        "mz_intercept": intercept,
        "mz_slope": slope,
        "har_residual_rmse": float(np.sqrt(np.mean((rhat - r) ** 2))),
        "residual_r2_vs_har": 1.0 - sse / sse0 if sse0 > 0 else np.nan,
    }


def main() -> None:
    a = parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(a.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(a.episodes, parse_dates=["branch_date"])
    har = pd.read_csv(a.har_features, parse_dates=["branch_date"])
    episodes = episodes.merge(
        har[["episode_id", "har_future_rv20_pred"]],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    episodes["actual_future_rv20"] = daily.iloc[
        episodes["branch_idx"].astype(int).to_numpy()
    ]["future_rv_20d"].to_numpy(float)
    episodes["har_log_residual_actual"] = np.log(
        episodes["actual_future_rv20"] / episodes["har_future_rv20_pred"]
    )

    family_frames = {
        "volatility_failure": (add_volatility_failure_channels(daily), VOLATILITY_FAILURE_COLUMNS),
        "har_aware": (add_har_aware_channels(daily, episodes), HAR_AWARE_COLUMNS),
    }
    feature_maps = {
        family: family_feature_maps(frame, episodes, columns)
        for family, (frame, columns) in family_frames.items()
    }

    protocol = EpisodePrequentialConfig(min_train_episodes=18, outcome_horizon_rows=40, embargo_rows=0)
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    lookup = episodes.set_index("episode_id", drop=False)
    rows: list[dict[str, object]] = []

    for step in steps.itertuples(index=False):
        test_id = int(step.test_episode_id)
        test_ep = lookup.loc[test_id]
        train_ids = assignments[
            (assignments["step"] == step.step) & (assignments["split"] == "train")
        ]["episode_id"].astype(int).tolist()
        y_train = lookup.loc[train_ids, "har_log_residual_actual"].to_numpy(float)

        for family, fmap_by_model in feature_maps.items():
            rows.append({
                "family": family,
                "step": int(step.step),
                "episode_id": test_id,
                "branch_date": test_ep["branch_date"],
                "outcome": test_ep["outcome"],
                "n_train": len(train_ids),
                "model": "har_zero_residual",
                "selected_alpha": np.nan,
                "har_log_residual_actual": float(test_ep["har_log_residual_actual"]),
                "har_log_residual_pred": 0.0,
                "har_future_rv20_pred": float(test_ep["har_future_rv20_pred"]),
                "actual_future_rv20": float(test_ep["actual_future_rv20"]),
            })
            for model_name, fmap in fmap_by_model.items():
                pred, alpha = fit_residual(
                    np.vstack([fmap[i] for i in train_ids]),
                    y_train,
                    fmap[test_id][None, :],
                )
                rows.append({
                    "family": family,
                    "step": int(step.step),
                    "episode_id": test_id,
                    "branch_date": test_ep["branch_date"],
                    "outcome": test_ep["outcome"],
                    "n_train": len(train_ids),
                    "model": model_name,
                    "selected_alpha": alpha,
                    "har_log_residual_actual": float(test_ep["har_log_residual_actual"]),
                    "har_log_residual_pred": pred,
                    "har_future_rv20_pred": float(test_ep["har_future_rv20_pred"]),
                    "actual_future_rv20": float(test_ep["actual_future_rv20"]),
                })

    pred = pd.DataFrame(rows)
    pred["combined_future_rv20_pred"] = (
        pred["har_future_rv20_pred"] * np.exp(pred["har_log_residual_pred"])
    ).clip(lower=1e-8)

    ensemble_rows = []
    for family in family_frames:
        family_pred = pred[pred["family"] == family]
        for reservoir_family in ("reset_esn", "continuous_esn"):
            seed_models = [f"{reservoir_family}_seed{s}" for s in SEEDS]
            subset = family_pred[family_pred["model"].isin(seed_models)]
            wide = subset.pivot(index="episode_id", columns="model", values="har_log_residual_pred")
            ensemble = wide.mean(axis=1)
            base = subset[subset["model"] == seed_models[0]].copy()
            base["model"] = f"{reservoir_family}_ensemble"
            base["selected_alpha"] = np.nan
            base["har_log_residual_pred"] = base["episode_id"].map(ensemble)
            base["combined_future_rv20_pred"] = (
                base["har_future_rv20_pred"] * np.exp(base["har_log_residual_pred"])
            ).clip(lower=1e-8)
            ensemble_rows.append(base)
    pred = pd.concat([pred, *ensemble_rows], ignore_index=True)

    summary = pd.DataFrame([
        metric_row(family, model, g)
        for (family, model), g in pred.groupby(["family", "model"])
    ]).sort_values(["future_rv_rmse", "qlike"])

    outcome_rows = []
    for (family, model, outcome), g in pred.groupby(["family", "model", "outcome"]):
        row = metric_row(family, model, g)
        row["outcome"] = outcome
        outcome_rows.append(row)
    by_outcome = pd.DataFrame(outcome_rows)

    pred.to_csv(a.output / "oos_predictions.csv", index=False)
    summary.to_csv(a.output / "summary_metrics.csv", index=False)
    by_outcome.to_csv(a.output / "metrics_by_outcome.csv", index=False)
    pd.DataFrame([
        {"family": family, "channel": channel}
        for family, (_, columns) in family_frames.items()
        for channel in columns
    ]).to_csv(a.output / "input_families.csv", index=False)
    (a.output / "run_manifest.json").write_text(json.dumps({
        "status": "quickshot exploratory input-family comparison",
        "target": "log(actual future RV20 / causal HAR future RV20 prediction)",
        "families": {
            "volatility_failure": list(VOLATILITY_FAILURE_COLUMNS),
            "har_aware": list(HAR_AWARE_COLUMNS),
        },
        "lookback": LOOKBACK,
        "esn_config": ESN_CONFIG,
        "seeds": list(SEEDS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "protocol": protocol.__dict__,
        "har_aware_causality": "forecast revisions and errors from forecasts made 20 rows earlier only",
    }, indent=2) + "\n")

    print("Summary metrics:")
    print(summary.to_string(index=False))
    print("\nOutcome metrics for HAR, linear path, and ensembles:")
    keep_models = [
        "har_zero_residual",
        "full_path_linear",
        "reset_esn_ensemble",
        "continuous_esn_ensemble",
    ]
    print(
        by_outcome[by_outcome["model"].isin(keep_models)]
        .sort_values(["family", "outcome", "future_rv_rmse"])
        .to_string(index=False)
    )
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
