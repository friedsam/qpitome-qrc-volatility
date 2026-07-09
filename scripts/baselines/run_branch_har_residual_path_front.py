"""Quickshot: predict causal HAR residuals from matched branch-path representations.

The target is the continuous log residual

    log(actual future RV20 / causal HAR future RV20 prediction)

for every eligible branch episode, including mixed outcomes. Predictions are
added back to HAR and scored with continuous volatility metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.branch_path_reservoir import (
    PATH_COLUMNS,
    add_causal_path_channels,
    continuous_esn_daily_states,
    episode_features_from_daily_states,
    extract_episode_windows,
    flattened_path_features,
    reset_esn_features,
)
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path("results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv")
DEFAULT_HAR = Path("results/baselines/branch_har_front_v1/har_episode_features.csv")
DEFAULT_OUTPUT = Path("results/baselines/branch_har_residual_path_front_v1")
LOOKBACK = 40
SEEDS = (0, 1, 2)
ESN_CONFIG = dict(n_reservoir=50, spectral_radius=0.9, input_scale=0.3, leak=0.3)
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    p.add_argument("--har-features", type=Path, default=DEFAULT_HAR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    yhat = np.maximum(np.asarray(yhat, dtype=float), 1e-12)
    ratio = y / yhat
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mz_fit(y: np.ndarray, yhat: np.ndarray) -> tuple[float, float]:
    x = np.column_stack([np.ones(len(yhat)), yhat])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return float(coef[0]), float(coef[1])


def fit_residual(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[float, float]:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS)),
    ])
    model.fit(x_train, y_train)
    pred = float(model.predict(x_test)[0])
    alpha = float(model.named_steps["ridge"].alpha_)
    return pred, alpha


def metric_row(model: str, g: pd.DataFrame) -> dict[str, object]:
    y = g["actual_future_rv20"].to_numpy(float)
    yhat = g["combined_future_rv20_pred"].to_numpy(float)
    residual = g["har_log_residual_actual"].to_numpy(float)
    residual_hat = g["har_log_residual_pred"].to_numpy(float)
    sse = float(np.sum((residual - residual_hat) ** 2))
    sse_zero = float(np.sum(residual**2))
    intercept, slope = mz_fit(y, yhat)
    return {
        "model": model,
        "n_oos": len(g),
        "future_rv_rmse": float(np.sqrt(np.mean((yhat - y) ** 2))),
        "future_rv_mae": float(np.mean(np.abs(yhat - y))),
        "qlike": qlike(y, yhat),
        "mz_intercept": intercept,
        "mz_slope": slope,
        "har_residual_rmse": float(np.sqrt(np.mean((residual_hat - residual) ** 2))),
        "residual_r2_vs_har": 1.0 - sse / sse_zero if sse_zero > 0 else np.nan,
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
    if episodes[["har_future_rv20_pred", "actual_future_rv20", "har_log_residual_actual"]].isna().any().any():
        raise ValueError("Missing HAR forecasts or continuous targets")

    channels = add_causal_path_channels(daily)
    ids, windows = extract_episode_windows(channels, episodes, LOOKBACK)
    if len(ids) != len(episodes):
        raise RuntimeError(f"Only {len(ids)} of {len(episodes)} episodes have complete path windows")

    flat = flattened_path_features(windows)
    feature_maps: dict[str, dict[int, np.ndarray]] = {
        "full_path_linear": {int(i): flat[k] for k, i in enumerate(ids)}
    }
    for seed in SEEDS:
        reset = reset_esn_features(windows, seed=seed, **ESN_CONFIG)
        feature_maps[f"reset_esn_seed{seed}"] = {int(i): reset[k] for k, i in enumerate(ids)}
        daily_states = continuous_esn_daily_states(channels, seed=seed, **ESN_CONFIG)
        c_ids, cont = episode_features_from_daily_states(daily_states, episodes)
        feature_maps[f"continuous_esn_seed{seed}"] = {
            int(i): cont[k] for k, i in enumerate(c_ids)
        }

    protocol = EpisodePrequentialConfig(
        min_train_episodes=18,
        outcome_horizon_rows=40,
        embargo_rows=0,
    )
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

        # HAR itself is the zero-residual reference prediction.
        rows.append({
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

        for model_name, fmap in feature_maps.items():
            x_train = np.vstack([fmap[i] for i in train_ids])
            x_test = fmap[test_id][None, :]
            pred, alpha = fit_residual(x_train, y_train, x_test)
            rows.append({
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

    # Fixed-seed ensembles average residual predictions before reconstruction.
    for family in ("reset_esn", "continuous_esn"):
        seed_models = [f"{family}_seed{s}" for s in SEEDS]
        subset = pred[pred["model"].isin(seed_models)]
        wide = subset.pivot(index="episode_id", columns="model", values="har_log_residual_pred")
        ensemble = wide.mean(axis=1)
        base = pred[pred["model"] == seed_models[0]].copy()
        base["model"] = f"{family}_ensemble"
        base["selected_alpha"] = np.nan
        base["har_log_residual_pred"] = base["episode_id"].map(ensemble)
        base["combined_future_rv20_pred"] = (
            base["har_future_rv20_pred"] * np.exp(base["har_log_residual_pred"])
        ).clip(lower=1e-8)
        pred = pd.concat([pred, base], ignore_index=True)

    summary = pd.DataFrame([
        metric_row(name, g) for name, g in pred.groupby("model")
    ]).sort_values(["future_rv_rmse", "qlike"])

    outcome_rows = []
    for (model, outcome), g in pred.groupby(["model", "outcome"]):
        row = metric_row(model, g)
        row["outcome"] = outcome
        outcome_rows.append(row)
    by_outcome = pd.DataFrame(outcome_rows)

    pred.to_csv(a.output / "oos_predictions.csv", index=False)
    summary.to_csv(a.output / "summary_metrics.csv", index=False)
    by_outcome.to_csv(a.output / "metrics_by_outcome.csv", index=False)
    pd.DataFrame({"channel": PATH_COLUMNS}).to_csv(a.output / "path_channels.csv", index=False)
    (a.output / "run_manifest.json").write_text(json.dumps({
        "status": "quickshot exploratory residual benchmark",
        "target": "log(actual future RV20 / causal HAR future RV20 prediction)",
        "n_expected_oos": 30,
        "lookback": LOOKBACK,
        "path_channels": list(PATH_COLUMNS),
        "models": ["HAR zero residual", "full path linear", "reset ESN", "continuous ESN"],
        "esn_config": ESN_CONFIG,
        "seeds": list(SEEDS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "ridge_selection": "training-only RidgeCV at each prequential step",
        "protocol": protocol.__dict__,
    }, indent=2) + "\n")

    print("Summary metrics:")
    print(summary.to_string(index=False))
    print("\nMetrics by outcome for HAR and ensembles:")
    keep = by_outcome[by_outcome["model"].isin([
        "har_zero_residual",
        "full_path_linear",
        "reset_esn_ensemble",
        "continuous_esn_ensemble",
    ])]
    print(keep.sort_values(["outcome", "future_rv_rmse"]).to_string(index=False))
    print("\nPer-episode residual predictions:")
    show = pred[pred["model"].isin([
        "har_zero_residual",
        "full_path_linear",
        "reset_esn_ensemble",
        "continuous_esn_ensemble",
    ])]
    print(show.pivot_table(
        index=["episode_id", "branch_date", "outcome", "har_log_residual_actual"],
        columns="model",
        values="har_log_residual_pred",
    ).reset_index().to_string(index=False))
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
