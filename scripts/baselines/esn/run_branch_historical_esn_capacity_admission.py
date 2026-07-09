"""Admission test for frozen historical NumPy ESN reservoir configurations.

This is not a hyperparameter search and not an exact replay of the historical
volatility-regression model. It reuses the four frozen historical reservoir
configurations while holding the branch-classification readout fixed, so the
only question is whether the current 50-unit branch ESN failed a basic capacity
admission test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.branch_path_reservoir import (
    add_causal_path_channels,
    extract_episode_windows,
    reset_esn_features,
)
from qpitome_qrc.baselines.numpy_esn import historical_numpy_esn_grid
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)

LOOKBACK = 40
SEEDS = (0, 1, 2)
READOUT_C = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def family_id(config: dict[str, object]) -> str:
    return (
        f"n{config['n']}_sr{config['sr']}_inp{config['inp']}_"
        f"leak{config['leak']}_histalpha{config['alpha']}"
    )


def fit_probability(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=READOUT_C, max_iter=5000, solver="lbfgs")),
    ])
    model.fit(x_train, y_train)
    return float(np.clip(model.predict_proba(x_test)[0, 1], 1e-6, 1 - 1e-6))


def metric_row(model: str, period: str, g: pd.DataFrame) -> dict[str, object]:
    y = g["y_true"].to_numpy(int)
    p = g["p_recovery"].to_numpy(float)
    return {
        "model": model,
        "period": period,
        "n_oos": len(g),
        "n_recovery": int(y.sum()),
        "n_relapse": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "pr_auc_recovery": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_p_recovery": float(p.mean()),
    }


def main() -> None:
    a = parse_args()
    for path in (a.data, a.episodes):
        if not path.exists():
            raise FileNotFoundError(path)
    a.output.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(a.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(a.episodes, parse_dates=["branch_date"]).sort_values("branch_date").reset_index(drop=True)

    channels = add_causal_path_channels(daily)
    ids, windows = extract_episode_windows(channels, episodes, LOOKBACK)
    if len(ids) != len(episodes):
        raise RuntimeError(f"Only {len(ids)} of {len(episodes)} episodes have complete path windows")

    grid = historical_numpy_esn_grid(list(SEEDS))
    feature_maps: dict[str, dict[int, np.ndarray]] = {}
    families: dict[str, list[str]] = {}

    for config in grid:
        family = family_id(config)
        model_name = f"{family}_seed{config['seed']}"
        print(f"Building {model_name}", flush=True)
        features = reset_esn_features(
            windows,
            n_reservoir=int(config["n"]),
            spectral_radius=float(config["sr"]),
            input_scale=float(config["inp"]),
            leak=float(config["leak"]),
            seed=int(config["seed"]),
        )
        feature_maps[model_name] = {int(i): features[row] for row, i in enumerate(ids)}
        families.setdefault(family, []).append(model_name)

    protocol = EpisodePrequentialConfig(min_train_episodes=18, outcome_horizon_rows=40, embargo_rows=0)
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    lookup = episodes.set_index("episode_id", drop=False)
    rows: list[dict[str, object]] = []

    for step in steps.itertuples(index=False):
        test_id = int(step.test_episode_id)
        test_ep = lookup.loc[test_id]
        if test_ep["outcome"] not in ("recovery", "relapse"):
            continue

        train_ids = assignments[
            (assignments["step"] == step.step) & (assignments["split"] == "train")
        ]["episode_id"].astype(int)
        train_ids = [i for i in train_ids if lookup.loc[i, "outcome"] in ("recovery", "relapse")]
        y_train = np.asarray([1 if lookup.loc[i, "outcome"] == "recovery" else 0 for i in train_ids])
        y_true = 1 if test_ep["outcome"] == "recovery" else 0

        for model_name, fmap in feature_maps.items():
            rows.append({
                "step": int(step.step),
                "episode_id": test_id,
                "branch_date": test_ep["branch_date"],
                "outcome": test_ep["outcome"],
                "y_true": y_true,
                "n_train_binary": len(train_ids),
                "model": model_name,
                "p_recovery": fit_probability(
                    np.vstack([fmap[i] for i in train_ids]),
                    y_train,
                    fmap[test_id][None, :],
                ),
            })

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError("No binary OOS predictions were produced")

    for family, seed_models in families.items():
        wide = pred[pred["model"].isin(seed_models)].pivot(
            index="episode_id", columns="model", values="p_recovery"
        )
        ensemble = wide.mean(axis=1)
        base = pred[pred["model"] == seed_models[0]].copy()
        base["model"] = f"{family}_ensemble"
        base["p_recovery"] = base["episode_id"].map(ensemble)
        pred = pd.concat([pred, base], ignore_index=True)

    periods = {
        "all": lambda g: g,
        "2000_plus": lambda g: g[g["branch_date"] >= pd.Timestamp("2000-01-01")],
        "2010_plus": lambda g: g[g["branch_date"] >= pd.Timestamp("2010-01-01")],
    }
    metrics: list[dict[str, object]] = []
    for model, g in pred.groupby("model"):
        for period, selector in periods.items():
            h = selector(g)
            if len(h):
                metrics.append(metric_row(model, period, h))

    summary = pd.DataFrame(metrics).sort_values(["period", "brier", "log_loss"])
    pred.to_csv(a.output / "oos_predictions.csv", index=False)
    summary.to_csv(a.output / "summary_metrics_by_period.csv", index=False)
    steps.to_csv(a.output / "protocol_steps.csv", index=False)
    (a.output / "run_manifest.json").write_text(json.dumps({
        "task": "binary recovery versus relapse",
        "purpose": "historical reservoir capacity admission test",
        "data": str(a.data),
        "episodes": str(a.episodes),
        "lookback": LOOKBACK,
        "historical_reservoir_grid": grid,
        "classification_readout": {"model": "StandardScaler + LogisticRegression", "C": READOUT_C},
        "note": "historical alpha values identify frozen configurations but are not used by the classification readout",
        "protocol": protocol.__dict__,
        "tuning": "none",
    }, indent=2) + "\n")

    print(f"\nComplete episodes: {len(episodes)}")
    print("Outcome counts:")
    print(episodes["outcome"].value_counts().to_string())
    print(f"Binary OOS episodes: {pred[pred['model'].str.endswith('_ensemble')]['episode_id'].nunique()}")
    print("\nHistorical-grid ensemble metrics:")
    print(summary[summary["model"].str.endswith("_ensemble")].to_string(index=False))
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
