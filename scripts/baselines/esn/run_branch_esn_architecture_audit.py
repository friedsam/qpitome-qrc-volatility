"""Run the bounded branch ESN architecture audit on one episode dataset.

Variants are fixed a priori:

1. primitive frozen reset ESN
2. fixed normalized inputs
3. fixed normalized inputs + constant bias channel
4. fixed normalized inputs + bias + mean reservoir-state summary

This runner does not search hyperparameters.
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

from qpitome_qrc.baselines.branch_esn_audit import reset_esn_audit_features
from qpitome_qrc.baselines.branch_path_reservoir import (
    add_causal_path_channels,
    extract_episode_windows,
    reset_esn_features,
)
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)

LOOKBACK = 40
SEEDS = (0, 1, 2)
ESN_CONFIG = dict(n_reservoir=50, spectral_radius=0.9, input_scale=0.3, leak=0.3)
READOUT_C = 0.1

VARIANTS = {
    "primitive": dict(normalized=False, bias=False, trajectory_mean=False),
    "normalized": dict(normalized=True, bias=False, trajectory_mean=False),
    "normalized_bias": dict(normalized=True, bias=True, trajectory_mean=False),
    "normalized_bias_mean": dict(normalized=True, bias=True, trajectory_mean=True),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


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
    row: dict[str, object] = {
        "model": model,
        "period": period,
        "n_oos": len(g),
        "n_recovery": int(y.sum()),
        "n_relapse": int((1 - y).sum()),
        "pr_auc_recovery": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_p_recovery": float(p.mean()),
    }
    row["roc_auc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan
    return row


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

    feature_maps: dict[str, dict[int, np.ndarray]] = {}
    for seed in SEEDS:
        primitive = reset_esn_features(windows, seed=seed, **ESN_CONFIG)
        feature_maps[f"primitive_seed{seed}"] = {
            int(i): primitive[row] for row, i in enumerate(ids)
        }

        for variant, config in VARIANTS.items():
            if variant == "primitive":
                continue
            features = reset_esn_audit_features(
                windows,
                seed=seed,
                **ESN_CONFIG,
                **config,
            )
            feature_maps[f"{variant}_seed{seed}"] = {
                int(i): features[row] for row, i in enumerate(ids)
            }

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

    for family in VARIANTS:
        seed_models = [f"{family}_seed{s}" for s in SEEDS]
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
            if len(h) == 0:
                continue
            metrics.append(metric_row(model, period, h))

    summary = pd.DataFrame(metrics).sort_values(["period", "brier", "log_loss"])
    pred.to_csv(a.output / "oos_predictions.csv", index=False)
    summary.to_csv(a.output / "summary_metrics_by_period.csv", index=False)
    steps.to_csv(a.output / "protocol_steps.csv", index=False)
    (a.output / "run_manifest.json").write_text(json.dumps({
        "task": "binary recovery versus relapse",
        "data": str(a.data),
        "episodes": str(a.episodes),
        "lookback": LOOKBACK,
        "variants": VARIANTS,
        "esn_config": ESN_CONFIG,
        "seeds": list(SEEDS),
        "readout": {"model": "StandardScaler + LogisticRegression", "C": READOUT_C},
        "protocol": protocol.__dict__,
        "tuning": "none; bounded architecture audit",
    }, indent=2) + "\n")

    print(f"Complete episodes: {len(episodes)}")
    print("Outcome counts:")
    print(episodes["outcome"].value_counts().to_string())
    print(f"Binary OOS episodes: {pred[pred['model'] == 'primitive_ensemble']['episode_id'].nunique()}")
    print("\nEnsemble metrics:")
    show = summary[summary["model"].str.endswith("_ensemble")]
    print(show.to_string(index=False))
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
