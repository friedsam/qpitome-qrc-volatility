"""Compare matched full-path linear, reset ESN, and continuous-state ESN models."""

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
    PATH_COLUMNS,
    add_causal_path_channels,
    continuous_esn_daily_states,
    episode_features_from_daily_states,
    extract_episode_windows,
    flattened_path_features,
    reset_esn_features,
)
from qpitome_qrc.evaluation.episode_prequential import EpisodePrequentialConfig, make_episode_prequential_steps

DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path("results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv")
DEFAULT_OUTPUT = Path("results/baselines/branch_path_reservoir_front_v1")
SEEDS = (0, 1, 2)
ESN_CONFIG = dict(n_reservoir=50, spectral_radius=0.9, input_scale=0.3, leak=0.3)
LOOKBACK = 40
READOUT_C = 0.1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p.parse_args()


def fit_probability(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> float:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=READOUT_C, max_iter=5000, solver="lbfgs")),
    ])
    model.fit(x_train, y_train)
    return float(np.clip(model.predict_proba(x_test)[0, 1], 1e-6, 1 - 1e-6))


def metrics(model: str, g: pd.DataFrame) -> dict[str, object]:
    y = g["y_true"].to_numpy(int)
    p = g["p_recovery"].to_numpy(float)
    return {
        "model": model,
        "n_oos": len(g),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_recovery": average_precision_score(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "mean_p_recovery": float(p.mean()),
    }


def main() -> None:
    a = parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(a.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(a.episodes, parse_dates=["branch_date"])
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
        feature_maps[f"continuous_esn_seed{seed}"] = {int(i): cont[k] for k, i in enumerate(c_ids)}

    protocol = EpisodePrequentialConfig(min_train_episodes=18, outcome_horizon_rows=40, embargo_rows=0)
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    lookup = episodes.set_index("episode_id", drop=False)
    rows: list[dict[str, object]] = []

    for step in steps.itertuples(index=False):
        test_id = int(step.test_episode_id)
        test_ep = lookup.loc[test_id]
        if test_ep["outcome"] not in ("recovery", "relapse"):
            continue
        train_ids = assignments[(assignments["step"] == step.step) & (assignments["split"] == "train")]["episode_id"].astype(int)
        train_ids = [i for i in train_ids if lookup.loc[i, "outcome"] in ("recovery", "relapse")]
        y_train = np.asarray([1 if lookup.loc[i, "outcome"] == "recovery" else 0 for i in train_ids])
        y_true = 1 if test_ep["outcome"] == "recovery" else 0
        for model_name, fmap in feature_maps.items():
            x_train = np.vstack([fmap[i] for i in train_ids])
            x_test = fmap[test_id][None, :]
            rows.append({
                "step": int(step.step),
                "episode_id": test_id,
                "branch_date": test_ep["branch_date"],
                "outcome": test_ep["outcome"],
                "y_true": y_true,
                "n_train_binary": len(train_ids),
                "model": model_name,
                "p_recovery": fit_probability(x_train, y_train, x_test),
            })

    pred = pd.DataFrame(rows)

    # Fixed-seed ensemble probabilities are averages, not selected seeds.
    for family in ("reset_esn", "continuous_esn"):
        seed_models = [f"{family}_seed{s}" for s in SEEDS]
        wide = pred[pred["model"].isin(seed_models)].pivot(index="episode_id", columns="model", values="p_recovery")
        ens = wide.mean(axis=1)
        base = pred[pred["model"] == seed_models[0]].copy()
        base["model"] = f"{family}_ensemble"
        base["p_recovery"] = base["episode_id"].map(ens)
        pred = pd.concat([pred, base], ignore_index=True)

    summary = pd.DataFrame([metrics(name, g) for name, g in pred.groupby("model")]).sort_values(["brier", "log_loss"])
    pred.to_csv(a.output / "oos_predictions.csv", index=False)
    summary.to_csv(a.output / "summary_metrics.csv", index=False)
    pd.DataFrame({"channel": PATH_COLUMNS}).to_csv(a.output / "path_channels.csv", index=False)
    (a.output / "run_manifest.json").write_text(json.dumps({
        "task": "binary recovery versus relapse",
        "lookback": LOOKBACK,
        "path_channels": list(PATH_COLUMNS),
        "models": ["full_path_linear", "reset_esn", "continuous_esn"],
        "esn_config": ESN_CONFIG,
        "seeds": list(SEEDS),
        "seed_policy": "fixed a priori; report individuals and unweighted probability ensemble",
        "readout": {"model": "StandardScaler + LogisticRegression", "C": READOUT_C},
        "protocol": protocol.__dict__,
        "historical_esn_module_modified": False,
    }, indent=2) + "\n")

    print("Summary metrics:")
    print(summary.to_string(index=False))
    print("\nPer-episode ensemble comparison:")
    show = pred[pred["model"].isin(["full_path_linear", "reset_esn_ensemble", "continuous_esn_ensemble"])]
    print(show.pivot_table(index=["episode_id", "branch_date", "outcome"], columns="model", values="p_recovery").reset_index().to_string(index=False))
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
