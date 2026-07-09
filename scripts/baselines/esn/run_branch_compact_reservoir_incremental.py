"""Test whether a compact ESN representation adds to state-plus-motion.

Bounded models:

1. state_plus_motion
2. state_plus_motion + 5 train-only PCA components of a 50-unit reset ESN
3. state_plus_motion + 10 train-only PCA components of a 50-unit reset ESN

The primitive reservoir is frozen. PCA is refit inside every prequential training
step and never sees the test episode. No hyperparameter search is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.branch_path_reservoir import (
    add_causal_path_channels,
    extract_episode_windows,
    reset_esn_features,
)
from qpitome_qrc.baselines.branch_probabilistic import (
    FEATURE_SETS,
    add_branch_baseline_features,
)
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)

LOOKBACK = 40
SEEDS = (0, 1, 2)
PCA_DIMS = (5, 10)
STATE_MOTION = FEATURE_SETS["state_plus_motion"]
ESN_CONFIG = dict(
    n_reservoir=50,
    spectral_radius=0.9,
    input_scale=0.3,
    leak=0.3,
)
READOUT_C = 1.0  # exact state_plus_motion baseline default


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


def compact_reservoir_features(
    train_states: np.ndarray,
    test_state: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit scaler and PCA on training reservoir states only."""
    if len(train_states) <= n_components:
        raise ValueError(
            f"Need more than {n_components} training episodes for PCA; got {len(train_states)}"
        )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_states)
    test_scaled = scaler.transform(test_state[None, :])

    pca = PCA(n_components=n_components, svd_solver="full")
    train_pc = pca.fit_transform(train_scaled)
    test_pc = pca.transform(test_scaled)
    explained = float(pca.explained_variance_ratio_.sum())
    return train_pc, test_pc, explained


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
    featured = add_branch_baseline_features(episodes, daily).set_index("episode_id", drop=False)

    channels = add_causal_path_channels(daily)
    ids, windows = extract_episode_windows(channels, episodes, LOOKBACK)
    if len(ids) != len(episodes):
        raise RuntimeError(f"Only {len(ids)} of {len(episodes)} episodes have complete path windows")

    # Keep only the 50 reservoir coordinates. The final six raw input coordinates
    # are deliberately excluded because state_plus_motion already supplies the
    # explicit low-dimensional current-state control.
    state_maps: dict[int, dict[int, np.ndarray]] = {}
    for seed in SEEDS:
        features = reset_esn_features(windows, seed=seed, **ESN_CONFIG)
        states = features[:, : ESN_CONFIG["n_reservoir"]]
        state_maps[seed] = {int(i): states[row] for row, i in enumerate(ids)}

    protocol = EpisodePrequentialConfig(min_train_episodes=18, outcome_horizon_rows=40, embargo_rows=0)
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    rows: list[dict[str, object]] = []

    for step in steps.itertuples(index=False):
        test_id = int(step.test_episode_id)
        test_ep = featured.loc[test_id]
        if test_ep["outcome"] not in ("recovery", "relapse"):
            continue

        train_ids = assignments[
            (assignments["step"] == step.step) & (assignments["split"] == "train")
        ]["episode_id"].astype(int)
        train_ids = [i for i in train_ids if featured.loc[i, "outcome"] in ("recovery", "relapse")]

        y_train = np.asarray([1 if featured.loc[i, "outcome"] == "recovery" else 0 for i in train_ids])
        y_true = 1 if test_ep["outcome"] == "recovery" else 0
        x_state_train = featured.loc[train_ids, list(STATE_MOTION)].to_numpy(dtype=float)
        x_state_test = featured.loc[[test_id], list(STATE_MOTION)].to_numpy(dtype=float)

        rows.append({
            "step": int(step.step),
            "episode_id": test_id,
            "branch_date": test_ep["branch_date"],
            "outcome": test_ep["outcome"],
            "y_true": y_true,
            "n_train_binary": len(train_ids),
            "model": "state_plus_motion",
            "seed": np.nan,
            "pca_dim": 0,
            "pca_explained_variance": np.nan,
            "p_recovery": fit_probability(x_state_train, y_train, x_state_test),
        })

        for seed in SEEDS:
            train_states = np.vstack([state_maps[seed][i] for i in train_ids])
            test_state = state_maps[seed][test_id]
            for n_components in PCA_DIMS:
                train_pc, test_pc, explained = compact_reservoir_features(
                    train_states, test_state, n_components
                )
                model_name = f"state_motion_plus_esn_pca{n_components}_seed{seed}"
                rows.append({
                    "step": int(step.step),
                    "episode_id": test_id,
                    "branch_date": test_ep["branch_date"],
                    "outcome": test_ep["outcome"],
                    "y_true": y_true,
                    "n_train_binary": len(train_ids),
                    "model": model_name,
                    "seed": seed,
                    "pca_dim": n_components,
                    "pca_explained_variance": explained,
                    "p_recovery": fit_probability(
                        np.hstack([x_state_train, train_pc]),
                        y_train,
                        np.hstack([x_state_test, test_pc]),
                    ),
                })

    pred = pd.DataFrame(rows)
    if pred.empty:
        raise RuntimeError("No binary OOS predictions were produced")

    for n_components in PCA_DIMS:
        seed_models = [f"state_motion_plus_esn_pca{n_components}_seed{s}" for s in SEEDS]
        wide = pred[pred["model"].isin(seed_models)].pivot(
            index="episode_id", columns="model", values="p_recovery"
        )
        ensemble = wide.mean(axis=1)
        base = pred[pred["model"] == seed_models[0]].copy()
        base["model"] = f"state_motion_plus_esn_pca{n_components}_ensemble"
        base["seed"] = np.nan
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
        "purpose": "compact reservoir incremental-information test",
        "data": str(a.data),
        "episodes": str(a.episodes),
        "lookback": LOOKBACK,
        "baseline_features": list(STATE_MOTION),
        "esn_config": ESN_CONFIG,
        "seeds": list(SEEDS),
        "pca_dims": list(PCA_DIMS),
        "pca_geometry": "StandardScaler + PCA fit on eligible binary training reservoir states inside each prequential step",
        "classification_readout": {"model": "StandardScaler + LogisticRegression", "C": READOUT_C},
        "protocol": protocol.__dict__,
        "tuning": "none; bounded 5/10 component test",
    }, indent=2) + "\n")

    print(f"Complete episodes: {len(episodes)}")
    print("Outcome counts:")
    print(episodes["outcome"].value_counts().to_string())
    print(f"Binary OOS episodes: {pred[pred['model'] == 'state_plus_motion']['episode_id'].nunique()}")
    print("\nPrimary comparison:")
    primary = summary[
        summary["model"].isin([
            "state_plus_motion",
            "state_motion_plus_esn_pca5_ensemble",
            "state_motion_plus_esn_pca10_ensemble",
        ])
    ]
    print(primary.to_string(index=False))
    print("\nMean PCA explained variance by seed/dimension:")
    print(
        pred[pred["pca_dim"] > 0]
        .groupby(["pca_dim", "seed"], dropna=True)["pca_explained_variance"]
        .mean()
        .to_string()
    )
    print(f"\nSaved: {a.output}")


if __name__ == "__main__":
    main()
