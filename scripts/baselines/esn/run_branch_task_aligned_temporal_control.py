"""Matched compact classical temporal control for task-aligned branch paths.

This is the admission control before TFIM/Rydberg QRC.

Models:

1. frozen state_plus_motion baseline
2. baseline + 8-unit ESN correction on the true ordered 4-channel path
3. baseline + the same ESN correction on a 5-day block-permuted path

The block permutation preserves all daily 4-channel tuples and within-block local
order while destroying their 40-day arrangement. The same fixed permutation is
used for all ESN seeds. No architecture or penalty search is performed.
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

from qpitome_qrc.baselines.branch_path_reservoir import reset_esn_features
from qpitome_qrc.baselines.branch_probabilistic import FEATURE_SETS
from qpitome_qrc.baselines.branch_residual_correction import (
    OffsetCorrectionConfig,
    fit_offset_logit_correction,
    logit_from_probability,
)
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)
from qpitome_qrc.regimes.branch_transition_diagnostics import permute_transition_windows
from qpitome_qrc.regimes.branch_transition_path import (
    add_transition_path_channels,
    extract_transition_windows,
)

LOOKBACK = 40
SEEDS = (0, 1, 2)
BLOCK_SIZE = 5
PERMUTATION_SEED = 20260709
STATE_MOTION = FEATURE_SETS["state_plus_motion"]
BASELINE_C = 1.0
ESN_CONFIG = dict(
    n_reservoir=8,
    spectral_radius=0.9,
    input_scale=0.3,
    leak=0.3,
)
CORRECTION_CONFIG = OffsetCorrectionConfig(l2_penalty=10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fit_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, float]:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=BASELINE_C,
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    p_train = np.clip(model.predict_proba(x_train)[:, 1], 1e-6, 1.0 - 1e-6)
    p_test = float(np.clip(model.predict_proba(x_test)[0, 1], 1e-6, 1.0 - 1e-6))
    return p_train, p_test


def metric_row(model: str, period: str, frame: pd.DataFrame) -> dict[str, object]:
    y = frame["y_true"].to_numpy(dtype=int)
    p = frame["p_recovery"].to_numpy(dtype=float)
    return {
        "model": model,
        "period": period,
        "n_oos": len(frame),
        "n_recovery": int(y.sum()),
        "n_relapse": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "pr_auc_recovery": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_p_recovery": float(p.mean()),
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

    featured = episodes.copy()
    featured["stress_ratio"] = featured["rv_20d"] / featured["branch_stress_cut"]
    missing = set(STATE_MOTION) - set(featured.columns)
    if missing:
        raise KeyError(f"Missing state_plus_motion features: {sorted(missing)}")
    featured = featured.set_index("episode_id", drop=False)

    channels = add_transition_path_channels(daily)
    episode_ids, ordered_windows = extract_transition_windows(
        channels,
        episodes,
        lookback=LOOKBACK,
    )
    if len(episode_ids) != len(episodes):
        raise RuntimeError(
            f"Only {len(episode_ids)} of {len(episodes)} episodes have complete windows"
        )

    permuted_windows = permute_transition_windows(
        ordered_windows,
        seed=PERMUTATION_SEED,
        block_size=BLOCK_SIZE,
    )

    feature_maps: dict[str, dict[int, dict[int, np.ndarray]]] = {
        "ordered": {},
        "block_permuted": {},
    }
    for seed in SEEDS:
        for path_name, windows in (
            ("ordered", ordered_windows),
            ("block_permuted", permuted_windows),
        ):
            features = reset_esn_features(windows, seed=seed, **ESN_CONFIG)
            hidden = features[:, : ESN_CONFIG["n_reservoir"]]
            feature_maps[path_name][seed] = {
                int(episode_id): hidden[row]
                for row, episode_id in enumerate(episode_ids)
            }

    protocol = EpisodePrequentialConfig(
        min_train_episodes=18,
        outcome_horizon_rows=40,
        embargo_rows=0,
    )
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    rows: list[dict[str, object]] = []

    for step in steps.itertuples(index=False):
        test_id = int(step.test_episode_id)
        test_episode = featured.loc[test_id]
        if test_episode["outcome"] not in ("recovery", "relapse"):
            continue

        train_ids = assignments[
            (assignments["step"] == step.step)
            & (assignments["split"] == "train")
        ]["episode_id"].astype(int)
        train_ids = [
            episode_id
            for episode_id in train_ids
            if featured.loc[episode_id, "outcome"] in ("recovery", "relapse")
        ]

        y_train = np.asarray(
            [
                1 if featured.loc[episode_id, "outcome"] == "recovery" else 0
                for episode_id in train_ids
            ],
            dtype=int,
        )
        y_true = 1 if test_episode["outcome"] == "recovery" else 0

        x_baseline_train = featured.loc[
            train_ids, list(STATE_MOTION)
        ].to_numpy(dtype=float)
        x_baseline_test = featured.loc[
            [test_id], list(STATE_MOTION)
        ].to_numpy(dtype=float)

        p_baseline_train, p_baseline_test = fit_baseline(
            x_baseline_train,
            y_train,
            x_baseline_test,
        )
        baseline_train_logit = logit_from_probability(p_baseline_train)
        baseline_test_logit = float(logit_from_probability([p_baseline_test])[0])

        rows.append(
            {
                "step": int(step.step),
                "episode_id": test_id,
                "branch_date": test_episode["branch_date"],
                "outcome": test_episode["outcome"],
                "y_true": y_true,
                "n_train_binary": len(train_ids),
                "model": "state_plus_motion",
                "path_geometry": "none",
                "seed": np.nan,
                "correction_l2_norm": 0.0,
                "p_baseline": p_baseline_test,
                "p_recovery": p_baseline_test,
            }
        )

        for path_name in ("ordered", "block_permuted"):
            for seed in SEEDS:
                train_hidden = np.vstack(
                    [feature_maps[path_name][seed][episode_id] for episode_id in train_ids]
                )
                test_hidden = feature_maps[path_name][seed][test_id][None, :]
                p_corrected, beta = fit_offset_logit_correction(
                    train_hidden,
                    y_train,
                    baseline_train_logit,
                    test_hidden,
                    baseline_test_logit,
                    CORRECTION_CONFIG,
                )
                rows.append(
                    {
                        "step": int(step.step),
                        "episode_id": test_id,
                        "branch_date": test_episode["branch_date"],
                        "outcome": test_episode["outcome"],
                        "y_true": y_true,
                        "n_train_binary": len(train_ids),
                        "model": f"task_esn8_{path_name}_seed{seed}",
                        "path_geometry": path_name,
                        "seed": seed,
                        "correction_l2_norm": float(np.linalg.norm(beta)),
                        "p_baseline": p_baseline_test,
                        "p_recovery": p_corrected,
                    }
                )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No binary OOS predictions were produced")

    for path_name in ("ordered", "block_permuted"):
        seed_models = [f"task_esn8_{path_name}_seed{seed}" for seed in SEEDS]
        wide = predictions[predictions["model"].isin(seed_models)].pivot(
            index="episode_id",
            columns="model",
            values="p_recovery",
        )
        ensemble = wide.mean(axis=1)
        base = predictions[predictions["model"] == seed_models[0]].copy()
        base["model"] = f"task_esn8_{path_name}_ensemble"
        base["seed"] = np.nan
        base["p_recovery"] = base["episode_id"].map(ensemble)
        predictions = pd.concat([predictions, base], ignore_index=True)

    periods = {
        "all": lambda frame: frame,
        "2000_plus": lambda frame: frame[
            frame["branch_date"] >= pd.Timestamp("2000-01-01")
        ],
        "2010_plus": lambda frame: frame[
            frame["branch_date"] >= pd.Timestamp("2010-01-01")
        ],
    }
    metrics: list[dict[str, object]] = []
    for model, group in predictions.groupby("model"):
        for period, selector in periods.items():
            selected = selector(group)
            if len(selected):
                metrics.append(metric_row(model, period, selected))

    summary = pd.DataFrame(metrics).sort_values(["period", "brier", "log_loss"])
    predictions.to_csv(args.output / "oos_predictions.csv", index=False)
    summary.to_csv(args.output / "summary_metrics_by_period.csv", index=False)
    steps.to_csv(args.output / "protocol_steps.csv", index=False)
    (args.output / "run_manifest.json").write_text(
        json.dumps(
            {
                "claim": (
                    "task-aligned temporal ordering adds incremental information "
                    "beyond state_plus_motion"
                ),
                "falsifying_result": (
                    "ordered-path correction does not improve over baseline and/or "
                    "does not outperform the matched block-permuted correction"
                ),
                "data": str(args.data),
                "episodes": str(args.episodes),
                "lookback": LOOKBACK,
                "baseline_features": list(STATE_MOTION),
                "baseline_model": {
                    "pipeline": "StandardScaler + LogisticRegression",
                    "C": BASELINE_C,
                },
                "temporal_control": {
                    "reservoir": ESN_CONFIG,
                    "seeds": list(SEEDS),
                    "readout_dimension": ESN_CONFIG["n_reservoir"],
                    "correction_geometry": "frozen baseline logit + no-intercept correction",
                    "correction_l2_penalty": CORRECTION_CONFIG.l2_penalty,
                },
                "ordering_ablation": {
                    "block_size": BLOCK_SIZE,
                    "permutation_seed": PERMUTATION_SEED,
                    "preserves": "daily four-channel tuples and within-block order",
                    "destroys": "40-day block arrangement",
                },
                "protocol": protocol.__dict__,
                "tuning": "none",
            },
            indent=2,
        )
        + "\n"
    )

    primary_models = [
        "state_plus_motion",
        "task_esn8_ordered_ensemble",
        "task_esn8_block_permuted_ensemble",
    ]
    print(f"Complete episodes: {len(episodes)}")
    print(f"Usable transition windows: {len(episode_ids)}")
    print(f"Binary OOS episodes: {predictions[predictions['model'] == 'state_plus_motion']['episode_id'].nunique()}")
    print("\nPrimary comparison:")
    print(summary[summary["model"].isin(primary_models)].to_string(index=False))

    print("\nMean correction norm by path geometry and seed:")
    correction_rows = predictions[predictions["seed"].notna()]
    print(
        correction_rows.groupby(["path_geometry", "seed"])["correction_l2_norm"]
        .mean()
        .to_string()
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
