"""Replicate direct branch-resolution forecasting on the canonical 1950--2026 episodes.

Primary task
------------
Forecast recovery versus relapse from information available at the branch point.
Mixed episodes remain in chronological protocol geometry but are excluded from
binary fitting and scoring.

This is a frozen replication of the existing modern-sample comparison, not an
ESN architecture search. The reservoir remains the current primitive branch ESN:
one 50-unit configuration, three fixed seeds, and final-state representations.

Models
------
- historical recovery rate
- current state logistic regression
- state + motion logistic regression
- flattened 40-day path logistic regression
- reset-window ESN logistic regression
- continuous-state ESN logistic regression

The canonical long-history reconstruction must already exist. No branch detector,
outcome label, path horizon, reservoir setting, or readout hyperparameter is tuned
in this runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
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
from qpitome_qrc.baselines.branch_probabilistic import (
    binary_episode_frame,
    empirical_prior_probability,
)
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)


DEFAULT_DATA = Path(
    "results/regimes/long_history_branch_reconstruction_v2/"
    "gspc_daily_engineered_1950_2026.csv"
)
DEFAULT_EPISODES = Path(
    "results/regimes/long_history_branch_reconstruction_v2/"
    "branch_episodes_1950_2026.csv"
)
DEFAULT_OUTPUT = Path(
    "results/baselines/long_history_branch_resolution_replication_v1"
)

LOOKBACK = 40
SEEDS = (0, 1, 2)
ESN_CONFIG = dict(
    n_reservoir=50,
    spectral_radius=0.9,
    input_scale=0.3,
    leak=0.3,
)
READOUT_C = 0.1
EXPECTED_COMPLETE_EPISODES = 80

CURRENT_STATE_COLUMNS = (
    "stress_ratio",
    "drawdown_120d",
    "rv_ratio_5_20_branch",
    "return_5d_branch",
)
STATE_PLUS_MOTION_COLUMNS = (
    *CURRENT_STATE_COLUMNS,
    "rv_5d_change_5d_branch",
    "worst_return_5d_in_prior_window",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fit_probability(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> float:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(
                    C=READOUT_C,
                    max_iter=5000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    return float(np.clip(model.predict_proba(x_test)[0, 1], 1e-6, 1.0 - 1e-6))


def metric_row(model: str, pred: pd.DataFrame) -> dict[str, object]:
    y = pred["y_true"].to_numpy(dtype=int)
    p = pred["p_recovery"].to_numpy(dtype=float)
    return {
        "model": model,
        "n_oos": len(pred),
        "n_recovery": int(y.sum()),
        "n_relapse": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_recovery": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "accuracy_0p5": float(accuracy_score(y, p >= 0.5)),
        "mean_p_recovery": float(np.mean(p)),
    }


def add_state_features(episodes: pd.DataFrame) -> pd.DataFrame:
    required = {
        "episode_id",
        "branch_idx",
        "branch_date",
        "outcome",
        "rv_20d",
        "branch_stress_cut",
        "drawdown_120d",
        "rv_ratio_5_20_branch",
        "rv_5d_change_5d_branch",
        "return_5d_branch",
        "worst_return_5d_in_prior_window",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise KeyError(f"Missing episode columns: {sorted(missing)}")

    out = episodes.copy()
    out["stress_ratio"] = out["rv_20d"] / out["branch_stress_cut"]
    return out


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
    if len(episodes) != EXPECTED_COMPLETE_EPISODES:
        raise AssertionError(
            f"Expected {EXPECTED_COMPLETE_EPISODES} canonical complete episodes; "
            f"found {len(episodes)}"
        )
    episodes = add_state_features(episodes)

    channels = add_causal_path_channels(daily)
    ids, windows = extract_episode_windows(channels, episodes, LOOKBACK)
    if len(ids) != len(episodes):
        raise RuntimeError(
            f"Only {len(ids)} of {len(episodes)} canonical episodes have complete path windows"
        )

    feature_maps: dict[str, dict[int, np.ndarray]] = {
        "full_path_linear": {
            int(episode_id): flattened_path_features(windows)[row]
            for row, episode_id in enumerate(ids)
        }
    }
    for seed in SEEDS:
        reset = reset_esn_features(windows, seed=seed, **ESN_CONFIG)
        feature_maps[f"reset_esn_seed{seed}"] = {
            int(episode_id): reset[row]
            for row, episode_id in enumerate(ids)
        }

        daily_states = continuous_esn_daily_states(
            channels,
            seed=seed,
            **ESN_CONFIG,
        )
        continuous_ids, continuous = episode_features_from_daily_states(
            daily_states,
            episodes,
        )
        feature_maps[f"continuous_esn_seed{seed}"] = {
            int(episode_id): continuous[row]
            for row, episode_id in enumerate(continuous_ids)
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
        test_all = lookup.loc[[test_id]].copy()
        if test_all.iloc[0]["outcome"] not in ("recovery", "relapse"):
            continue

        train_ids_all = assignments[
            (assignments["step"] == step.step)
            & (assignments["split"] == "train")
        ]["episode_id"].astype(int)
        train_all = lookup.loc[train_ids_all].copy()
        train = binary_episode_frame(train_all)
        test = binary_episode_frame(test_all)

        y_train = train["y_recovery"].to_numpy(dtype=int)
        y_true = int(test.iloc[0]["y_recovery"])
        train_binary_ids = train["episode_id"].astype(int).tolist()

        base = {
            "step": int(step.step),
            "episode_id": test_id,
            "branch_date": test.iloc[0]["branch_date"],
            "outcome": test.iloc[0]["outcome"],
            "y_true": y_true,
            "n_train_all": len(train_all),
            "n_train_binary": len(train),
            "n_train_recovery": int(y_train.sum()),
            "n_train_relapse": int((1 - y_train).sum()),
            "leakage_gap_rows": int(step.leakage_gap_rows),
        }

        rows.append(
            {
                **base,
                "model": "historical_class_rate",
                "p_recovery": empirical_prior_probability(train),
            }
        )

        for model_name, columns in (
            ("current_state", CURRENT_STATE_COLUMNS),
            ("state_plus_motion", STATE_PLUS_MOTION_COLUMNS),
        ):
            rows.append(
                {
                    **base,
                    "model": model_name,
                    "p_recovery": fit_probability(
                        train.loc[:, columns].to_numpy(dtype=float),
                        y_train,
                        test.loc[:, columns].to_numpy(dtype=float),
                    ),
                }
            )

        for model_name, feature_map in feature_maps.items():
            rows.append(
                {
                    **base,
                    "model": model_name,
                    "p_recovery": fit_probability(
                        np.vstack([feature_map[i] for i in train_binary_ids]),
                        y_train,
                        feature_map[test_id][None, :],
                    ),
                }
            )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No binary OOS predictions were produced")

    for reservoir_family in ("reset_esn", "continuous_esn"):
        seed_models = [f"{reservoir_family}_seed{seed}" for seed in SEEDS]
        subset = predictions[predictions["model"].isin(seed_models)]
        wide = subset.pivot(
            index="episode_id",
            columns="model",
            values="p_recovery",
        )
        ensemble = wide.mean(axis=1)
        base = predictions[predictions["model"] == seed_models[0]].copy()
        base["model"] = f"{reservoir_family}_ensemble"
        base["p_recovery"] = base["episode_id"].map(ensemble)
        predictions = pd.concat([predictions, base], ignore_index=True)

    metrics = pd.DataFrame(
        [metric_row(model, group) for model, group in predictions.groupby("model")]
    ).sort_values(["brier", "log_loss"]).reset_index(drop=True)

    predictions.to_csv(args.output / "oos_predictions.csv", index=False)
    metrics.to_csv(args.output / "summary_metrics.csv", index=False)
    steps.to_csv(args.output / "protocol_steps.csv", index=False)
    pd.DataFrame({"channel": PATH_COLUMNS}).to_csv(
        args.output / "path_channels.csv",
        index=False,
    )

    manifest = {
        "status": "frozen direct branch-resolution replication on canonical long history",
        "primary_task": "forecast recovery versus relapse at the branch point",
        "data": str(args.data),
        "episodes": str(args.episodes),
        "complete_episode_count": len(episodes),
        "mixed_handling": "retained in chronology; excluded from binary fitting and scoring",
        "state_models": {
            "current_state": list(CURRENT_STATE_COLUMNS),
            "state_plus_motion": list(STATE_PLUS_MOTION_COLUMNS),
        },
        "path_lookback": LOOKBACK,
        "path_channels": list(PATH_COLUMNS),
        "path_models": [
            "full_path_linear",
            "reset_esn",
            "continuous_esn",
        ],
        "esn_config": ESN_CONFIG,
        "seeds": list(SEEDS),
        "seed_policy": "fixed a priori; report individuals and unweighted probability ensemble",
        "readout": {
            "model": "StandardScaler + LogisticRegression",
            "C": READOUT_C,
        },
        "protocol": protocol.__dict__,
        "qualification": (
            "This is a replication of the existing primitive branch ESN, not an improved ESN architecture search."
        ),
    }
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"Canonical long episodes: {len(episodes)}")
    print("Outcome counts:")
    print(episodes["outcome"].value_counts().to_string())
    print(f"Binary OOS episodes: {predictions['episode_id'].nunique()}")
    first = predictions.groupby("episode_id", sort=False).first()
    print(
        "Binary OOS class counts: "
        f"recovery={(first['y_true'] == 1).sum()}, "
        f"relapse={(first['y_true'] == 0).sum()}"
    )
    print("\nSummary metrics:")
    print(metrics.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
