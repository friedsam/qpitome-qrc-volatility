"""Run one fixed TFIM ordering control on the branch-transition target."""

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

from qpitome_qrc.baselines.branch_probabilistic import FEATURE_SETS
from qpitome_qrc.baselines.branch_residual_correction import (
    OffsetCorrectionConfig,
    fit_offset_logit_correction,
    logit_from_probability,
)
from qpitome_qrc.evaluation.episode_prequential import EpisodePrequentialConfig, make_episode_prequential_steps
from qpitome_qrc.qrc.tfim_transition import TFIMTransitionConfig, tfim_transition_features
from qpitome_qrc.regimes.branch_transition_diagnostics import permute_transition_windows
from qpitome_qrc.regimes.branch_transition_path import add_transition_path_channels, extract_transition_windows

LOOKBACK = 40
BLOCK_SIZE = 5
PERMUTATION_SEED = 20260709
STATE_MOTION = FEATURE_SETS["state_plus_motion"]
TFIM_CONFIG = TFIMTransitionConfig()
CORRECTION_CONFIG = OffsetCorrectionConfig(l2_penalty=10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fit_baseline(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, float]:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")),
    ])
    model.fit(x_train, y_train)
    p_train = np.clip(model.predict_proba(x_train)[:, 1], 1e-6, 1.0 - 1e-6)
    p_test = float(np.clip(model.predict_proba(x_test)[0, 1], 1e-6, 1.0 - 1e-6))
    return p_train, p_test


def metrics(model: str, period: str, frame: pd.DataFrame) -> dict[str, object]:
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
    args.output.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(args.episodes, parse_dates=["branch_date"]).sort_values("branch_date").reset_index(drop=True)

    featured = episodes.copy()
    featured["stress_ratio"] = featured["rv_20d"] / featured["branch_stress_cut"]
    featured = featured.set_index("episode_id", drop=False)

    channels = add_transition_path_channels(daily)
    episode_ids, ordered_windows = extract_transition_windows(channels, episodes, lookback=LOOKBACK)
    if len(episode_ids) != len(episodes):
        raise RuntimeError(f"Only {len(episode_ids)} of {len(episodes)} episodes have complete windows")

    permuted_windows = permute_transition_windows(
        ordered_windows,
        seed=PERMUTATION_SEED,
        block_size=BLOCK_SIZE,
    )
    print("Computing ordered TFIM features...")
    ordered_features = tfim_transition_features(ordered_windows, TFIM_CONFIG)
    print("Computing 5-day block-permuted TFIM features...")
    permuted_features = tfim_transition_features(permuted_windows, TFIM_CONFIG)

    feature_maps = {}
    for name, matrix in (("ordered", ordered_features), ("block_permuted", permuted_features)):
        feature_maps[name] = {
            int(episode_id): matrix[row]
            for row, episode_id in enumerate(episode_ids)
        }

    protocol = EpisodePrequentialConfig(min_train_episodes=18, outcome_horizon_rows=40, embargo_rows=0)
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    rows: list[dict[str, object]] = []

    for step in steps.itertuples(index=False):
        test_id = int(step.test_episode_id)
        test = featured.loc[test_id]
        if test["outcome"] not in ("recovery", "relapse"):
            continue
        train_ids = assignments[(assignments["step"] == step.step) & (assignments["split"] == "train")]["episode_id"].astype(int)
        train_ids = [i for i in train_ids if featured.loc[i, "outcome"] in ("recovery", "relapse")]
        y_train = np.asarray([1 if featured.loc[i, "outcome"] == "recovery" else 0 for i in train_ids], dtype=int)
        y_true = 1 if test["outcome"] == "recovery" else 0

        xb_train = featured.loc[train_ids, list(STATE_MOTION)].to_numpy(dtype=float)
        xb_test = featured.loc[[test_id], list(STATE_MOTION)].to_numpy(dtype=float)
        p_train, p_test = fit_baseline(xb_train, y_train, xb_test)
        train_offset = logit_from_probability(p_train)
        test_offset = float(logit_from_probability([p_test])[0])

        common = {
            "step": int(step.step),
            "episode_id": test_id,
            "branch_date": test["branch_date"],
            "outcome": test["outcome"],
            "y_true": y_true,
            "n_train_binary": len(train_ids),
            "p_baseline": p_test,
        }
        rows.append({**common, "model": "state_plus_motion", "path_geometry": "none", "correction_l2_norm": 0.0, "p_recovery": p_test})

        for name in ("ordered", "block_permuted"):
            x_train = np.vstack([feature_maps[name][i] for i in train_ids])
            x_test = feature_maps[name][test_id][None, :]
            p_corrected, beta = fit_offset_logit_correction(
                x_train,
                y_train,
                train_offset,
                x_test,
                test_offset,
                CORRECTION_CONFIG,
            )
            rows.append({
                **common,
                "model": f"tfim8_{name}",
                "path_geometry": name,
                "correction_l2_norm": float(np.linalg.norm(beta)),
                "p_recovery": p_corrected,
            })

    predictions = pd.DataFrame(rows)
    periods = {
        "all": lambda f: f,
        "2000_plus": lambda f: f[f["branch_date"] >= pd.Timestamp("2000-01-01")],
        "2010_plus": lambda f: f[f["branch_date"] >= pd.Timestamp("2010-01-01")],
    }
    metric_rows = []
    for model, group in predictions.groupby("model"):
        for period, select in periods.items():
            selected = select(group)
            if len(selected):
                metric_rows.append(metrics(model, period, selected))
    summary = pd.DataFrame(metric_rows).sort_values(["period", "brier", "log_loss"])

    predictions.to_csv(args.output / "oos_predictions.csv", index=False)
    summary.to_csv(args.output / "summary_metrics_by_period.csv", index=False)
    steps.to_csv(args.output / "protocol_steps.csv", index=False)
    np.save(args.output / "ordered_tfim_features.npy", ordered_features)
    np.save(args.output / "block_permuted_tfim_features.npy", permuted_features)
    (args.output / "run_manifest.json").write_text(json.dumps({
        "claim": "fixed TFIM dynamics extract incremental information from ordered task-aligned paths",
        "falsifying_result": "ordered TFIM does not improve over baseline and/or matched block-permuted TFIM",
        "lookback": LOOKBACK,
        "ordering_ablation_block_size": BLOCK_SIZE,
        "permutation_seed": PERMUTATION_SEED,
        "tfim": TFIM_CONFIG.__dict__,
        "observable_count": 8,
        "correction_l2_penalty": CORRECTION_CONFIG.l2_penalty,
        "tuning": "none",
    }, indent=2) + "\n")

    delta = np.abs(ordered_features - permuted_features)
    print(f"Complete episodes: {len(episodes)}")
    print(f"Usable transition windows: {len(episode_ids)}")
    print(f"TFIM feature shape: {ordered_features.shape}")
    print(f"Binary OOS episodes: {predictions[predictions['model'] == 'state_plus_motion']['episode_id'].nunique()}")
    print("\nPrimary comparison:")
    print(summary.to_string(index=False))
    print("\nMean correction norm by path geometry:")
    print(predictions[predictions["model"] != "state_plus_motion"].groupby("path_geometry")["correction_l2_norm"].mean().to_string())
    print("\nTFIM order sensitivity:")
    print(f"mean absolute feature change: {delta.mean():.6f}")
    print(f"max absolute feature change:  {delta.max():.6f}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
