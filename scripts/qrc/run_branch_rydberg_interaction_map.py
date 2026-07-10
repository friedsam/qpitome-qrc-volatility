"""Last bounded small-sample interaction-map experiment.

Comparison:

1. state_plus_motion baseline
2. baseline + six classical ring-product interaction features
3. baseline + six Rydberg connected-correlation features

All nonlinear features are built from the same six state_plus_motion inputs after
train-only standardization. The baseline is fitted first and frozen. Only a
strongly regularized no-intercept additive logit correction is learned.
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
from qpitome_qrc.qrc.rydberg_interaction_map import (
    RydbergInteractionConfig,
    classical_ring_products,
    rydberg_interaction_features,
)

STATE_MOTION = FEATURE_SETS["state_plus_motion"]
BASELINE_C = 1.0
CORRECTION_CONFIG = OffsetCorrectionConfig(l2_penalty=10.0)
RYDBERG_CONFIG = RydbergInteractionConfig()


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
) -> tuple[Pipeline, np.ndarray, float]:
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
    return model, p_train, p_test


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

    daily = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(args.episodes, parse_dates=["branch_date"]).sort_values("branch_date").reset_index(drop=True)

    featured = episodes.copy()
    featured["stress_ratio"] = featured["rv_20d"] / featured["branch_stress_cut"]
    missing = set(STATE_MOTION) - set(featured.columns)
    if missing:
        raise KeyError(f"Missing state_plus_motion features: {sorted(missing)}")
    featured = featured.set_index("episode_id", drop=False)

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

        x_train_raw = featured.loc[train_ids, list(STATE_MOTION)].to_numpy(dtype=float)
        x_test_raw = featured.loc[[test_id], list(STATE_MOTION)].to_numpy(dtype=float)

        baseline_model, p_train, p_test = fit_baseline(
            x_train_raw,
            y_train,
            x_test_raw,
        )
        train_offset = logit_from_probability(p_train)
        test_offset = float(logit_from_probability([p_test])[0])

        scaler = baseline_model.named_steps["scale"]
        x_train_scaled = scaler.transform(x_train_raw)
        x_test_scaled = scaler.transform(x_test_raw)

        classical_train = classical_ring_products(x_train_scaled)
        classical_test = classical_ring_products(x_test_scaled)
        rydberg_train = rydberg_interaction_features(x_train_scaled, RYDBERG_CONFIG)
        rydberg_test = rydberg_interaction_features(x_test_scaled, RYDBERG_CONFIG)

        common = {
            "step": int(step.step),
            "episode_id": test_id,
            "branch_date": test_episode["branch_date"],
            "outcome": test_episode["outcome"],
            "y_true": y_true,
            "n_train_binary": len(train_ids),
            "p_baseline": p_test,
        }

        rows.append(
            {
                **common,
                "model": "state_plus_motion",
                "correction_l2_norm": 0.0,
                "p_recovery": p_test,
            }
        )

        for model_name, x_train, x_test in (
            ("classical_ring_products", classical_train, classical_test),
            ("rydberg_connected_correlations", rydberg_train, rydberg_test),
        ):
            p_corrected, beta = fit_offset_logit_correction(
                x_train,
                y_train,
                train_offset,
                x_test,
                test_offset,
                CORRECTION_CONFIG,
            )
            rows.append(
                {
                    **common,
                    "model": model_name,
                    "correction_l2_norm": float(np.linalg.norm(beta)),
                    "p_recovery": p_corrected,
                }
            )

    predictions = pd.DataFrame(rows)
    if predictions.empty:
        raise RuntimeError("No binary OOS predictions were produced")

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
                    "a fixed structured Rydberg interaction map adds stable nonlinear "
                    "branch-transition information beyond the linear state_plus_motion baseline"
                ),
                "falsifying_result": (
                    "Rydberg correction does not improve over baseline and does not "
                    "outperform the matched six-feature classical ring-product control"
                ),
                "data": str(args.data),
                "episodes": str(args.episodes),
                "baseline_features": list(STATE_MOTION),
                "input_scaling": "train-only StandardScaler reused from baseline pipeline",
                "classical_control": "six adjacent products on clipped standardized inputs",
                "rydberg": RYDBERG_CONFIG.__dict__,
                "rydberg_observables": "six nearest-neighbor connected occupation correlations",
                "correction_geometry": "frozen baseline logit + no-intercept correction",
                "correction_l2_penalty": CORRECTION_CONFIG.l2_penalty,
                "protocol": protocol.__dict__,
                "tuning": "none",
                "small_sample_stop_rule": (
                    "If this bounded Rydberg interaction map also fails, close the "
                    "episode-level small-N architecture search."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Complete episodes: {len(episodes)}")
    print(
        "Binary OOS episodes: "
        f"{predictions[predictions['model'] == 'state_plus_motion']['episode_id'].nunique()}"
    )
    print("\nPrimary comparison:")
    print(summary.to_string(index=False))

    print("\nMean correction norm by model:")
    print(
        predictions[predictions["model"] != "state_plus_motion"]
        .groupby("model")["correction_l2_norm"]
        .mean()
        .to_string()
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
