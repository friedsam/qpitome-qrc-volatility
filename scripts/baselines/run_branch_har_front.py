"""Evaluate causal HAR-derived branch-resolution baselines.

HAR itself remains the historical continuous-volatility model. For every branch
episode, a causal HAR forecast is generated using only daily rows whose 20-day
future target is fully observable before the branch point. Small logistic
readouts then test whether the forecasted volatility path predicts recovery
versus relapse under the accepted prequential episode protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from qpitome_qrc.baselines.branch_har import (
    HAR_FEATURES,
    add_causal_har_episode_features,
)
from qpitome_qrc.baselines.branch_probabilistic import (
    add_branch_baseline_features,
    binary_episode_frame,
    fit_logit_probability,
)
from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)


DEFAULT_DATA = Path("data/processed/phase3_spy_vix_volatility_extended.csv")
DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/baselines/branch_har_front_v1")

HAR_BRANCH_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "har_forecast_gap": ("har_log_future_to_current",),
    "har_forecast_level_gap": (
        "har_future_rv20_pred",
        "har_log_future_to_current",
    ),
    "har_plus_state": (
        "har_log_future_to_current",
        "stress_ratio",
        "drawdown_120d",
        "rv_ratio_5_20_branch",
        "return_5d_branch",
    ),
    "har_plus_motion": (
        "har_log_future_to_current",
        "stress_ratio",
        "drawdown_120d",
        "rv_ratio_5_20_branch",
        "return_5d_branch",
        "rv_5d_change_5d_branch",
        "worst_return_5d_in_prior_window",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def metric_row(model: str, pred: pd.DataFrame) -> dict[str, object]:
    y = pred["y_true"].to_numpy(dtype=int)
    p = pred["p_recovery"].to_numpy(dtype=float)
    return {
        "model": model,
        "n_oos": len(pred),
        "n_recovery": int(y.sum()),
        "n_relapse": int((1 - y).sum()),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc_recovery": average_precision_score(y, p),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "accuracy_0p5": accuracy_score(y, p >= 0.5),
        "mean_p_recovery": float(np.mean(p)),
    }


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)
    if not args.episodes.exists():
        raise FileNotFoundError(args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(args.data, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    episodes = pd.read_csv(args.episodes, parse_dates=["branch_date"])
    episodes = add_branch_baseline_features(episodes, daily)
    episodes = add_causal_har_episode_features(daily, episodes)

    protocol = EpisodePrequentialConfig(
        min_train_episodes=18,
        outcome_horizon_rows=40,
        embargo_rows=0,
    )
    assignments, steps = make_episode_prequential_steps(episodes, protocol)
    episode_lookup = episodes.set_index("episode_id", drop=False)

    prediction_rows: list[dict[str, object]] = []
    for step in steps.itertuples(index=False):
        test_episode_id = int(step.test_episode_id)
        test_episode = episode_lookup.loc[[test_episode_id]].copy()
        if test_episode.iloc[0]["outcome"] not in ("recovery", "relapse"):
            continue

        train_ids = assignments[
            (assignments["step"] == step.step)
            & (assignments["split"] == "train")
        ]["episode_id"].astype(int)
        train_all = episode_lookup.loc[train_ids].copy()
        train = binary_episode_frame(train_all)
        test = binary_episode_frame(test_episode)

        base = {
            "step": int(step.step),
            "episode_id": test_episode_id,
            "branch_date": test_episode.iloc[0]["branch_date"],
            "outcome": test_episode.iloc[0]["outcome"],
            "y_true": int(test.iloc[0]["y_recovery"]),
            "n_train_all": len(train_all),
            "n_train_binary": len(train),
            "n_train_recovery": int(train["y_recovery"].sum()),
            "n_train_relapse": int((1 - train["y_recovery"]).sum()),
            "leakage_gap_rows": int(step.leakage_gap_rows),
            "har_future_rv20_pred": float(test_episode.iloc[0]["har_future_rv20_pred"]),
            "har_log_future_to_current": float(test_episode.iloc[0]["har_log_future_to_current"]),
            "har_daily_train_rows": int(test_episode.iloc[0]["har_daily_train_rows"]),
        }

        for model_name, feature_columns in HAR_BRANCH_FEATURE_SETS.items():
            prediction_rows.append(
                {
                    **base,
                    "model": model_name,
                    "p_recovery": fit_logit_probability(
                        train,
                        test,
                        feature_columns,
                    ),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    if predictions.empty:
        raise RuntimeError("No HAR branch predictions were produced")

    metrics = pd.DataFrame(
        [metric_row(model, group) for model, group in predictions.groupby("model")]
    ).sort_values(["brier", "log_loss"]).reset_index(drop=True)

    har_episode_features = episodes[
        [
            "episode_id",
            "branch_date",
            "outcome",
            "rv_20d",
            "har_future_rv20_pred",
            "har_log_future_to_current",
            "har_future_minus_current",
            "har_daily_train_rows",
            "har_latest_target_row",
        ]
    ].copy()

    predictions.to_csv(args.output / "oos_predictions.csv", index=False)
    metrics.to_csv(args.output / "summary_metrics.csv", index=False)
    har_episode_features.to_csv(args.output / "har_episode_features.csv", index=False)
    steps.to_csv(args.output / "protocol_steps.csv", index=False)

    manifest = {
        "data": str(args.data),
        "episodes": str(args.episodes),
        "task": "binary recovery versus relapse",
        "positive_class": "recovery",
        "mixed_handling": "retained in chronology; excluded from binary fitting and scoring",
        "protocol": protocol.__dict__,
        "har_model": {
            "features": list(HAR_FEATURES),
            "target": "future_rv_20d",
            "scaler": "StandardScaler",
            "estimator": "Ridge",
            "alpha": 1.0,
            "target_horizon_rows": 20,
            "prediction_floor": 1e-8,
            "daily_fit_rule": (
                "for each episode, fit only rows whose future_rv_20d target is complete before branch_idx"
            ),
        },
        "branch_readouts": {
            name: list(columns) for name, columns in HAR_BRANCH_FEATURE_SETS.items()
        },
        "readout_model": {
            "scaler": "StandardScaler fit on each episode training set",
            "estimator": "LogisticRegression",
            "C": 1.0,
            "tuning": "none",
        },
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Data: {args.data}")
    print(f"Episodes: {args.episodes}")
    print(f"Binary OOS episodes: {predictions['episode_id'].nunique()}")
    print("\nSummary metrics:")
    print(metrics.to_string(index=False))
    print("\nPer-episode HAR diagnostics and predictions:")
    pivot = predictions.pivot_table(
        index=[
            "episode_id",
            "branch_date",
            "outcome",
            "n_train_binary",
            "har_future_rv20_pred",
            "har_log_future_to_current",
        ],
        columns="model",
        values="p_recovery",
    ).reset_index()
    print(pivot.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
