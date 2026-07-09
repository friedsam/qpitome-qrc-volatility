"""Run the first boring probabilistic branch-resolution baseline ladder.

Task: recovery versus relapse among branch episodes.
Geometry: leakage-safe expanding prequential prediction.
Mixed episodes remain in chronology but are excluded from binary fitting/scoring.
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

from qpitome_qrc.baselines.branch_probabilistic import (
    FEATURE_SETS,
    add_branch_baseline_features,
    binary_episode_frame,
    empirical_prior_probability,
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
DEFAULT_OUTPUT = Path("results/baselines/branch_probabilistic_front_v1")


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
        }

        prediction_rows.append(
            {
                **base,
                "model": "historical_class_rate",
                "p_recovery": empirical_prior_probability(train),
            }
        )
        for model_name, feature_columns in FEATURE_SETS.items():
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
        raise RuntimeError("No binary OOS predictions were produced")

    metrics = pd.DataFrame(
        [metric_row(model, group) for model, group in predictions.groupby("model")]
    ).sort_values(["brier", "log_loss"]).reset_index(drop=True)

    feature_manifest = pd.DataFrame(
        [
            {"model": "historical_class_rate", "features": "historical recovery rate"},
            *[
                {"model": name, "features": " | ".join(columns)}
                for name, columns in FEATURE_SETS.items()
            ],
        ]
    )

    predictions.to_csv(args.output / "oos_predictions.csv", index=False)
    metrics.to_csv(args.output / "summary_metrics.csv", index=False)
    feature_manifest.to_csv(args.output / "feature_manifest.csv", index=False)
    steps.to_csv(args.output / "protocol_steps.csv", index=False)

    manifest = {
        "data": str(args.data),
        "episodes": str(args.episodes),
        "task": "binary recovery versus relapse",
        "positive_class": "recovery",
        "mixed_handling": "retained in chronology; excluded from binary fitting and scoring",
        "protocol": protocol.__dict__,
        "models": ["historical_class_rate", *FEATURE_SETS.keys()],
        "logit": {
            "standardization": "fit on each prequential training set only",
            "C": 1.0,
            "solver": "lbfgs",
            "tuning": "none",
        },
        "status": "first boring probabilistic baseline ladder",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Data: {args.data}")
    print(f"Episodes: {args.episodes}")
    print(f"Binary OOS episodes: {predictions['episode_id'].nunique()}")
    print(
        "Binary OOS range: "
        f"{predictions['branch_date'].min().date()} -> {predictions['branch_date'].max().date()}"
    )
    first = predictions.groupby("episode_id", sort=False).first()
    print(
        "OOS class counts: "
        f"recovery={(first['y_true'] == 1).sum()}, relapse={(first['y_true'] == 0).sum()}"
    )
    print("\nSummary metrics:")
    print(metrics.to_string(index=False))
    print("\nPer-episode predictions:")
    pivot = predictions.pivot_table(
        index=["episode_id", "branch_date", "outcome", "n_train_binary"],
        columns="model",
        values="p_recovery",
    ).reset_index()
    print(pivot.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
