"""Audit the formal episode-level walk-forward geometry before model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.evaluation.episode_walkforward import (
    EpisodeWalkForwardConfig,
    make_episode_walkforward_folds,
)


DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/regimes/episode_walkforward_audit_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def split_outcome_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby(["fold", "split", "outcome"])
        .size()
        .rename("count")
        .reset_index()
    )


def binary_task_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    binary = assignments[assignments["outcome"].isin(["recovery", "relapse"])]
    return (
        binary.groupby(["fold", "split", "outcome"])
        .size()
        .rename("count")
        .reset_index()
    )


def main() -> None:
    args = parse_args()
    if not args.episodes.exists():
        raise FileNotFoundError(args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(args.episodes, parse_dates=["branch_date"])
    config = EpisodeWalkForwardConfig(
        n_folds=5,
        min_train_episodes=18,
        val_episodes=6,
        min_test_episodes=4,
        outcome_horizon_rows=40,
        embargo_rows=0,
    )

    assignments, summary = make_episode_walkforward_folds(episodes, config)
    outcome_counts = split_outcome_counts(assignments)
    binary_counts = binary_task_counts(assignments)

    assignments.to_csv(args.output / "fold_assignments.csv", index=False)
    summary.to_csv(args.output / "fold_summary.csv", index=False)
    outcome_counts.to_csv(args.output / "split_outcome_counts.csv", index=False)
    binary_counts.to_csv(args.output / "binary_task_counts.csv", index=False)

    manifest = {
        "episode_source": str(args.episodes),
        "n_complete_episodes": int(episodes["outcome_complete"].sum()),
        "config": config.__dict__,
        "unit_of_evaluation": "branch episode",
        "leakage_rule": (
            "train/validation outcome window must end strictly before first test branch point"
        ),
        "test_blocks": "chronological, non-overlapping episode blocks",
        "validation": "most recent leakage-safe prior episodes",
        "status": "protocol audit; no model fitting",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Episodes: {args.episodes}")
    print(f"Complete episodes: {manifest['n_complete_episodes']}")
    print("\nFold geometry:")
    print(summary.to_string(index=False))
    print("\nAll-outcome counts by split:")
    print(outcome_counts.to_string(index=False))
    print("\nRecovery/relapse-only counts by split:")
    print(binary_counts.to_string(index=False))
    print("\nTest episode assignments:")
    test = assignments[assignments["split"] == "test"]
    print(test[["fold", "episode_id", "branch_date", "outcome"]].to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
