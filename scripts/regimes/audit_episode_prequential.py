"""Audit expanding leakage-safe prequential geometry before model fitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.evaluation.episode_prequential import (
    EpisodePrequentialConfig,
    make_episode_prequential_steps,
)


DEFAULT_EPISODES = Path(
    "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv"
)
DEFAULT_OUTPUT = Path("results/regimes/episode_prequential_audit_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.episodes.exists():
        raise FileNotFoundError(args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(args.episodes, parse_dates=["branch_date"])
    config = EpisodePrequentialConfig(
        min_train_episodes=18,
        outcome_horizon_rows=40,
        embargo_rows=0,
    )

    assignments, summary = make_episode_prequential_steps(episodes, config)
    tests = summary.copy()

    outcome_counts = (
        tests.groupby("test_outcome")
        .size()
        .rename("count")
        .reset_index()
    )
    binary = tests[tests["test_outcome"].isin(["recovery", "relapse"])].copy()
    binary_counts = (
        binary.groupby("test_outcome")
        .size()
        .rename("count")
        .reset_index()
    )

    assignments.to_csv(args.output / "step_assignments.csv", index=False)
    summary.to_csv(args.output / "step_summary.csv", index=False)
    outcome_counts.to_csv(args.output / "oos_outcome_counts.csv", index=False)
    binary_counts.to_csv(args.output / "oos_binary_counts.csv", index=False)

    manifest = {
        "episode_source": str(args.episodes),
        "n_complete_episodes": int(episodes["outcome_complete"].sum()),
        "config": config.__dict__,
        "unit_of_evaluation": "branch episode",
        "evaluation": "expanding one-step prequential",
        "leakage_rule": (
            "training episode outcome window must end strictly before test branch point"
        ),
        "status": "protocol audit; no model fitting",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Episodes: {args.episodes}")
    print(f"Complete episodes: {manifest['n_complete_episodes']}")
    print(f"Eligible OOS episodes: {len(summary)}")
    print(f"First OOS date: {summary['test_branch_date'].min().date()}")
    print(f"Last OOS date: {summary['test_branch_date'].max().date()}")
    print(f"Minimum training size: {summary['n_train'].min()}")
    print(f"Maximum training size: {summary['n_train'].max()}")
    print(f"Minimum leakage gap rows: {summary['leakage_gap_rows'].min()}")

    print("\nPooled OOS outcome counts:")
    print(outcome_counts.to_string(index=False))
    print("\nPooled OOS recovery/relapse counts:")
    print(binary_counts.to_string(index=False))

    print("\nPrequential test sequence:")
    print(
        summary[
            [
                "step",
                "test_episode_id",
                "test_branch_date",
                "test_outcome",
                "n_train",
                "leakage_gap_rows",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
