"""Leakage-safe expanding prequential evaluation for branch episodes.

Each eligible episode receives exactly one strictly out-of-sample prediction.
Before predicting episode i, training may use only earlier episodes whose full
future outcome window has already completed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from qpitome_qrc.evaluation.episode_walkforward import prepare_episode_table


@dataclass(frozen=True)
class EpisodePrequentialConfig:
    min_train_episodes: int = 18
    outcome_horizon_rows: int = 40
    embargo_rows: int = 0

    def __post_init__(self) -> None:
        if self.min_train_episodes < 1:
            raise ValueError("min_train_episodes must be positive")
        if self.outcome_horizon_rows < 1:
            raise ValueError("outcome_horizon_rows must be positive")
        if self.embargo_rows < 0:
            raise ValueError("embargo_rows must be non-negative")


def make_episode_prequential_steps(
    episodes: pd.DataFrame,
    config: EpisodePrequentialConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-test-episode training assignments and step summaries.

    For a test episode with branch index ``b_test``, an earlier episode is
    eligible for training only when:

        branch_idx + outcome_horizon_rows < b_test - embargo_rows

    This guarantees that the full label window of every training episode is
    observable before the test branch point.
    """
    cfg = config or EpisodePrequentialConfig()
    eps = prepare_episode_table(episodes)

    assignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    step_id = 0

    for test_pos in range(len(eps)):
        test = eps.iloc[test_pos]
        test_branch_idx = int(test["branch_idx"])
        cutoff = test_branch_idx - cfg.embargo_rows

        prior = eps.iloc[:test_pos].copy()
        prior["outcome_available_idx"] = (
            prior["branch_idx"].astype(int) + cfg.outcome_horizon_rows
        )
        train = prior[prior["outcome_available_idx"] < cutoff].copy()

        if len(train) < cfg.min_train_episodes:
            continue

        step_id += 1
        for row in train.itertuples(index=False):
            assignment_rows.append(
                {
                    "step": step_id,
                    "split": "train",
                    "episode_id": int(row.episode_id),
                    "branch_idx": int(row.branch_idx),
                    "branch_date": row.branch_date,
                    "outcome": row.outcome,
                    "test_episode_id": int(test["episode_id"]),
                }
            )

        assignment_rows.append(
            {
                "step": step_id,
                "split": "test",
                "episode_id": int(test["episode_id"]),
                "branch_idx": test_branch_idx,
                "branch_date": test["branch_date"],
                "outcome": test["outcome"],
                "test_episode_id": int(test["episode_id"]),
            }
        )

        latest_train_available_idx = int(
            (train["branch_idx"].astype(int) + cfg.outcome_horizon_rows).max()
        )
        summary_rows.append(
            {
                "step": step_id,
                "test_episode_id": int(test["episode_id"]),
                "test_branch_idx": test_branch_idx,
                "test_branch_date": test["branch_date"],
                "test_outcome": test["outcome"],
                "n_train": len(train),
                "train_start": train.iloc[0]["branch_date"],
                "train_end": train.iloc[-1]["branch_date"],
                "latest_train_outcome_available_idx": latest_train_available_idx,
                "leakage_gap_rows": test_branch_idx - latest_train_available_idx,
            }
        )

    assignments = pd.DataFrame(assignment_rows)
    summary = pd.DataFrame(summary_rows)
    return assignments, summary
