"""Episode-level chronological evaluation for branching-state forecasting.

The unit of evaluation is an extracted branch episode, not a daily row.
Training and validation episodes are eligible only when their full future
outcome window is observable before the first branch point in the test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EpisodeWalkForwardConfig:
    """Geometry for sparse episode-level walk-forward evaluation."""

    n_folds: int = 5
    min_train_episodes: int = 18
    val_episodes: int = 6
    min_test_episodes: int = 4
    outcome_horizon_rows: int = 40
    embargo_rows: int = 0

    def __post_init__(self) -> None:
        if self.n_folds < 1:
            raise ValueError("n_folds must be positive")
        if self.min_train_episodes < 1:
            raise ValueError("min_train_episodes must be positive")
        if self.val_episodes < 1:
            raise ValueError("val_episodes must be positive")
        if self.min_test_episodes < 1:
            raise ValueError("min_test_episodes must be positive")
        if self.outcome_horizon_rows < 1:
            raise ValueError("outcome_horizon_rows must be positive")
        if self.embargo_rows < 0:
            raise ValueError("embargo_rows must be non-negative")


def prepare_episode_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """Validate and chronologically order complete labeled episodes."""
    required = {
        "episode_id",
        "branch_idx",
        "branch_date",
        "outcome",
        "outcome_complete",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise KeyError(f"Missing episode columns: {sorted(missing)}")

    out = episodes[episodes["outcome_complete"]].copy()
    out["branch_date"] = pd.to_datetime(out["branch_date"])
    out = out.sort_values(["branch_idx", "episode_id"]).reset_index(drop=True)

    if out["branch_idx"].duplicated().any():
        raise ValueError("Duplicate branch_idx values are not allowed")
    if not out["branch_idx"].is_monotonic_increasing:
        raise ValueError("Episodes are not chronologically ordered")
    return out


def _partition_test_blocks(
    n_episodes: int,
    first_test_pos: int,
    n_folds: int,
    min_test_episodes: int,
) -> list[tuple[int, int]]:
    remaining = n_episodes - first_test_pos
    if remaining < n_folds * min_test_episodes:
        raise ValueError(
            "Not enough remaining episodes for requested folds: "
            f"remaining={remaining}, n_folds={n_folds}, "
            f"min_test_episodes={min_test_episodes}"
        )

    sizes = np.full(n_folds, remaining // n_folds, dtype=int)
    sizes[: remaining % n_folds] += 1
    if int(sizes.min()) < min_test_episodes:
        raise ValueError("Constructed test block smaller than minimum")

    blocks: list[tuple[int, int]] = []
    start = first_test_pos
    for size in sizes:
        end = start + int(size)
        blocks.append((start, end))
        start = end
    return blocks


def make_episode_walkforward_folds(
    episodes: pd.DataFrame,
    config: EpisodeWalkForwardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create purged expanding walk-forward folds at episode level.

    Test blocks are chronological and non-overlapping.

    For each test block, prior episodes are eligible for train/validation only if
    their full outcome window ends strictly before the first test branch point,
    after applying any additional row embargo.

    The most recent eligible ``val_episodes`` form validation; all earlier
    eligible episodes form the expanding training set.
    """
    cfg = config or EpisodeWalkForwardConfig()
    eps = prepare_episode_table(episodes)

    first_test_pos = cfg.min_train_episodes + cfg.val_episodes
    test_blocks = _partition_test_blocks(
        len(eps),
        first_test_pos,
        cfg.n_folds,
        cfg.min_test_episodes,
    )

    assignment_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for fold_id, (test_start, test_end) in enumerate(test_blocks, start=1):
        test = eps.iloc[test_start:test_end]
        first_test_branch_idx = int(test.iloc[0]["branch_idx"])
        availability_cutoff = first_test_branch_idx - cfg.embargo_rows

        prior = eps.iloc[:test_start].copy()
        prior["outcome_available_idx"] = (
            prior["branch_idx"].astype(int) + cfg.outcome_horizon_rows
        )
        eligible = prior[
            prior["outcome_available_idx"] < availability_cutoff
        ].copy()

        if len(eligible) < cfg.min_train_episodes + cfg.val_episodes:
            raise ValueError(
                f"Fold {fold_id} has only {len(eligible)} leakage-safe prior episodes; "
                f"need {cfg.min_train_episodes + cfg.val_episodes}"
            )

        val = eligible.iloc[-cfg.val_episodes :]
        train = eligible.iloc[: -cfg.val_episodes]

        if len(train) < cfg.min_train_episodes:
            raise ValueError(
                f"Fold {fold_id} train size {len(train)} below minimum "
                f"{cfg.min_train_episodes}"
            )

        split_frames = {
            "train": train,
            "val": val,
            "test": test,
        }
        for split, frame in split_frames.items():
            for row in frame.itertuples(index=False):
                assignment_rows.append(
                    {
                        "fold": fold_id,
                        "split": split,
                        "episode_id": int(row.episode_id),
                        "branch_idx": int(row.branch_idx),
                        "branch_date": row.branch_date,
                        "outcome": row.outcome,
                    }
                )

        summary_rows.append(
            {
                "fold": fold_id,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "train_start": train.iloc[0]["branch_date"],
                "train_end": train.iloc[-1]["branch_date"],
                "val_start": val.iloc[0]["branch_date"],
                "val_end": val.iloc[-1]["branch_date"],
                "test_start": test.iloc[0]["branch_date"],
                "test_end": test.iloc[-1]["branch_date"],
                "first_test_branch_idx": first_test_branch_idx,
                "latest_val_outcome_available_idx": int(
                    val["branch_idx"].max() + cfg.outcome_horizon_rows
                ),
                "leakage_gap_rows": int(
                    first_test_branch_idx
                    - (val["branch_idx"].max() + cfg.outcome_horizon_rows)
                ),
            }
        )

    assignments = pd.DataFrame(assignment_rows)
    summaries = pd.DataFrame(summary_rows)
    return assignments, summaries
