from __future__ import annotations

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_transition_diagnostics import (
    order_destruction_audit,
    permute_transition_windows,
    summarize_transition_windows,
)


def make_windows() -> tuple[np.ndarray, np.ndarray]:
    episode_ids = np.asarray([1, 2, 3], dtype=int)
    base = np.arange(40, dtype=float)
    windows = np.stack(
        [
            np.column_stack(
                [
                    base,
                    base[::-1],
                    np.sin(base / 4.0),
                    np.cos(base / 5.0),
                ]
            ),
            np.column_stack(
                [
                    base * 0.5,
                    np.roll(base, 5),
                    np.sin(base / 3.0),
                    np.cos(base / 6.0),
                ]
            ),
            np.column_stack(
                [
                    -base,
                    np.sqrt(base + 1.0),
                    np.sin(base / 2.0),
                    np.cos(base / 7.0),
                ]
            ),
        ]
    )
    return episode_ids, windows


def test_full_permutation_preserves_contemporaneous_channel_tuples() -> None:
    _, windows = make_windows()
    permuted = permute_transition_windows(windows, seed=7, block_size=1)

    for episode_idx in range(len(windows)):
        original_rows = sorted(map(tuple, windows[episode_idx].tolist()))
        permuted_rows = sorted(map(tuple, permuted[episode_idx].tolist()))
        assert original_rows == permuted_rows


def test_order_insensitive_summaries_are_permutation_invariant() -> None:
    episode_ids, windows = make_windows()
    original = summarize_transition_windows(episode_ids, windows)
    permuted = summarize_transition_windows(
        episode_ids,
        permute_transition_windows(windows, seed=11, block_size=1),
    )

    for column in original.columns:
        if column.endswith("__mean") or column.endswith("__std"):
            np.testing.assert_allclose(original[column], permuted[column])


def test_order_sensitive_summaries_change_after_permutation() -> None:
    episode_ids, windows = make_windows()
    original = summarize_transition_windows(episode_ids, windows)
    permuted = summarize_transition_windows(
        episode_ids,
        permute_transition_windows(windows, seed=13, block_size=1),
    )
    audit = order_destruction_audit(
        original,
        permuted,
        permutation_name="full",
    )

    changed = audit[
        audit["summary_type"].isin(
            ["last", "late_minus_early", "slope", "lag1_autocorr"]
        )
    ]
    assert (changed["mean_abs_change_over_original_sd"] > 0).any()


def test_block_permutation_preserves_within_block_row_order() -> None:
    _, windows = make_windows()
    permuted = permute_transition_windows(windows[:1], seed=17, block_size=5)[0]

    original_blocks = [windows[0, start : start + 5] for start in range(0, 40, 5)]
    permuted_blocks = [permuted[start : start + 5] for start in range(0, 40, 5)]

    for block in permuted_blocks:
        assert any(np.array_equal(block, original) for original in original_blocks)
