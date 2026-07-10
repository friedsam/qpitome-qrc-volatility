"""Audit task-aligned branch-transition path channels before reservoir modeling.

This runner performs two fixed admission tests:

1. endpoint redundancy: can state_plus_motion reconstruct path summaries?
2. ordering destruction: do explicitly temporal summaries change under within-
   episode time permutation while contemporaneous channel tuples are preserved?

No outcome model is fitted here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.regimes.branch_transition_diagnostics import (
    ORDER_INSENSITIVE_SUMMARIES,
    ORDER_SENSITIVE_SUMMARIES,
    order_destruction_audit,
    permute_transition_windows,
    redundancy_audit,
    summarize_transition_windows,
)
from qpitome_qrc.regimes.branch_transition_path import (
    TRANSITION_PATH_COLUMNS,
    add_transition_path_channels,
    extract_transition_windows,
)

LOOKBACK = 40
PERMUTATION_SEED = 20260709
BLOCK_SIZE = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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

    channels = add_transition_path_channels(daily)
    episode_ids, windows = extract_transition_windows(
        channels,
        episodes,
        lookback=LOOKBACK,
    )
    if len(episode_ids) != len(episodes):
        raise RuntimeError(
            f"Only {len(episode_ids)} of {len(episodes)} episodes have complete "
            "task-aligned transition windows"
        )

    original = summarize_transition_windows(episode_ids, windows)

    full_permuted_windows = permute_transition_windows(
        windows,
        seed=PERMUTATION_SEED,
        block_size=1,
    )
    block_permuted_windows = permute_transition_windows(
        windows,
        seed=PERMUTATION_SEED,
        block_size=BLOCK_SIZE,
    )
    full_permuted = summarize_transition_windows(episode_ids, full_permuted_windows)
    block_permuted = summarize_transition_windows(episode_ids, block_permuted_windows)

    redundancy = redundancy_audit(episodes, original)
    order_full = order_destruction_audit(
        original,
        full_permuted,
        permutation_name="full_permutation",
    )
    order_block = order_destruction_audit(
        original,
        block_permuted,
        permutation_name=f"block_permutation_{BLOCK_SIZE}",
    )
    order = pd.concat([order_full, order_block], ignore_index=True)

    original.to_csv(args.output / "episode_path_summaries.csv", index=False)
    redundancy.to_csv(args.output / "endpoint_redundancy_audit.csv", index=False)
    order.to_csv(args.output / "ordering_destruction_audit.csv", index=False)
    (args.output / "run_manifest.json").write_text(
        json.dumps(
            {
                "claim": (
                    "task-specific transition paths contain temporal structure "
                    "not reducible to state_plus_motion endpoints"
                ),
                "falsifying_result": (
                    "order-sensitive summaries are reconstructible from the "
                    "baseline and/or remain nearly unchanged after temporal destruction"
                ),
                "data": str(args.data),
                "episodes": str(args.episodes),
                "lookback": LOOKBACK,
                "channels": list(TRANSITION_PATH_COLUMNS),
                "baseline": "state_plus_motion",
                "redundancy_model": "LOO StandardScaler + Ridge(alpha=10)",
                "permutation_seed": PERMUTATION_SEED,
                "block_size": BLOCK_SIZE,
                "outcome_model_fitted": False,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"Complete episodes: {len(episodes)}")
    print(f"Usable transition windows: {len(episode_ids)}")
    print(f"Window shape: {windows.shape}")

    print("\nMost baseline-reconstructible summaries:")
    print(
        redundancy.head(12).to_string(index=False)
    )

    sensitive = order[order["summary_type"].isin(ORDER_SENSITIVE_SUMMARIES)]
    print("\nOrder-sensitive summary response to temporal destruction:")
    print(
        sensitive.sort_values(
            ["permutation", "mean_abs_change_over_original_sd"],
            ascending=[True, False],
        )
        .groupby("permutation", sort=False)
        .head(12)
        .to_string(index=False)
    )

    insensitive = order[order["summary_type"].isin(ORDER_INSENSITIVE_SUMMARIES)]
    print("\nPermutation invariance controls (should be near zero change):")
    print(
        insensitive.groupby("permutation")[
            "mean_abs_change_over_original_sd"
        ]
        .max()
        .to_string()
    )

    print("\nMedian diagnostics by summary type:")
    merged_summary = (
        order.groupby(["permutation", "summary_type"], as_index=False)
        .agg(
            median_normalized_change=(
                "mean_abs_change_over_original_sd",
                "median",
            ),
            median_spearman=("spearman_original_vs_permuted", "median"),
        )
    )
    print(merged_summary.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
