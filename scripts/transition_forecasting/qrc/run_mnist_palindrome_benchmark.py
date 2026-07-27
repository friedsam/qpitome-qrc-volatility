from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.mnist_palindrome_benchmark import (
    MnistPalindromeBenchmarkConfig,
    merge_mnist_palindrome_shards,
    run_mnist_palindrome_feature_shard,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)

DEFAULT_RAW_DIR = Path("data/raw/mnist")
DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/mnist_palindrome_benchmark"
)


def add_common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--subset-seed", type=int, default=20260726)
    parser.add_argument("--pool-rows", type=int, default=4)
    parser.add_argument("--pool-cols", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--logistic-cs",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
    )
    parser.add_argument("--inner-validation-fraction", type=float, default=0.20)
    parser.add_argument("--readout-seed", type=int, default=20260726)
    parser.add_argument("--include-interaction-off", action="store_true")


def build_config(args: argparse.Namespace) -> MnistPalindromeBenchmarkConfig:
    return MnistPalindromeBenchmarkConfig(
        train_size=args.train_size,
        test_size=args.test_size,
        subset_seed=args.subset_seed,
        pool_rows=args.pool_rows,
        pool_cols=args.pool_cols,
        logistic_cs=tuple(args.logistic_cs),
        inner_validation_fraction=args.inner_validation_fraction,
        readout_seed=args.readout_seed,
        batch_size=args.batch_size,
        include_interaction_off=args.include_interaction_off,
    )


def frozen_reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        spacing_short_um=8.5,
        spacing_long_um=10.0,
        defect_edge=1,
        defect_offset_um=0.6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.02,
        probe_fractions=(0.25, 0.5, 1.0),
        shots=None,
        shot_seed=20260726,
    )


def frozen_geometry() -> StaggeredLadderGeometryConfig:
    return StaggeredLadderGeometryConfig(
        longitudinal_spacing_um=8.5,
        row_spacing_um=9.0,
        stagger_fraction=0.35,
        bottom_spacing_scale=1.05,
        defect_site=4,
        defect_dx_um=0.35,
        defect_dy_um=-0.40,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run or merge the primary A/B/A-palindrome MNIST benchmark."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    shard = subparsers.add_parser(
        "shard",
        help="Generate one resumable exact-feature shard.",
    )
    shard.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    shard.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    shard.add_argument("--run-id", required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--resume", action="store_true")
    add_common_config(shard)

    merge = subparsers.add_parser(
        "merge",
        help="Validate all shards, merge features, and train final readouts.",
    )
    merge.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    merge.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    merge.add_argument("--run-id", required=True)
    add_common_config(merge)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.mode == "shard":
        run_dir = run_mnist_palindrome_feature_shard(
            raw_dir=args.raw_dir,
            results_root=args.out_root,
            run_id=args.run_id,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            config=config,
            reservoir=frozen_reservoir(),
            geometry=frozen_geometry(),
            interaction_scale=1.25,
            drive_phase_rad=0.0,
            resume=args.resume,
        )
    else:
        run_dir = merge_mnist_palindrome_shards(
            shard_dirs=args.shard_dirs,
            results_root=args.out_root,
            run_id=args.run_id,
            config=config,
        )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
