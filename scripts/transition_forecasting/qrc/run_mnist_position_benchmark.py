from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.mnist_position_benchmark import (
    MnistPositionBenchmarkConfig,
    run_mnist_position_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the position-encoded Rydberg MNIST benchmark."
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mnist"))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/run_mnist_position_benchmark"
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument(
        "--n-atoms", type=int, nargs="+", default=[5, 6, 7, 9]
    )
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--subsample-seed", type=int, default=20260722)
    parser.add_argument("--random-feature-seed", type=int, default=20260722)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--shot-seed", type=int, default=20260722)
    parser.add_argument("--total-time-us", type=float, default=1.6)
    parser.add_argument(
        "--probe-fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MnistPositionBenchmarkConfig(
        atom_counts=tuple(args.n_atoms),
        train_size=args.train_size,
        test_size=args.test_size,
        subsample_seed=args.subsample_seed,
        random_feature_seed=args.random_feature_seed,
        shots=args.shots,
        shot_seed=args.shot_seed,
        total_time_us=args.total_time_us,
        probe_fractions=tuple(args.probe_fractions),
    )
    run_dir = run_mnist_position_benchmark(
        raw_dir=args.raw_dir,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
