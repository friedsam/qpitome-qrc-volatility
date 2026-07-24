from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.mnist_acquisition import acquire_mnist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire MNIST from Kaggle or a hash-verified repository fallback."
        )
    )
    parser.add_argument(
        "--destination", type=Path, default=Path("data/raw/mnist")
    )
    parser.add_argument(
        "--fallback",
        type=Path,
        default=Path("data/fallback/transition_forecasting/mnist"),
    )
    parser.add_argument(
        "--source-mode",
        choices=["auto", "live", "fallback"],
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = acquire_mnist(
        args.destination,
        args.fallback,
        source_mode=args.source_mode,
        force=args.force,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
