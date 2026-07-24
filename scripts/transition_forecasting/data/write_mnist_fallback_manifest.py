from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.data.mnist_acquisition import (
    validate_source,
    write_fallback_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an MNIST fallback snapshot and write its hash manifest."
    )
    parser.add_argument(
        "--fallback",
        type=Path,
        default=Path("data/fallback/transition_forecasting/mnist"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_source(args.fallback)
    path = write_fallback_manifest(args.fallback)
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
