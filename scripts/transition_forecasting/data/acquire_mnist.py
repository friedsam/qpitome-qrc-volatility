from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.mnist_acquisition import acquire_mnist
from transition_forecasting.data.submission_provenance import (
    destination_filesystem_tempdir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire MNIST from the anonymous checksum-pinned Keras mirror or "
            "a hash-verified repository fallback."
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
    # ``acquire_mnist`` installs its verified candidate with an atomic rename.
    # qBraid mounts /tmp separately from /home/jovyan, so place tempfile-backed
    # staging beside the final destination to keep that rename on one filesystem.
    with destination_filesystem_tempdir(args.destination):
        report = acquire_mnist(
            args.destination,
            args.fallback,
            source_mode=args.source_mode,
            force=args.force,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
