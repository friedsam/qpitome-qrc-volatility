from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.three_channel import build_three_channel_dataset


DEFAULT_SOURCE = Path(
    "data/processed/transition_forecasting/global_transition_dataset_1d"
)
DEFAULT_OUTPUT = Path(
    "data/processed/transition_forecasting/global_transition_dataset_3d"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the causal three-channel transition dataset from the validated one-channel dataset."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_three_channel_dataset(
        args.source_dir,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
