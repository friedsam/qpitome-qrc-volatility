#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.dataset import (
    DEFAULT_FROZEN_INVENTORY,
    DEFAULT_FROZEN_RANGE_QUALITY,
    build_processed_dataset,
)
from transition_forecasting.data.three_channel import build_three_channel_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build aligned one- and three-channel transition datasets."
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-1d", type=Path, required=True)
    parser.add_argument("--output-3d", type=Path, required=True)
    parser.add_argument(
        "--inventory-contract",
        type=Path,
        default=DEFAULT_FROZEN_INVENTORY,
    )
    parser.add_argument(
        "--range-quality-contract",
        type=Path,
        default=DEFAULT_FROZEN_RANGE_QUALITY,
    )
    parser.add_argument("--expected-structural-flags", type=int, default=12)
    parser.add_argument("--expected-affected-indices", type=int, default=2)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    one_channel = build_processed_dataset(
        args.raw_root,
        args.output_1d,
        inventory_contract=args.inventory_contract,
        range_quality_contract=args.range_quality_contract,
        expected_structural_flags=args.expected_structural_flags,
        expected_affected_indices=args.expected_affected_indices,
        controls_per_positive=args.controls_per_positive,
        force=args.force,
    )
    three_channel = build_three_channel_dataset(
        args.output_1d,
        args.output_3d,
        force=args.force,
    )
    print(
        json.dumps(
            {"one_channel": one_channel, "three_channel": three_channel},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
