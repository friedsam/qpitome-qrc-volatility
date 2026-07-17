#!/usr/bin/env python3
"""Build the reusable eight-index Parkinson-volatility dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from data.build_transition_volatility_dataset import build_transition_volatility_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/transition_forecasting/eight_index_ohlc"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "data/processed/transition_forecasting/eight_index_parkinson_volatility"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_transition_volatility_dataset(args.raw_root, args.output_root)
    print(f"Saved: {result.output_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Rows: {result.rows}")
    print(f"Indices: {result.indices}")
    print(f"Date range: {result.date_start} -> {result.date_end}")


if __name__ == "__main__":
    main()
