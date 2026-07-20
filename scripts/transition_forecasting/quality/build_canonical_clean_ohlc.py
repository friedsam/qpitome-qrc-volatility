from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.cleaning import build_cleaned_ohlc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the canonical cleaned OHLC dataset used by downstream transition workflows."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/raw/transition_forecasting/global_stock_indices_historical_data/individual_indices_data"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed/transition_forecasting/canonical_ohlc"),
    )
    parser.add_argument("--expected-structural-flags", type=int, default=12)
    parser.add_argument("--expected-affected-indices", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_cleaned_ohlc(
        args.raw_root,
        args.output_root,
        expected_structural_flags=args.expected_structural_flags,
        expected_affected_indices=args.expected_affected_indices,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
