from __future__ import annotations

import argparse
from pathlib import Path

from data.build_volatility_dataset import build_volatility_dataset

DEFAULT_RAW_DIR = Path("data/raw/yahoo_daily_history")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a processed volatility dataset")
    parser.add_argument(
        "--market-data",
        type=Path,
        default=DEFAULT_RAW_DIR / "historical_market_data.csv",
    )
    parser.add_argument(
        "--volatility-data",
        type=Path,
        default=DEFAULT_RAW_DIR / "historical_volatility_data.csv",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-name", default="spy_vix_volatility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_volatility_dataset(
        market_data_path=args.market_data,
        volatility_data_path=args.volatility_data,
        output_root=args.output_root,
        output_name=args.output_name,
    )
    print(f"Saved: {result.output_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Shape: ({result.rows}, {result.columns})")
    print(f"Date range: {result.date_start} -> {result.date_end}")


if __name__ == "__main__":
    main()
