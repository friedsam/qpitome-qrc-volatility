from __future__ import annotations

import argparse
from pathlib import Path

from data.download_market_data import download_market_data

DEFAULT_RAW_DIR = Path("data/raw/yahoo_daily_history")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download canonical raw SPY and VIX histories")
    parser.add_argument("--start-date", default="1990-01-01")
    parser.add_argument("--end-date", default="2026-06-05", help="Exclusive cutoff")
    parser.add_argument("--market-symbol", default="SPY")
    parser.add_argument("--volatility-symbol", default="^VIX")
    parser.add_argument(
        "--market-output",
        type=Path,
        default=DEFAULT_RAW_DIR / "historical_market_data.csv",
    )
    parser.add_argument(
        "--volatility-output",
        type=Path,
        default=DEFAULT_RAW_DIR / "historical_volatility_data.csv",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_RAW_DIR / "manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = download_market_data(
        market_symbol=args.market_symbol,
        volatility_symbol=args.volatility_symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        market_output_path=args.market_output,
        volatility_output_path=args.volatility_output,
        manifest_path=args.manifest_output,
    )
    print(f"Market rows: {result.market_rows}")
    print(f"Volatility rows: {result.volatility_rows}")
    print(f"Market file: {result.market_path}")
    print(f"Volatility file: {result.volatility_path}")
    print(f"Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
