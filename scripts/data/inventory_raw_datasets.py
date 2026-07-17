from __future__ import annotations

import argparse
from pathlib import Path

from data.inventory_raw_datasets import inventory_raw_datasets


CANONICAL_REQUIRED_RELATIVE_PATHS = [
    "yahoo_daily_history/historical_market_data.csv",
    "yahoo_daily_history/historical_volatility_data.csv",
    "macro_fred_monthly/fred_TB3MS_three_month_tbill.csv",
    "macro_fred_monthly/fred_CPIAUCSL_cpi.csv",
    "macro_fred_monthly/fred_INDPRO_industrial_production.csv",
    "macro_fred_monthly/fred_AAA_aaa_corporate_yield.csv",
    "macro_fred_monthly/fred_BAA_baa_corporate_yield.csv",
    "equity_factor_returns_monthly/ff3_monthly.zip",
    "equity_factor_returns_monthly/short_term_reversal_monthly.zip",
    "international_equity_indices/nikkei_225_fred_raw.csv",
    "international_equity_indices/russell_2000_yahoo_raw.csv",
    "international_equity_indices/ftse_100_yahoo_raw.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect canonical raw CSV and ZIP datasets before standardization"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/raw_dataset_inventory.json"),
    )
    parser.add_argument("--sample-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required_paths = [
        args.raw_root / relative_path
        for relative_path in CANONICAL_REQUIRED_RELATIVE_PATHS
    ]
    results = inventory_raw_datasets(
        args.raw_root,
        output_path=args.output,
        sample_size=args.sample_size,
        required_paths=required_paths,
    )
    print(f"Inventoried {len(results)} raw data files")
    for result in results:
        print(f"{result.path}: {result.size_bytes} bytes")
        if result.columns:
            print(f"  columns: {result.columns}")
        if result.zip_members:
            print(f"  zip members: {result.zip_members}")
        if result.error:
            print(f"  inspection error: {result.error}")
    print(f"Inventory: {args.output}")


if __name__ == "__main__":
    main()
