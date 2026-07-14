from __future__ import annotations

import argparse
from pathlib import Path

from data.validate_volatility_dataset import validate_volatility_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a processed volatility dataset")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/spy_vix_volatility/spy_vix_volatility.csv"),
    )
    parser.add_argument("--reference", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_volatility_dataset(args.dataset, reference_path=args.reference)
    print(f"Validated: {args.dataset}")
    print(f"Shape: ({result.rows}, {result.columns})")
    print(f"Date range: {result.date_start} -> {result.date_end}")
    if result.reference_overlap_rows is not None:
        print(f"Reference overlap rows: {result.reference_overlap_rows}")
        print(f"Reference numeric MAE: {result.reference_numeric_mae}")


if __name__ == "__main__":
    main()
