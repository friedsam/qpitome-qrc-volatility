from __future__ import annotations

import argparse
from pathlib import Path

from data.inventory_raw_datasets import inventory_raw_datasets


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
    results = inventory_raw_datasets(
        args.raw_root,
        output_path=args.output,
        sample_size=args.sample_size,
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
