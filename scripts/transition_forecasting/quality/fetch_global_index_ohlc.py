from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DATASET = "guillemservera/global-stock-indices-historical-data"
DATASET_URL = "https://www.kaggle.com/datasets/guillemservera/global-stock-indices-historical-data"
LICENSE = "CC-BY-NC-4.0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the global stock-indices OHLC dataset from Kaggle.")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/transition_forecasting/global_stock_indices_historical_data"),
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing destination directory.")
    args = parser.parse_args()

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit(
            "Kaggle CLI not found. Install it with `python -m pip install kaggle`, "
            "then configure Kaggle credentials before rerunning."
        )

    destination = args.destination
    if destination.exists() and any(destination.iterdir()):
        if not args.force:
            raise SystemExit(f"{destination} is not empty; rerun with --force to replace it.")
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            kaggle,
            "datasets",
            "download",
            "--dataset",
            DATASET,
            "--path",
            str(destination),
            "--unzip",
        ],
        check=True,
    )

    csv_files = sorted(destination.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"Download completed but no CSV files were found under {destination}")

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "source_description": "Daily global stock-index OHLCV data sourced by the dataset author from Yahoo Finance.",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "destination": str(destination),
        "csv_files": len(csv_files),
        "redistribution_note": "Non-commercial attribution license; preserve this manifest with any retained copy.",
    }
    (destination / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
