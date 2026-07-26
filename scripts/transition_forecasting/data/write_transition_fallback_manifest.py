#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.acquisition import DEFAULT_FROZEN_INVENTORY
from transition_forecasting.data.submission_provenance import (
    write_submission_fallback_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FALLBACK = (
    REPO_ROOT
    / "data/fallback/transition_forecasting/global_stock_indices_historical_data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the transition-data fallback against the frozen raw contract "
            "and write its fallback manifest."
        )
    )
    parser.add_argument("--fallback-root", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument(
        "--frozen-inventory",
        type=Path,
        default=DEFAULT_FROZEN_INVENTORY,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = write_submission_fallback_manifest(
        args.fallback_root,
        args.frozen_inventory,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
