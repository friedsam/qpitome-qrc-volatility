#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.acquisition import (
    DEFAULT_FROZEN_INVENTORY,
    acquire_source,
)
from transition_forecasting.data.submission_provenance import (
    expose_active_environment_executable,
    preserve_source_manifest,
    verify_submission_fallback,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FALLBACK = (
    REPO_ROOT
    / "data/fallback/transition_forecasting/global_stock_indices_historical_data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire and verify the global stock-index OHLC source snapshot."
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument(
        "--source-mode",
        choices=("auto", "live", "fallback"),
        default="auto",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expose_active_environment_executable("kaggle")

    fallback_verification = None
    if (args.fallback / "fallback_manifest.json").is_file():
        fallback_verification = verify_submission_fallback(
            args.fallback,
            DEFAULT_FROZEN_INVENTORY,
        )

    report = acquire_source(
        args.destination,
        args.fallback,
        source_mode=args.source_mode,
        force=args.force,
    )
    report["submission_fallback_verification"] = fallback_verification
    report["source_manifest_preserved_from_fallback"] = preserve_source_manifest(
        args.fallback,
        args.destination,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
