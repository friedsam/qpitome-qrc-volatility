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
    destination_filesystem_tempdir,
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


def _persist_submission_provenance(
    destination: Path,
    report: dict[str, object],
) -> None:
    manifest_path = destination / "raw_acquisition_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"acquisition completed without required manifest: {manifest_path}"
        )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def main() -> int:
    args = parse_args()
    expose_active_environment_executable("kaggle")

    fallback_verification = None
    if (args.fallback / "fallback_manifest.json").is_file():
        fallback_verification = verify_submission_fallback(
            args.fallback,
            DEFAULT_FROZEN_INVENTORY,
        )

    with destination_filesystem_tempdir(args.destination):
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
    _persist_submission_provenance(args.destination, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
