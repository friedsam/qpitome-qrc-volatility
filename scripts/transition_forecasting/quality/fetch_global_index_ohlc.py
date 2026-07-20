from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATASET = "guillemservera/global-stock-indices-historical-data"
DATASET_URL = "https://www.kaggle.com/datasets/guillemservera/global-stock-indices-historical-data"
LICENSE = "CC-BY-NC-4.0"
DEFAULT_DESTINATION = Path("data/raw/transition_forecasting/global_stock_indices_historical_data")
DEFAULT_FALLBACK = Path("data/fallback/transition_forecasting/global_stock_indices_historical_data")
REQUIRED_COMBINED_COLUMNS = {"date", "open", "high", "low", "close", "ticker"}
REQUIRED_INDEX_COLUMNS = {"date", "open", "high", "low", "close"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_inventory(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"raw_acquisition_manifest.json", ".DS_Store"}:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _normalized_columns(path: Path) -> set[str]:
    frame = pd.read_csv(path, nrows=5)
    return {str(column).strip().lower() for column in frame.columns}


def validate_source(root: Path) -> dict[str, object]:
    combined = root / "all_indices_data.csv"
    individual_root = root / "individual_indices_data"
    if not combined.is_file():
        raise ValueError(f"missing required combined panel: {combined}")
    if not individual_root.is_dir():
        raise ValueError(f"missing required individual-index directory: {individual_root}")

    combined_columns = _normalized_columns(combined)
    missing_combined = REQUIRED_COMBINED_COLUMNS.difference(combined_columns)
    if missing_combined:
        raise ValueError(f"{combined}: missing columns {sorted(missing_combined)}")

    index_files = sorted(individual_root.glob("*.csv"))
    if len(index_files) < 30:
        raise ValueError(f"expected at least 30 individual index CSV files, found {len(index_files)}")

    schema_failures: list[dict[str, object]] = []
    for path in index_files:
        columns = _normalized_columns(path)
        missing = REQUIRED_INDEX_COLUMNS.difference(columns)
        if missing:
            schema_failures.append({"path": str(path), "missing": sorted(missing)})
    if schema_failures:
        raise ValueError(f"individual-index schema failures: {schema_failures[:5]}")

    records = file_inventory(root)
    if not records:
        raise ValueError(f"no source files found under {root}")
    return {
        "combined_panel": combined.relative_to(root).as_posix(),
        "individual_index_csv_files": len(index_files),
        "file_count": len(records),
        "files": records,
    }


def verify_fallback_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "fallback_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"fallback manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
    }
    if not expected:
        raise ValueError(f"fallback manifest contains no file hashes: {manifest_path}")

    actual = {item["path"]: item["sha256"] for item in file_inventory(root)}
    actual.pop("fallback_manifest.json", None)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(path for path in set(expected) & set(actual) if expected[path] != actual[path])
    if missing or extra or changed:
        raise ValueError(
            "fallback hash verification failed: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
        )
    return {
        "manifest_path": str(manifest_path),
        "verified_files": len(expected),
        "manifest_sha256": sha256_file(manifest_path),
    }


def download_live(destination: Path) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("Kaggle CLI not found")
    destination.mkdir(parents=True, exist_ok=False)
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
        capture_output=True,
        text=True,
    )


def copy_source(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".DS_Store", "raw_acquisition_manifest.json"))


def install_candidate(candidate: Path, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; use --force only when intentionally replacing the raw snapshot"
            )
        shutil.rmtree(destination)
    candidate.rename(destination)


def acquire(
    destination: Path,
    fallback: Path,
    *,
    source_mode: str,
    force: bool,
) -> dict[str, object]:
    started = utc_now()
    live_error: str | None = None
    selected_mode: str | None = None
    fallback_verification: dict[str, object] | None = None

    if destination.exists() and any(destination.iterdir()) and not force:
        validation = validate_source(destination)
        selected_mode = "existing"
    else:
        with tempfile.TemporaryDirectory(prefix="global-index-raw-") as temporary:
            temp_root = Path(temporary)
            candidate = temp_root / "candidate"

            if source_mode in {"auto", "live"}:
                try:
                    download_live(candidate)
                    validation = validate_source(candidate)
                    selected_mode = "live"
                except Exception as exc:
                    live_error = f"{type(exc).__name__}: {exc}"
                    if candidate.exists():
                        shutil.rmtree(candidate)
                    if source_mode == "live":
                        raise

            if selected_mode is None:
                fallback_verification = verify_fallback_manifest(fallback)
                validate_source(fallback)
                copy_source(fallback, candidate)
                validation = validate_source(candidate)
                selected_mode = "fallback"

            install_candidate(candidate, destination, force=force)

    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "source_description": "Daily global stock-index OHLCV data sourced by the dataset author from Yahoo Finance.",
        "source_mode_requested": source_mode,
        "source_mode_used": selected_mode,
        "live_retrieval_error": live_error,
        "fallback_path": str(fallback),
        "fallback_verification": fallback_verification,
        "destination": str(destination),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "validation": validation,
        "redistribution_note": "Non-commercial attribution license; preserve source and acquisition manifests with retained copies.",
    }
    manifest_path = destination / "raw_acquisition_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire and verify the global stock-index raw OHLC snapshot, using the repository fallback when needed."
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--fallback", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--source-mode", choices=("auto", "live", "fallback"), default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination only after a new candidate has passed source validation.",
    )
    args = parser.parse_args()

    manifest = acquire(
        args.destination,
        args.fallback,
        source_mode=args.source_mode,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
