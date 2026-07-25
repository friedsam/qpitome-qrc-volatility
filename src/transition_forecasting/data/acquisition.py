from __future__ import annotations

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
SETUP_DOC = "docs/transition_forecasting/kaggle_source_setup.md"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FROZEN_INVENTORY = (
    REPO_ROOT / "config/transition_forecasting/contracts/global_index_ohlc_inventory.csv"
)
REQUIRED_COMBINED_COLUMNS = {"date", "open", "high", "low", "close", "ticker"}
REQUIRED_INDEX_COLUMNS = {"date", "open", "high", "low", "close"}
_METADATA_FILES = {
    "fallback_manifest.json",
    "raw_acquisition_manifest.json",
    "source_manifest.json",
    ".DS_Store",
}


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
        if path.name in _METADATA_FILES:
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def _hash_map(root: Path) -> dict[str, str]:
    return {
        str(item["path"]): str(item["sha256"])
        for item in file_inventory(root)
    }


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

    missing_combined = REQUIRED_COMBINED_COLUMNS.difference(_normalized_columns(combined))
    if missing_combined:
        raise ValueError(f"{combined}: missing columns {sorted(missing_combined)}")

    index_files = sorted(individual_root.glob("*.csv"))
    if len(index_files) < 30:
        raise ValueError(
            f"expected at least 30 individual index CSV files, found {len(index_files)}"
        )

    schema_failures: list[dict[str, object]] = []
    for path in index_files:
        missing = REQUIRED_INDEX_COLUMNS.difference(_normalized_columns(path))
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


def _load_fallback_hashes(root: Path) -> tuple[Path, dict[str, str]]:
    manifest_path = root / "fallback_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"fallback manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("files", [])
        if Path(str(item.get("path", ""))).name not in _METADATA_FILES
    }
    if not expected:
        raise ValueError(f"fallback manifest contains no file hashes: {manifest_path}")
    return manifest_path, expected


def _load_frozen_inventory_hashes(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"frozen raw inventory missing: {path}")
    frame = pd.read_csv(path)
    required = {"path", "sha256"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    expected: dict[str, str] = {}
    for _, row in frame.iterrows():
        source = Path(str(row["path"]))
        relative = (
            Path("all_indices_data.csv")
            if source.name == "all_indices_data.csv"
            else Path("individual_indices_data") / source.name
        )
        expected[relative.as_posix()] = str(row["sha256"])
    if not expected:
        raise ValueError(f"frozen raw inventory contains no file hashes: {path}")
    return expected


def compare_hash_maps(
    actual: dict[str, str],
    expected: dict[str, str],
    *,
    reference: str,
) -> dict[str, object]:
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    changed = sorted(
        path for path in set(expected) & set(actual) if expected[path] != actual[path]
    )
    return {
        "reference": reference,
        "matched": not (missing or extra or changed),
        "expected_files": len(expected),
        "actual_files": len(actual),
        "missing": missing,
        "extra": extra,
        "changed": changed,
    }


def compare_source_to_fallback(source: Path, fallback: Path) -> dict[str, object]:
    manifest_path, expected = _load_fallback_hashes(fallback)
    return compare_hash_maps(
        _hash_map(source),
        expected,
        reference=str(manifest_path),
    )


def compare_source_to_frozen_inventory(
    source: Path,
    inventory_path: Path = DEFAULT_FROZEN_INVENTORY,
) -> dict[str, object]:
    expected = _load_frozen_inventory_hashes(inventory_path)
    return compare_hash_maps(
        _hash_map(source),
        expected,
        reference=str(inventory_path),
    )


def verify_fallback_manifest(root: Path) -> dict[str, object]:
    manifest_path, expected = _load_fallback_hashes(root)
    actual = _hash_map(root)
    comparison = compare_hash_maps(actual, expected, reference=str(manifest_path))
    if not comparison["matched"]:
        raise ValueError(
            "fallback hash verification failed: "
            f"missing={comparison['missing'][:5]}, "
            f"extra={comparison['extra'][:5]}, "
            f"changed={comparison['changed'][:5]}"
        )
    return {
        "manifest_path": str(manifest_path),
        "verified_files": len(expected),
        "manifest_sha256": sha256_file(manifest_path),
        "comparison": comparison,
    }


def write_live_source_manifest(destination: Path) -> None:
    csv_files = sorted(destination.rglob("*.csv"))
    manifest = {
        "schema_version": 1,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "source_description": (
            "Daily global stock-index OHLCV data sourced by the dataset author "
            "from Yahoo Finance."
        ),
        "downloaded_at_utc": utc_now(),
        "csv_files": len(csv_files),
        "redistribution_note": (
            "Non-commercial attribution license; preserve this manifest with any "
            "retained copy."
        ),
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def download_live(destination: Path) -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError(
            "Kaggle CLI not found. Install the project dependencies in the active "
            "environment and verify `which kaggle`. "
            f"Setup: {SETUP_DOC}"
        )
    destination.mkdir(parents=True, exist_ok=False)
    try:
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
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "Kaggle download failed. Public dataset downloads may be anonymous; "
            "if Kaggle requires authentication, configure it securely. "
            f"Setup: {SETUP_DOC}. Kaggle output: {details or '<none>'}"
        ) from exc
    write_live_source_manifest(destination)


def copy_source(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(*_METADATA_FILES),
    )


def install_candidate(candidate: Path, destination: Path, *, force: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not force:
            raise FileExistsError(
                f"{destination} already exists; use --force only when intentionally "
                "replacing the raw snapshot"
            )
        shutil.rmtree(destination)
    candidate.rename(destination)


def _verified_fallback(fallback: Path) -> dict[str, object] | None:
    if not (fallback / "fallback_manifest.json").is_file():
        return None
    verification = verify_fallback_manifest(fallback)
    validate_source(fallback)
    return verification


def acquire_source(
    destination: Path,
    fallback: Path,
    *,
    source_mode: str = "auto",
    force: bool = False,
    frozen_inventory: Path = DEFAULT_FROZEN_INVENTORY,
) -> dict[str, object]:
    if source_mode not in {"auto", "live", "fallback"}:
        raise ValueError(f"unsupported source mode: {source_mode}")

    started = utc_now()
    live_error: str | None = None
    selected_mode: str | None = None
    fallback_verification = _verified_fallback(fallback)
    source_comparison: dict[str, object] | None = None
    fallback_substitution = False
    substitution_reason: str | None = None

    if destination.exists() and any(destination.iterdir()) and not force:
        validation = validate_source(destination)
        source_comparison = (
            compare_source_to_fallback(destination, fallback)
            if fallback_verification is not None
            else compare_source_to_frozen_inventory(destination, frozen_inventory)
        )
        if not source_comparison["matched"]:
            raise RuntimeError(
                "Existing raw source does not match the authoritative reference: "
                + json.dumps(source_comparison, sort_keys=True)
            )
        selected_mode = "existing_verified"
    else:
        with tempfile.TemporaryDirectory(prefix="global-index-raw-") as temporary:
            temp_root = Path(temporary)
            candidate = temp_root / "candidate"

            if source_mode in {"auto", "live"}:
                try:
                    download_live(candidate)
                    validate_source(candidate)
                    source_comparison = (
                        compare_source_to_fallback(candidate, fallback)
                        if fallback_verification is not None
                        else compare_source_to_frozen_inventory(candidate, frozen_inventory)
                    )
                    if source_comparison["matched"]:
                        selected_mode = (
                            "live_verified_against_fallback"
                            if fallback_verification is not None
                            else "live_verified_against_frozen_inventory"
                        )
                    elif fallback_verification is not None:
                        shutil.rmtree(candidate)
                        copy_source(fallback, candidate)
                        validation = validate_source(candidate)
                        selected_mode = "fallback_after_live_mismatch"
                        fallback_substitution = True
                        substitution_reason = (
                            "Live Kaggle snapshot did not match the verified fallback; "
                            "the live candidate was discarded before installation."
                        )
                    else:
                        raise RuntimeError(
                            "Live Kaggle snapshot does not match the frozen raw inventory "
                            "and no verified fallback is available: "
                            + json.dumps(source_comparison, sort_keys=True)
                        )
                except Exception as exc:
                    live_error = f"{type(exc).__name__}: {exc}"
                    if candidate.exists():
                        shutil.rmtree(candidate)
                    if source_mode == "live" and fallback_verification is None:
                        raise

            if selected_mode is None:
                if fallback_verification is None:
                    raise RuntimeError(
                        "Source acquisition failed. "
                        f"Live attempt: {live_error or '<not attempted>'}. "
                        "Verified fallback: unavailable. "
                        f"Kaggle setup: {SETUP_DOC}"
                    )
                copy_source(fallback, candidate)
                validation = validate_source(candidate)
                selected_mode = (
                    "fallback_after_live_failure"
                    if live_error is not None
                    else "fallback"
                )
                fallback_substitution = live_error is not None
                substitution_reason = (
                    "Live Kaggle acquisition failed; the verified fallback was installed."
                    if live_error is not None
                    else None
                )

            install_candidate(candidate, destination, force=force)

    manifest = {
        "schema_version": 2,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "source_description": (
            "Daily global stock-index OHLCV data sourced by the dataset author "
            "from Yahoo Finance."
        ),
        "source_mode_requested": source_mode,
        "source_mode_used": selected_mode,
        "live_retrieval_error": live_error,
        "fallback_path": str(fallback),
        "fallback_verification": fallback_verification,
        "source_comparison": source_comparison,
        "fallback_substitution": fallback_substitution,
        "substitution_reason": substitution_reason,
        "authoritative_source_verified": True,
        "destination": str(destination),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "validation": validation,
        "redistribution_note": (
            "Non-commercial attribution license; preserve source and acquisition "
            "manifests with retained copies."
        ),
    }
    manifest_path = destination / "raw_acquisition_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
