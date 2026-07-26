from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from transition_forecasting.data.acquisition import (
    DATASET,
    DATASET_URL,
    LICENSE,
    compare_source_to_frozen_inventory,
    file_inventory,
    sha256_file,
    utc_now,
    validate_source,
    verify_fallback_manifest,
)

SOURCE_MANIFEST_NAME = "source_manifest.json"


def active_environment_executable(
    name: str,
    *,
    python_executable: str | Path | None = None,
) -> str | None:
    """Prefer an executable installed beside the active Python interpreter."""

    raw_executable = str(python_executable or sys.executable or "")
    if raw_executable:
        candidate = Path(raw_executable).with_name(name)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def expose_active_environment_executable(name: str) -> str | None:
    """Make the active environment's executable discoverable by PATH consumers."""

    executable = active_environment_executable(name)
    if executable is None:
        return None
    parent = str(Path(executable).parent)
    path_entries = [
        entry for entry in os.environ.get("PATH", "").split(os.pathsep) if entry
    ]
    if parent not in path_entries:
        os.environ["PATH"] = os.pathsep.join([parent, *path_entries])
    return executable


@contextmanager
def destination_filesystem_tempdir(destination: Path) -> Iterator[None]:
    """Place tempfile-backed acquisition staging beside the final destination.

    ``Path.rename`` is atomic only within one filesystem. qBraid mounts the
    workspace separately from ``/tmp``, so acquisition staging must be created
    under the destination parent to avoid ``EXDEV`` cross-device failures.
    """

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous = tempfile.tempdir
    tempfile.tempdir = str(destination.parent)
    try:
        yield
    finally:
        tempfile.tempdir = previous


def verify_submission_fallback(
    fallback_root: Path,
    frozen_inventory: Path,
) -> dict[str, object]:
    """Verify a fallback against both its manifest and the frozen raw contract."""

    fallback_root = Path(fallback_root)
    frozen_inventory = Path(frozen_inventory)
    manifest_verification = verify_fallback_manifest(fallback_root)
    source_validation = validate_source(fallback_root)
    frozen_comparison = compare_source_to_frozen_inventory(
        fallback_root,
        frozen_inventory,
    )
    if not frozen_comparison.get("matched", False):
        raise ValueError(
            "fallback does not match the frozen raw inventory: "
            + json.dumps(frozen_comparison, sort_keys=True)
        )
    return {
        "manifest_verification": manifest_verification,
        "source_validation": source_validation,
        "frozen_inventory_comparison": frozen_comparison,
    }


def write_submission_fallback_manifest(
    fallback_root: Path,
    frozen_inventory: Path,
) -> dict[str, object]:
    """Create a fallback manifest only after exact frozen-contract verification."""

    fallback_root = Path(fallback_root)
    frozen_inventory = Path(frozen_inventory)
    source_validation = validate_source(fallback_root)
    frozen_comparison = compare_source_to_frozen_inventory(
        fallback_root,
        frozen_inventory,
    )
    if not frozen_comparison.get("matched", False):
        raise ValueError(
            "refusing to write fallback manifest for a snapshot that differs from "
            "the frozen raw inventory: "
            + json.dumps(frozen_comparison, sort_keys=True)
        )

    source_manifest = fallback_root / SOURCE_MANIFEST_NAME
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"fallback source attribution manifest is missing: {source_manifest}"
        )

    files = file_inventory(fallback_root)
    manifest_path = fallback_root / "fallback_manifest.json"
    payload = {
        "schema_version": 2,
        "dataset": DATASET,
        "dataset_url": DATASET_URL,
        "license": LICENSE,
        "created_at_utc": utc_now(),
        "source_manifest": SOURCE_MANIFEST_NAME,
        "source_manifest_sha256": sha256_file(source_manifest),
        "frozen_inventory": str(frozen_inventory),
        "frozen_inventory_sha256": sha256_file(frozen_inventory),
        "frozen_inventory_comparison": frozen_comparison,
        "source_validation": source_validation,
        "files": files,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    verification = verify_submission_fallback(fallback_root, frozen_inventory)
    return {
        "manifest": str(manifest_path),
        "files": len(files),
        "verification": verification,
    }


def preserve_source_manifest(source_root: Path, destination_root: Path) -> bool:
    """Copy source attribution metadata when acquisition omitted it."""

    source = Path(source_root) / SOURCE_MANIFEST_NAME
    destination = Path(destination_root) / SOURCE_MANIFEST_NAME
    if not source.is_file() or destination.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def remove_manifest_self_hash(dataset_root: Path) -> bool:
    """Remove the impossible self-hash from a dataset's internal inventory."""

    manifest_path = Path(dataset_root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or "manifest.json" not in files:
        return False
    files.pop("manifest.json")
    manifest["manifest_hash_scope"] = (
        "manifest.json is excluded from its own internal inventory; the outer "
        "run manifest hashes the final published manifest."
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return True
