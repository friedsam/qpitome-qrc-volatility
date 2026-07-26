from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from transition_forecasting.data.acquisition import (
    compare_source_to_frozen_inventory,
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

    executable = Path(python_executable or sys.executable)
    candidate = executable.with_name(name)
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return shutil.which(name)


def expose_active_environment_executable(name: str) -> str | None:
    """Make the active environment's executable discoverable by PATH consumers."""

    executable = active_environment_executable(name)
    if executable is None:
        return None
    parent = str(Path(executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if parent not in path_entries:
        os.environ["PATH"] = os.pathsep.join([parent, *path_entries])
    return executable


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
