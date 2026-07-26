from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

PACKAGE_FILES = {
    "run_manifest.json": "run_manifest.json",
    "raw_acquisition_manifest.json": (
        "files/transition_forecasting/raw/global_stock_indices_historical_data/"
        "raw_acquisition_manifest.json"
    ),
    "dataset_manifest.json": (
        "files/transition_forecasting/processed/global_transition_dataset_1d/"
        "manifest.json"
    ),
    "fold_summary.json": (
        "files/transition_forecasting/processed/global_transition_dataset_1d/"
        "purged_walk_forward_folds/summary.json"
    ),
    "data_pipeline_audit.json": (
        "files/transition_forecasting/validation/data_pipeline_audit.json"
    ),
    "data_pipeline_checksums.json": (
        "files/transition_forecasting/validation/data_pipeline_checksums.json"
    ),
}

INDEX_COLUMNS = (
    "experiment_name",
    "run_id",
    "commit",
    "platform",
    "status",
    "aggregate_path",
    "evidence_path",
    "headline_artifact",
    "notes",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _duration_seconds(started: str | None, finished: str | None) -> float | None:
    if not started or not finished:
        return None
    return round(
        (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds(),
        6,
    )


def _validate_run(
    run_root: Path,
    run_manifest: dict[str, Any],
    raw_manifest: dict[str, Any],
    audit: dict[str, Any],
    checksums: dict[str, Any],
) -> None:
    if run_manifest.get("workflow") != "transition-data":
        raise ValueError("only transition-data runs can be packaged")
    if run_manifest.get("status") != "succeeded":
        raise ValueError("run status is not succeeded")

    commands = run_manifest.get("commands")
    if not isinstance(commands, list) or not commands:
        raise ValueError("run manifest contains no command records")
    failed = [
        index
        for index, command in enumerate(commands, start=1)
        if not isinstance(command, dict) or command.get("returncode") != 0
    ]
    if failed:
        raise ValueError(f"run contains failed command records: {failed}")

    required = run_manifest.get("required_outputs")
    if not isinstance(required, dict) or not required:
        raise ValueError("run manifest contains no required-output inventory")
    missing = [
        path
        for path, item in required.items()
        if not isinstance(item, dict) or not item.get("exists", False)
    ]
    if missing:
        raise ValueError(f"required outputs are missing: {missing}")

    if raw_manifest.get("authoritative_source_verified") is not True:
        raise ValueError("raw source was not verified against its authority")
    if audit.get("passed") is not True:
        raise ValueError("data pipeline audit did not pass")
    if checksums.get("passed") is not True:
        raise ValueError("data pipeline checksum report did not pass")

    for packaged_name, relative in PACKAGE_FILES.items():
        source = run_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"missing package source {packaged_name}: {source}")


def _write_command_runtimes(path: Path, commands: list[dict[str, Any]]) -> None:
    columns = (
        "command_index",
        "argv",
        "started_at_utc",
        "finished_at_utc",
        "duration_seconds",
        "returncode",
        "log_path",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, command in enumerate(commands, start=1):
            writer.writerow(
                {
                    "command_index": index,
                    "argv": json.dumps(command.get("argv", []), separators=(",", ":")),
                    "started_at_utc": command.get("started_at_utc"),
                    "finished_at_utc": command.get("finished_at_utc"),
                    "duration_seconds": command.get("duration_seconds"),
                    "returncode": command.get("returncode"),
                    "log_path": command.get("log_path"),
                }
            )


def _write_artifact_inventory(
    path: Path,
    required_outputs: dict[str, dict[str, Any]],
) -> None:
    columns = ("path", "exists", "size_bytes", "sha256")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for artifact_path, item in sorted(required_outputs.items()):
            writer.writerow(
                {
                    "path": artifact_path,
                    "exists": item.get("exists"),
                    "size_bytes": item.get("size_bytes"),
                    "sha256": item.get("sha256"),
                }
            )


def _write_index(index_path: Path, row: dict[str, str]) -> None:
    existing: list[dict[str, str]] = []
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != INDEX_COLUMNS:
                raise ValueError(
                    f"unexpected result-index columns in {index_path}: {reader.fieldnames}"
                )
            existing = [dict(item) for item in reader]

    existing = [item for item in existing if item.get("run_id") != row["run_id"]]
    existing.append(row)
    existing.sort(key=lambda item: (item.get("experiment_name", ""), item.get("run_id", "")))

    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(existing)
    temporary.replace(index_path)


def package_data_pipeline_run(
    *,
    run_root: Path,
    output_root: Path,
    index_path: Path,
    experiment_name: str,
    platform_name: str,
) -> dict[str, Any]:
    run_root = Path(run_root)
    output_root = Path(output_root)
    index_path = Path(index_path)

    run_manifest = _load_json(run_root / "run_manifest.json")
    run_id = str(run_manifest.get("run_id") or run_root.name)
    if run_id != run_root.name:
        raise ValueError(
            f"run ID mismatch: directory={run_root.name!r}, manifest={run_id!r}"
        )

    raw_manifest = _load_json(run_root / PACKAGE_FILES["raw_acquisition_manifest.json"])
    dataset_manifest = _load_json(run_root / PACKAGE_FILES["dataset_manifest.json"])
    fold_summary = _load_json(run_root / PACKAGE_FILES["fold_summary.json"])
    audit = _load_json(run_root / PACKAGE_FILES["data_pipeline_audit.json"])
    checksums = _load_json(run_root / PACKAGE_FILES["data_pipeline_checksums.json"])
    _validate_run(run_root, run_manifest, raw_manifest, audit, checksums)

    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(f"evidence package already exists: {destination}")
    destination.mkdir(parents=True)

    for packaged_name, relative in PACKAGE_FILES.items():
        shutil.copy2(run_root / relative, destination / packaged_name)

    commands = run_manifest["commands"]
    required_outputs = run_manifest["required_outputs"]
    _write_command_runtimes(destination / "command_runtimes.csv", commands)
    _write_artifact_inventory(destination / "artifact_inventory.csv", required_outputs)

    environment = {
        "repository": run_manifest.get("repository"),
        "environment": run_manifest.get("environment"),
        "parameters": run_manifest.get("parameters"),
    }
    (destination / "environment.json").write_text(
        json.dumps(environment, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema_version": 1,
        "experiment_name": experiment_name,
        "run_id": run_id,
        "workflow": run_manifest.get("workflow"),
        "status": run_manifest.get("status"),
        "platform": platform_name,
        "commit": (run_manifest.get("repository") or {}).get("commit"),
        "branch": (run_manifest.get("repository") or {}).get("branch"),
        "started_at_utc": run_manifest.get("started_at_utc"),
        "finished_at_utc": run_manifest.get("finished_at_utc"),
        "duration_seconds": _duration_seconds(
            run_manifest.get("started_at_utc"),
            run_manifest.get("finished_at_utc"),
        ),
        "source_mode_requested": raw_manifest.get("source_mode_requested"),
        "source_mode_used": raw_manifest.get("source_mode_used"),
        "authoritative_source_verified": raw_manifest.get(
            "authoritative_source_verified"
        ),
        "fallback_substitution": raw_manifest.get("fallback_substitution"),
        "dataset_counts": dataset_manifest.get("counts"),
        "transition_summary": dataset_manifest.get("transition_summary"),
        "stage_d_summary": dataset_manifest.get("stage_d_summary"),
        "fold_summary": fold_summary,
        "data_pipeline_audit_passed": audit.get("passed"),
        "checksum_report_passed": checksums.get("passed"),
        "test_evaluated": dataset_manifest.get("test_evaluated"),
        "aggregate_path": run_manifest.get("run_directory"),
        "evidence_path": destination.as_posix(),
        "required_output_count": len(required_outputs),
    }
    (destination / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = f"""# Verified transition-data run: `{run_id}`

- **Experiment:** `{experiment_name}`
- **Platform:** `{platform_name}`
- **Commit:** `{summary['commit']}`
- **Status:** `{summary['status']}`
- **Source mode:** requested `{summary['source_mode_requested']}`, used `{summary['source_mode_used']}`
- **Authoritative source verified:** `{summary['authoritative_source_verified']}`
- **Fallback substitution:** `{summary['fallback_substitution']}`
- **Data audit passed:** `{summary['data_pipeline_audit_passed']}`
- **Checksum report passed:** `{summary['checksum_report_passed']}`
- **Test set evaluated:** `{summary['test_evaluated']}`
- **Aggregate working run:** `{summary['aggregate_path']}`

## Reproduction

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \\
  --run-id {run_id} \\
  --transition-source-mode fallback
```

The large aggregate run remains the authoritative execution record. This directory
contains compact manifests, validation reports, command timings, and the required
output inventory needed to inspect and map the run without committing generated
tensors or reconstructed datasets.
"""
    (destination / "README.md").write_text(readme, encoding="utf-8")

    package_files = sorted(
        item for item in destination.iterdir() if item.is_file() and item.name != "package_manifest.json"
    )
    package_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "files": [
            {
                "path": item.name,
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
            for item in package_files
        ],
    }
    (destination / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    index_row = {
        "experiment_name": experiment_name,
        "run_id": run_id,
        "commit": str(summary["commit"] or ""),
        "platform": platform_name,
        "status": "verified",
        "aggregate_path": str(summary["aggregate_path"] or ""),
        "evidence_path": destination.as_posix(),
        "headline_artifact": (destination / "run_summary.json").as_posix(),
        "notes": "fallback source verified; data audit and checksum report passed",
    }
    _write_index(index_path, index_row)

    return {
        "run_id": run_id,
        "destination": destination.as_posix(),
        "index": index_path.as_posix(),
        "packaged_files": len(package_manifest["files"]) + 1,
        "summary": summary,
    }
