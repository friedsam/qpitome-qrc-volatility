#!/usr/bin/env python3
"""Audit repository-maintenance policy and result-migration metadata.

This audit is intentionally non-destructive. It validates the maintenance
contract and reports repository conditions that require human review.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "AGENTS.md"
MANIFEST = ROOT / "docs/result_migration_manifest.csv"
PROVENANCE_DOC = ROOT / "docs/results_layout_and_provenance.md"

REQUIRED_COLUMNS = [
    "old_path",
    "new_path",
    "producer_script",
    "consumer_scripts",
    "historical_or_new_default",
    "status",
    "provenance_confidence",
    "notes",
]
ALLOWED_STATUS = {
    "planned",
    "migrated",
    "preserved",
    "missing",
    "unverified",
    "regenerated",
    "retired",
}
ALLOWED_ORIGIN = {"historical", "new_default", "both", "unknown"}
ALLOWED_CONFIDENCE = {"high", "medium", "low", "unknown"}
TEMPLATE_MARKERS = ("<", ">", "*", "?")


def is_template(path: str) -> bool:
    return any(marker in path for marker in TEMPLATE_MARKERS)


def repository_path(path: str) -> Path:
    return ROOT / path


def validate_repo_relative(path: str, field: str, row_number: int, errors: list[str]) -> None:
    if not path:
        return
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        errors.append(f"row {row_number}: {field} must be a repository-relative path: {path}")


def read_manifest(errors: list[str]) -> list[dict[str, str]]:
    if not MANIFEST.exists():
        errors.append(f"missing manifest: {MANIFEST.relative_to(ROOT)}")
        return []

    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != REQUIRED_COLUMNS:
            errors.append(
                "manifest columns must exactly equal: " + ",".join(REQUIRED_COLUMNS)
            )
            return []
        return list(reader)


def validate_manifest(rows: list[dict[str, str]], errors: list[str], warnings: list[str]) -> None:
    old_paths: list[str] = []
    destination_counts: Counter[str] = Counter()

    for row_number, row in enumerate(rows, start=2):
        old_path = row["old_path"].strip()
        new_path = row["new_path"].strip()
        producer = row["producer_script"].strip()
        consumers = [item.strip() for item in row["consumer_scripts"].split(";") if item.strip()]
        status = row["status"].strip()
        origin = row["historical_or_new_default"].strip()
        confidence = row["provenance_confidence"].strip()
        notes = row["notes"].strip()

        if not old_path:
            errors.append(f"row {row_number}: old_path is required")
        else:
            old_paths.append(old_path)
            validate_repo_relative(old_path, "old_path", row_number, errors)

        validate_repo_relative(new_path, "new_path", row_number, errors)
        validate_repo_relative(producer, "producer_script", row_number, errors)
        for consumer in consumers:
            validate_repo_relative(consumer, "consumer_scripts", row_number, errors)

        if status not in ALLOWED_STATUS:
            errors.append(f"row {row_number}: invalid status {status!r}")
        if origin not in ALLOWED_ORIGIN:
            errors.append(f"row {row_number}: invalid historical_or_new_default {origin!r}")
        if confidence not in ALLOWED_CONFIDENCE:
            errors.append(f"row {row_number}: invalid provenance_confidence {confidence!r}")
        if not notes:
            errors.append(f"row {row_number}: notes are required")

        if status in {"migrated", "regenerated"} and not new_path:
            errors.append(f"row {row_number}: status {status!r} requires new_path")
        if status == "migrated" and new_path:
            destination_counts[new_path] += 1
            if not is_template(new_path) and not repository_path(new_path).exists():
                errors.append(f"row {row_number}: migrated destination does not exist: {new_path}")
        if status == "missing" and new_path:
            warnings.append(
                f"row {row_number}: missing historical artifact also has a new_path; "
                "verify that the destination is clearly marked as a later default or regeneration"
            )

        if producer and not is_template(producer) and not repository_path(producer).exists():
            errors.append(f"row {row_number}: producer script does not exist: {producer}")
        for consumer in consumers:
            if not is_template(consumer) and not repository_path(consumer).exists():
                errors.append(f"row {row_number}: consumer script does not exist: {consumer}")

    duplicate_old = sorted(path for path, count in Counter(old_paths).items() if count > 1)
    if duplicate_old:
        errors.append("duplicate old_path entries: " + ", ".join(duplicate_old))

    duplicate_destinations = sorted(path for path, count in destination_counts.items() if count > 1)
    if duplicate_destinations:
        errors.append(
            "multiple migrated rows target the same destination: " + ", ".join(duplicate_destinations)
        )


def scan_nested_modeling_directories(warnings: list[str]) -> None:
    modeling = ROOT / "results/modeling"
    if not modeling.exists():
        return

    nested = []
    for path in modeling.rglob("*"):
        if not path.is_dir():
            continue
        relative = path.relative_to(modeling)
        if len(relative.parts) >= 3:
            nested.append(str(path.relative_to(ROOT)))

    if nested:
        warnings.append(
            "nested result directories remain and require explicit manifest-backed review:\n  "
            + "\n  ".join(sorted(nested))
        )


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for required in (POLICY, PROVENANCE_DOC):
        if not required.exists():
            errors.append(f"missing required maintenance document: {required.relative_to(ROOT)}")

    rows = read_manifest(errors)
    if rows:
        validate_manifest(rows, errors, warnings)
    else:
        errors.append("result migration manifest contains no data rows")

    scan_nested_modeling_directories(warnings)

    print("Repository maintenance audit")
    print(f"Manifest rows: {len(rows)}")

    if warnings:
        print("\nWARNINGS")
        for warning in warnings:
            print(f"- {warning}")

    if errors:
        print("\nERRORS")
        for error in errors:
            print(f"- {error}")
        return 1

    print("\nPASS: maintenance policy and migration manifest are structurally valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
