#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

RELATIVE_FILES = (
    "sample_manifest.csv",
    "sequence_tensors.npz",
    "manifest.json",
    "control_candidate_manifest.csv",
    "control_candidate_tensors.npz",
    "candidate_pool_summary.json",
    "purged_walk_forward_folds/rematched_rolling_manifest.csv",
    "purged_walk_forward_folds/rematched_rolling_tensors.npz",
    "purged_walk_forward_folds/control_match_audit.csv",
    "purged_walk_forward_folds/summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write SHA-256 fingerprints for the canonical 1D transition data stage."
    )
    parser.add_argument("--dataset-1d", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {
        "schema_version": 2,
        "dataset": {
            "label": "1d",
            "root": str(args.dataset_1d),
            "files": {},
        },
        "three_channel_dataset_built": False,
    }
    missing: list[str] = []

    files: dict[str, object] = {}
    for relative in RELATIVE_FILES:
        path = args.dataset_1d / relative
        exists = path.is_file()
        item: dict[str, object] = {"exists": exists}
        if exists:
            item.update(
                {
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        else:
            item.update({"size_bytes": None, "sha256": None})
            missing.append(str(path))
        files[relative] = item
    report["dataset"]["files"] = files
    report["files_checked"] = len(RELATIVE_FILES)
    report["missing_files"] = missing
    report["passed"] = not missing

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
