#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data/raw/transition_forecasting/global_stock_indices_historical_data"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data/processed/transition_forecasting/builds"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(argv) + "\n\n")
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}; see {log_path}")


def require_files(root: Path, relative_paths: list[str]) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            inventory[relative] = {"exists": False}
        else:
            inventory[relative] = {
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    if missing:
        raise RuntimeError(f"Processed build is incomplete; missing files: {missing}")
    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one immutable, submission-grade transition-forecasting processed dataset. "
            "All stages are written under one build directory; existing builds are never overwritten."
        )
    )
    parser.add_argument("--build-id", required=True)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--expected-structural-flags", type=int, default=12)
    parser.add_argument("--expected-affected-indices", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root if args.raw_root.is_absolute() else REPO_ROOT / args.raw_root
    processed_root = args.processed_root if args.processed_root.is_absolute() else REPO_ROOT / args.processed_root
    final_root = processed_root / args.build_id
    staging_root = processed_root / f".{args.build_id}.staging"

    if final_root.exists():
        raise SystemExit(f"Refusing to overwrite existing processed build: {final_root}")
    if staging_root.exists():
        raise SystemExit(
            f"Staging directory already exists from an interrupted build: {staging_root}. "
            "Inspect it before removing it."
        )

    individual_raw = raw_root / "individual_indices_data"
    if not individual_raw.is_dir():
        raise SystemExit(f"Missing raw individual-index directory: {individual_raw}")

    staging_root.mkdir(parents=True, exist_ok=False)
    logs = staging_root / "logs"
    manifest_path = staging_root / "build_manifest.json"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "build_id": args.build_id,
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "raw_root": str(raw_root),
        "processed_root": str(processed_root),
        "expected_structural_flags": args.expected_structural_flags,
        "expected_affected_indices": args.expected_affected_indices,
        "rules": {
            "raw_immutable": True,
            "interpolation": False,
            "forward_fill": False,
            "winsorization": False,
            "arbitrary_clipping": False,
            "synthetic_dates": False,
            "structural_bad_prints_removed_before_volatility": True,
            "controls_rematched_after_correction": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        canonical_root = staging_root / "canonical_ohlc"
        inventory_root = staging_root / "quality/global_index_ohlc_audit"
        range_root = staging_root / "quality/global_ohlc_range_quality"
        catalogue_root = staging_root / "catalogue/global_transition_catalogue"
        stage_d_root = staging_root / "modeling/global_stage_d_dataset"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/quality/build_canonical_clean_ohlc.py",
                "--raw-root",
                str(individual_raw),
                "--output-root",
                str(canonical_root),
                "--expected-structural-flags",
                str(args.expected_structural_flags),
                "--expected-affected-indices",
                str(args.expected_affected_indices),
            ],
            logs / "01_build_canonical_clean_ohlc.log",
        )

        canonical_data = canonical_root / "individual_indices_data"
        run(
            [
                sys.executable,
                "scripts/transition_forecasting/quality/audit_global_index_ohlc.py",
                "--data-dir",
                str(canonical_data),
                "--out-dir",
                str(inventory_root),
                "--run-id",
                "build",
            ],
            logs / "02_audit_global_index_ohlc.log",
        )
        inventory = inventory_root / "build/global_index_ohlc_inventory.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/quality/audit_global_ohlc_range_quality.py",
                "--inventory",
                str(inventory),
                "--out-dir",
                str(range_root),
                "--run-id",
                "build",
            ],
            logs / "03_audit_global_ohlc_range_quality.log",
        )
        range_quality = range_root / "build/global_range_quality.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/catalogue/build_global_transition_catalogue.py",
                "--data-dir",
                str(canonical_data),
                "--inventory",
                str(inventory),
                "--range-quality",
                str(range_quality),
                "--out-dir",
                str(catalogue_root),
                "--run-id",
                "build",
            ],
            logs / "04_build_global_transition_catalogue.log",
        )
        representative = catalogue_root / "build/representative_transition_catalogue.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/modeling/build_global_stage_d_dataset.py",
                "--representative-catalogue",
                str(representative),
                "--inventory",
                str(inventory),
                "--out-dir",
                str(stage_d_root),
                "--run-id",
                "build",
            ],
            logs / "05_build_global_stage_d_dataset.log",
        )

        required = [
            "canonical_ohlc/manifest.json",
            "canonical_ohlc/row_corrections.csv",
            "canonical_ohlc/file_manifest.csv",
            "quality/global_index_ohlc_audit/build/global_index_ohlc_inventory.csv",
            "quality/global_ohlc_range_quality/build/global_range_quality.csv",
            "catalogue/global_transition_catalogue/build/raw_transition_catalogue.csv",
            "catalogue/global_transition_catalogue/build/representative_transition_catalogue.csv",
            "modeling/global_stage_d_dataset/build/sample_manifest.csv",
            "modeling/global_stage_d_dataset/build/sequence_tensors.npz",
            "modeling/global_stage_d_dataset/build/summary.json",
        ]
        outputs = require_files(staging_root, required)

        correction_manifest = json.loads((canonical_root / "manifest.json").read_text(encoding="utf-8"))
        if correction_manifest["structural_removed_rows"] != args.expected_structural_flags:
            raise RuntimeError("Structural removed-row count changed after canonical build")
        if correction_manifest["affected_indices"] != args.expected_affected_indices:
            raise RuntimeError("Affected-index count changed after canonical build")

        manifest.update(
            {
                "status": "succeeded",
                "finished_at_utc": utc_now(),
                "outputs": outputs,
                "correction_summary": correction_manifest,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        processed_root.mkdir(parents=True, exist_ok=True)
        os.replace(staging_root, final_root)
        print(f"Processed build completed: {final_root}")
        print(f"Manifest: {final_root / 'build_manifest.json'}")
        return 0
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "finished_at_utc": utc_now(),
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Processed build failed: {exc}", file=sys.stderr)
        print(f"Preserved staging directory for inspection: {staging_root}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
