#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.global_transition_catalogue import _load_generic_ohlc
from transition_forecasting.catalogue.transition_events import log_parkinson

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data/raw/transition_forecasting/global_stock_indices_historical_data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/transition_forecasting/global_transition_dataset"
FINAL_FILES = (
    "cleaned_ohlc.csv.gz",
    "daily_volatility.csv.gz",
    "transition_catalogue.csv",
    "sample_manifest.csv",
    "sequence_tensors.npz",
    "row_corrections.csv",
    "manifest.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(argv: list[str], log_path: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the single canonical transition-forecasting processed dataset."
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--expected-structural-flags", type=int, default=12)
    parser.add_argument("--expected-affected-indices", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def consolidate_cleaned_ohlc(cleaned_root: Path, output_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(cleaned_root.glob("*.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "index", path.stem)
        frames.append(frame)
    if not frames:
        raise RuntimeError(f"No cleaned OHLC files found under {cleaned_root}")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="raise")
    combined = combined.sort_values(["index", "date"]).reset_index(drop=True)
    combined.to_csv(output_path, index=False, compression="gzip", date_format="%Y-%m-%d")
    return combined


def build_daily_volatility(range_quality_path: Path, output_path: Path) -> pd.DataFrame:
    quality = pd.read_csv(range_quality_path)
    rows: list[pd.DataFrame] = []
    for _, item in quality.iterrows():
        path = Path(str(item["path"]))
        index_name = str(item["index"])
        effective_start = pd.Timestamp(item["recommended_effective_start"])
        frame = _load_generic_ohlc(path)
        series = log_parkinson(frame[frame.index >= effective_start])
        piece = pd.DataFrame(
            {
                "index": index_name,
                "date": series.index,
                "log_parkinson_volatility": series.to_numpy(dtype=float),
                "effective_start": effective_start,
            }
        )
        rows.append(piece)
    if not rows:
        raise RuntimeError("No eligible daily volatility series were produced")
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(["index", "date"]).reset_index(drop=True)
    combined.to_csv(output_path, index=False, compression="gzip", date_format="%Y-%m-%d")
    return combined


def validate_final_dataset(output_dir: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for name in FINAL_FILES:
        path = output_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        inventory[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    if missing:
        raise RuntimeError(f"Missing final dataset files: {missing}")

    sample_manifest = pd.read_csv(output_dir / "sample_manifest.csv")
    tensors = np.load(output_dir / "sequence_tensors.npz", allow_pickle=False)
    if len(sample_manifest) != len(tensors["sample_id"]):
        raise RuntimeError("Sample manifest and tensor sample counts differ")
    if sample_manifest["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate sample IDs found")
    if set(sample_manifest["sample_id"].astype(str)) != set(tensors["sample_id"].astype(str)):
        raise RuntimeError("Sample manifest and tensor sample IDs differ")
    if not np.isfinite(tensors["X"]).all():
        raise RuntimeError("Non-finite sequence tensor values found")
    if sample_manifest.groupby("episode_id")["split"].nunique().gt(1).any():
        raise RuntimeError("At least one global episode crosses split boundaries")
    return inventory


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root if args.raw_root.is_absolute() else REPO_ROOT / args.raw_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = REPO_ROOT / "results/runs/transition-process"
    elif not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if output_dir.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing dataset: {output_dir}")

    individual_raw = raw_root / "individual_indices_data"
    if not individual_raw.is_dir():
        raise SystemExit(f"Missing raw individual-index directory: {individual_raw}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="transition-process-", dir=output_dir.parent) as temp_name:
        temp_root = Path(temp_name)
        canonical_root = temp_root / "canonical"
        inventory_root = temp_root / "inventory"
        range_root = temp_root / "range"
        catalogue_root = temp_root / "catalogue"
        stage_d_root = temp_root / "stage_d"
        final_root = temp_root / "final"
        final_root.mkdir()

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
            log_dir / "01_build_canonical_clean_ohlc.log",
        )
        cleaned_root = canonical_root / "individual_indices_data"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/quality/audit_global_index_ohlc.py",
                "--data-dir",
                str(cleaned_root),
                "--out-dir",
                str(inventory_root),
                "--run-id",
                "build",
            ],
            log_dir / "02_audit_global_index_ohlc.log",
        )
        inventory_path = inventory_root / "build/global_index_ohlc_inventory.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/quality/audit_global_ohlc_range_quality.py",
                "--inventory",
                str(inventory_path),
                "--out-dir",
                str(range_root),
                "--run-id",
                "build",
            ],
            log_dir / "03_audit_global_ohlc_range_quality.log",
        )
        range_quality_path = range_root / "build/global_range_quality.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/catalogue/build_global_transition_catalogue.py",
                "--data-dir",
                str(cleaned_root),
                "--inventory",
                str(inventory_path),
                "--range-quality",
                str(range_quality_path),
                "--out-dir",
                str(catalogue_root),
                "--run-id",
                "build",
            ],
            log_dir / "04_build_global_transition_catalogue.log",
        )
        representative_path = catalogue_root / "build/representative_transition_catalogue.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/modeling/build_global_stage_d_dataset.py",
                "--representative-catalogue",
                str(representative_path),
                "--inventory",
                str(inventory_path),
                "--out-dir",
                str(stage_d_root),
                "--run-id",
                "build",
            ],
            log_dir / "05_build_global_stage_d_dataset.log",
        )

        cleaned = consolidate_cleaned_ohlc(
            cleaned_root, final_root / "cleaned_ohlc.csv.gz"
        )
        daily = build_daily_volatility(
            range_quality_path, final_root / "daily_volatility.csv.gz"
        )
        shutil.copy2(representative_path, final_root / "transition_catalogue.csv")
        shutil.copy2(stage_d_root / "build/sample_manifest.csv", final_root / "sample_manifest.csv")
        shutil.copy2(stage_d_root / "build/sequence_tensors.npz", final_root / "sequence_tensors.npz")
        shutil.copy2(canonical_root / "row_corrections.csv", final_root / "row_corrections.csv")

        correction_summary = json.loads((canonical_root / "manifest.json").read_text())
        stage_d_summary = json.loads(
            (stage_d_root / "build/stage_d_dataset_summary.json").read_text()
        )
        transition_summary = json.loads(
            (catalogue_root / "build/transition_count_summary.json").read_text()
        )

        manifest = {
            "schema_version": 1,
            "dataset": "global_transition_dataset",
            "built_at_utc": utc_now(),
            "raw_root": str(raw_root),
            "rules": {
                "interpolation": False,
                "forward_fill": False,
                "winsorization": False,
                "arbitrary_clipping": False,
                "synthetic_dates": False,
                "structural_bad_prints_removed_before_volatility": True,
                "controls_rematched_after_correction": True,
            },
            "counts": {
                "cleaned_ohlc_rows": int(len(cleaned)),
                "daily_volatility_rows": int(len(daily)),
                "structural_removed_rows": int(correction_summary["structural_removed_rows"]),
                "affected_indices": int(correction_summary["affected_indices"]),
                "transition_events": int(transition_summary["representative_market_events"]),
                "samples": int(stage_d_summary["total_samples"]),
                "positive_samples": int(stage_d_summary["positive_samples"]),
                "control_samples": int(stage_d_summary["negative_samples"]),
            },
            "stage_d_summary": stage_d_summary,
            "transition_summary": transition_summary,
        }
        (final_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        manifest["files"] = validate_final_dataset(final_root)
        (final_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        backup = None
        if output_dir.exists():
            backup = output_dir.with_name(output_dir.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(output_dir, backup)
        try:
            os.replace(final_root, output_dir)
        except Exception:
            if backup is not None and backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)

    print(f"WROTE {output_dir}")
    print("FILES:")
    for name in FINAL_FILES:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
