#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = REPO_ROOT / "data/raw/transition_forecasting/global_stock_indices_historical_data"
DEFAULT_REFERENCE_DIR = (
    REPO_ROOT
    / "results/transition_forecasting/modeling/stage_d_dataset/20260718T010839Z"
)


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
        raise RuntimeError(
            f"Baseline parity command failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )


def normalize_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date_columns = [
        "event_onset",
        "origin_date",
        "input_start_date",
        "target_end_date",
    ]
    for column in date_columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column], errors="raise").dt.strftime("%Y-%m-%d")
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].fillna("").astype(str)
    return result.sort_values("sample_id").reset_index(drop=True)


def compare_manifests(reference_path: Path, rebuilt_path: Path) -> dict[str, object]:
    reference = normalize_manifest(pd.read_csv(reference_path))
    rebuilt = normalize_manifest(pd.read_csv(rebuilt_path))

    reference_columns = list(reference.columns)
    rebuilt_columns = list(rebuilt.columns)
    common_columns = [column for column in reference_columns if column in rebuilt_columns]

    result: dict[str, object] = {
        "reference_rows": int(len(reference)),
        "rebuilt_rows": int(len(rebuilt)),
        "reference_columns": reference_columns,
        "rebuilt_columns": rebuilt_columns,
        "same_columns": reference_columns == rebuilt_columns,
        "same_sample_id_set": set(reference["sample_id"]) == set(rebuilt["sample_id"]),
        "mismatch_counts": {},
    }

    if not result["same_sample_id_set"]:
        result["only_reference_ids"] = sorted(
            set(reference["sample_id"]) - set(rebuilt["sample_id"])
        )[:50]
        result["only_rebuilt_ids"] = sorted(
            set(rebuilt["sample_id"]) - set(reference["sample_id"])
        )[:50]
        result["passed"] = False
        return result

    reference = reference.set_index("sample_id").loc[sorted(reference["sample_id"])]
    rebuilt = rebuilt.set_index("sample_id").loc[sorted(rebuilt["sample_id"])]

    mismatch_counts: dict[str, int] = {}
    for column in common_columns:
        if column == "sample_id":
            continue
        left = reference[column]
        right = rebuilt[column]
        left_numeric = pd.to_numeric(left, errors="coerce")
        right_numeric = pd.to_numeric(right, errors="coerce")
        numeric = left_numeric.notna() | right_numeric.notna()
        if numeric.all():
            equal = np.isclose(
                left_numeric.to_numpy(dtype=float),
                right_numeric.to_numpy(dtype=float),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            equal = left.astype(str).to_numpy() == right.astype(str).to_numpy()
        count = int((~equal).sum())
        if count:
            mismatch_counts[column] = count

    result["mismatch_counts"] = mismatch_counts
    result["passed"] = (
        len(reference) == len(rebuilt)
        and bool(result["same_columns"])
        and not mismatch_counts
    )
    return result


def load_tensor(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name] for name in archive.files}


def compare_tensors(reference_path: Path, rebuilt_path: Path) -> dict[str, object]:
    reference = load_tensor(reference_path)
    rebuilt = load_tensor(rebuilt_path)

    result: dict[str, object] = {
        "reference_keys": sorted(reference),
        "rebuilt_keys": sorted(rebuilt),
        "same_keys": set(reference) == set(rebuilt),
    }
    if "sample_id" not in reference or "sample_id" not in rebuilt or "X" not in reference or "X" not in rebuilt:
        result["passed"] = False
        result["reason"] = "Both archives must contain sample_id and X"
        return result

    reference_ids = reference["sample_id"].astype(str)
    rebuilt_ids = rebuilt["sample_id"].astype(str)
    result["same_sample_id_set"] = set(reference_ids) == set(rebuilt_ids)
    if not result["same_sample_id_set"]:
        result["passed"] = False
        return result

    reference_row = {sample_id: i for i, sample_id in enumerate(reference_ids)}
    rebuilt_row = {sample_id: i for i, sample_id in enumerate(rebuilt_ids)}
    ordered_ids = sorted(reference_row)
    reference_x = np.stack([reference["X"][reference_row[sample_id]] for sample_id in ordered_ids])
    rebuilt_x = np.stack([rebuilt["X"][rebuilt_row[sample_id]] for sample_id in ordered_ids])

    result["reference_shape"] = list(reference_x.shape)
    result["rebuilt_shape"] = list(rebuilt_x.shape)
    result["same_shape"] = reference_x.shape == rebuilt_x.shape
    if result["same_shape"]:
        difference = np.abs(reference_x.astype(float) - rebuilt_x.astype(float))
        result["max_abs_difference"] = float(difference.max()) if difference.size else 0.0
        result["exact_match"] = bool(np.array_equal(reference_x, rebuilt_x))
        result["numeric_match"] = bool(
            np.allclose(reference_x, rebuilt_x, rtol=1e-12, atol=1e-12, equal_nan=True)
        )
    else:
        result["max_abs_difference"] = None
        result["exact_match"] = False
        result["numeric_match"] = False

    result["passed"] = bool(
        result["same_keys"]
        and result["same_sample_id_set"]
        and result["same_shape"]
        and result["numeric_match"]
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the pre-correction Stage D dataset from the original raw OHLC and "
            "require parity with the existing Stage D reference before any correction is applied."
        )
    )
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root if args.raw_root.is_absolute() else REPO_ROOT / args.raw_root
    reference_dir = (
        args.reference_dir
        if args.reference_dir.is_absolute()
        else REPO_ROOT / args.reference_dir
    )
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    log_dir = args.log_dir if args.log_dir.is_absolute() else REPO_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    reference_manifest = reference_dir / "sample_manifest.csv"
    reference_tensors = reference_dir / "sequence_tensors.npz"
    for path in (reference_manifest, reference_tensors):
        if not path.is_file():
            raise SystemExit(f"Missing baseline reference artifact: {path}")

    raw_data_dir = raw_root / "individual_indices_data"
    if not raw_data_dir.is_dir():
        raise SystemExit(f"Missing original individual-index OHLC directory: {raw_data_dir}")

    with tempfile.TemporaryDirectory(prefix="stage-d-parity-", dir=report_path.parent) as temp_name:
        temp_root = Path(temp_name)
        inventory_root = temp_root / "inventory"
        range_root = temp_root / "range"
        catalogue_root = temp_root / "catalogue"
        stage_d_root = temp_root / "stage_d"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/quality/audit_global_index_ohlc.py",
                "--data-dir",
                str(raw_data_dir),
                "--out-dir",
                str(inventory_root),
                "--run-id",
                "parity",
            ],
            log_dir / "01_parity_inventory.log",
        )
        inventory_path = inventory_root / "parity/global_index_ohlc_inventory.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/quality/audit_global_ohlc_range_quality.py",
                "--inventory",
                str(inventory_path),
                "--out-dir",
                str(range_root),
                "--run-id",
                "parity",
            ],
            log_dir / "02_parity_range_quality.log",
        )
        range_quality_path = range_root / "parity/global_range_quality.csv"

        run(
            [
                sys.executable,
                "scripts/transition_forecasting/catalogue/build_global_transition_catalogue.py",
                "--data-dir",
                str(raw_data_dir),
                "--inventory",
                str(inventory_path),
                "--range-quality",
                str(range_quality_path),
                "--out-dir",
                str(catalogue_root),
                "--run-id",
                "parity",
            ],
            log_dir / "03_parity_catalogue.log",
        )
        representative_path = catalogue_root / "parity/representative_transition_catalogue.csv"

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
                "parity",
            ],
            log_dir / "04_parity_stage_d.log",
        )
        rebuilt_dir = stage_d_root / "parity"

        manifest_comparison = compare_manifests(
            reference_manifest,
            rebuilt_dir / "sample_manifest.csv",
        )
        tensor_comparison = compare_tensors(
            reference_tensors,
            rebuilt_dir / "sequence_tensors.npz",
        )
        report = {
            "schema_version": 1,
            "reference_dir": str(reference_dir),
            "raw_root": str(raw_root),
            "manifest": manifest_comparison,
            "tensors": tensor_comparison,
            "passed": bool(manifest_comparison["passed"] and tensor_comparison["passed"]),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(
            "Stage D baseline parity failed. Post-parity corrections are blocked; "
            f"inspect {report_path}."
        )
    print(f"Stage D baseline parity passed: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
