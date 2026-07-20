#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_FILES = (
    "cleaned_ohlc.csv.gz",
    "daily_volatility.csv.gz",
    "transition_catalogue.csv",
    "sample_manifest.csv",
    "sequence_tensors.npz",
    "row_corrections.csv",
    "manifest.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit(dataset_dir: Path) -> dict[str, object]:
    failures: list[str] = []
    files: dict[str, object] = {}
    for name in REQUIRED_FILES:
        path = dataset_dir / name
        require(path.is_file(), f"missing required file: {name}", failures)
        if path.is_file():
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

    if failures:
        return {"passed": False, "failures": failures, "files": files}

    cleaned = pd.read_csv(dataset_dir / "cleaned_ohlc.csv.gz", parse_dates=["date"])
    daily = pd.read_csv(dataset_dir / "daily_volatility.csv.gz", parse_dates=["date"])
    catalogue = pd.read_csv(dataset_dir / "transition_catalogue.csv")
    manifest = pd.read_csv(dataset_dir / "sample_manifest.csv")
    corrections = pd.read_csv(dataset_dir / "row_corrections.csv")
    metadata = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(dataset_dir / "sequence_tensors.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    require(not cleaned.empty, "cleaned OHLC is empty", failures)
    require(not daily.empty, "daily volatility is empty", failures)
    require(not catalogue.empty, "transition catalogue is empty", failures)
    require(not manifest.empty, "sample manifest is empty", failures)

    ohlc_columns = {"index", "date", "open", "high", "low", "close"}
    require(ohlc_columns.issubset(cleaned.columns), "cleaned OHLC schema incomplete", failures)
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(cleaned[column], errors="coerce")
        require(values.notna().all(), f"cleaned OHLC has nonnumeric {column}", failures)
        require((values > 0).all(), f"cleaned OHLC has nonpositive {column}", failures)
    require((cleaned["high"] >= cleaned["low"]).all(), "cleaned OHLC has high below low", failures)
    require(not cleaned.duplicated(["index", "date"]).any(), "duplicate index/date OHLC rows", failures)
    require(cleaned.sort_values(["index", "date"]).index.equals(cleaned.index), "cleaned OHLC is not sorted", failures)

    volatility_column = "log_parkinson_volatility"
    require(volatility_column in daily.columns, "daily volatility column missing", failures)
    if volatility_column in daily.columns:
        volatility = pd.to_numeric(daily[volatility_column], errors="coerce")
        require(np.isfinite(volatility).all(), "daily volatility contains nonfinite values", failures)
    require(not daily.duplicated(["index", "date"]).any(), "duplicate daily volatility rows", failures)

    required_manifest = {
        "sample_id", "label", "index", "episode_id", "split", "lead",
        "matched_positive_id", "origin_date",
    }
    require(required_manifest.issubset(manifest.columns), "sample manifest schema incomplete", failures)
    require(not manifest["sample_id"].astype(str).duplicated().any(), "duplicate sample IDs", failures)
    require(set(manifest["label"].dropna().astype(int)) == {0, 1}, "labels are not exactly {0,1}", failures)
    require(set(manifest["split"].astype(str)).issubset({"train", "val", "test"}), "invalid split value", failures)
    require(not manifest.groupby("episode_id")["split"].nunique().gt(1).any(), "episode split leakage", failures)

    positives = manifest[manifest["label"] == 1]
    controls = manifest[manifest["label"] == 0]
    control_counts = controls.groupby("matched_positive_id").size()
    require(len(positives) > 0, "no positive samples", failures)
    require(len(controls) == 5 * len(positives), "control/positive ratio is not 5:1", failures)
    require(set(control_counts.index.astype(str)) == set(positives["sample_id"].astype(str)), "control linkage does not cover all positives", failures)
    require(control_counts.eq(5).all(), "at least one positive does not have exactly five controls", failures)
    require(positives["matched_positive_id"].isna().all(), "positive sample has matched_positive_id", failures)
    require(controls["matched_positive_id"].notna().all(), "control missing matched_positive_id", failures)

    if "origin_pos" in manifest.columns:
        reused = controls.duplicated(["index", "lead", "split", "origin_pos"])
        require(not reused.any(), "control origin reused within index/lead/split", failures)

    require("X" in arrays and "sample_id" in arrays, "tensor archive missing X or sample_id", failures)
    if "X" in arrays and "sample_id" in arrays:
        tensor_ids = arrays["sample_id"].astype(str)
        require(len(tensor_ids) == len(manifest), "tensor and manifest row counts differ", failures)
        require(set(tensor_ids) == set(manifest["sample_id"].astype(str)), "tensor and manifest IDs differ", failures)
        require(arrays["X"].ndim == 3, "X tensor is not rank 3", failures)
        require(arrays["X"].shape[1:] == (40, 1), "X tensor trailing shape is not (40,1)", failures)
        require(np.isfinite(arrays["X"]).all(), "X tensor contains nonfinite values", failures)

    structural = corrections[corrections.get("reason", pd.Series(dtype=str)).astype(str) == "structural_bad_print"]
    expected_structural = int(metadata.get("counts", {}).get("structural_removed_rows", -1))
    expected_indices = int(metadata.get("counts", {}).get("affected_indices", -1))
    require(len(structural) == expected_structural, "structural correction count disagrees with manifest", failures)
    if len(structural):
        require(structural["index"].nunique() == expected_indices, "affected-index count disagrees with manifest", failures)
        require(structural.get("action", pd.Series(dtype=str)).astype(str).eq("drop").all(), "structural correction action is not drop", failures)

    rules = metadata.get("rules", {})
    for forbidden in ("interpolation", "forward_fill", "winsorization", "arbitrary_clipping", "synthetic_dates"):
        require(rules.get(forbidden) is False, f"forbidden transformation enabled: {forbidden}", failures)

    counts = {
        "cleaned_ohlc_rows": int(len(cleaned)),
        "daily_volatility_rows": int(len(daily)),
        "transition_events": int(len(catalogue)),
        "samples": int(len(manifest)),
        "positive_samples": int(len(positives)),
        "control_samples": int(len(controls)),
        "indices": int(cleaned["index"].nunique()),
        "episodes": int(manifest["episode_id"].nunique()),
    }
    return {
        "passed": not failures,
        "failures": failures,
        "counts": counts,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the canonical transition processed dataset")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/processed/transition_forecasting/global_transition_dataset"),
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    report = audit(args.dataset_dir)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(f"Processed transition dataset audit failed; inspect {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
