from __future__ import annotations

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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def audit_processed_dataset(
    dataset_dir: Path,
    *,
    controls_per_positive: int = 3,
) -> dict[str, object]:
    """Audit the canonical transition dataset without mutating it."""
    dataset_dir = Path(dataset_dir)
    failures: list[str] = []
    files: dict[str, object] = {}

    for name in REQUIRED_FILES:
        path = dataset_dir / name
        _require(path.is_file(), f"missing required file: {name}", failures)
        if path.is_file():
            files[name] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    if failures:
        return {"passed": False, "failures": failures, "files": files}

    cleaned = pd.read_csv(dataset_dir / "cleaned_ohlc.csv.gz", parse_dates=["date"])
    daily = pd.read_csv(dataset_dir / "daily_volatility.csv.gz", parse_dates=["date"])
    catalogue = pd.read_csv(dataset_dir / "transition_catalogue.csv")
    sample_manifest = pd.read_csv(dataset_dir / "sample_manifest.csv")
    corrections = pd.read_csv(dataset_dir / "row_corrections.csv")
    metadata = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    with np.load(dataset_dir / "sequence_tensors.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}

    _require(not cleaned.empty, "cleaned OHLC is empty", failures)
    _require(not daily.empty, "daily volatility is empty", failures)
    _require(not catalogue.empty, "transition catalogue is empty", failures)
    _require(not sample_manifest.empty, "sample manifest is empty", failures)

    ohlc_columns = {"index", "date", "open", "high", "low", "close"}
    _require(ohlc_columns.issubset(cleaned.columns), "cleaned OHLC schema incomplete", failures)
    if ohlc_columns.issubset(cleaned.columns):
        for column in ("open", "high", "low", "close"):
            values = pd.to_numeric(cleaned[column], errors="coerce")
            _require(values.notna().all(), f"cleaned OHLC has nonnumeric {column}", failures)
            _require((values > 0).all(), f"cleaned OHLC has nonpositive {column}", failures)
        _require((cleaned["high"] >= cleaned["low"]).all(), "cleaned OHLC has high below low", failures)
        _require(
            not cleaned.duplicated(["index", "date"]).any(),
            "duplicate index/date OHLC rows",
            failures,
        )
        expected_order = cleaned.sort_values(["index", "date"]).reset_index(drop=True)
        _require(cleaned.reset_index(drop=True).equals(expected_order), "cleaned OHLC is not sorted", failures)

    volatility_column = "log_parkinson_volatility"
    _require(volatility_column in daily.columns, "daily volatility column missing", failures)
    if volatility_column in daily.columns:
        volatility = pd.to_numeric(daily[volatility_column], errors="coerce")
        _require(np.isfinite(volatility).all(), "daily volatility contains nonfinite values", failures)
    if {"index", "date"}.issubset(daily.columns):
        _require(
            not daily.duplicated(["index", "date"]).any(),
            "duplicate daily volatility rows",
            failures,
        )
        expected_daily = daily.sort_values(["index", "date"]).reset_index(drop=True)
        _require(daily.reset_index(drop=True).equals(expected_daily), "daily volatility is not sorted", failures)

    required_manifest = {
        "sample_id",
        "label",
        "index",
        "episode_id",
        "split",
        "lead",
        "matched_positive_id",
        "origin_date",
    }
    _require(
        required_manifest.issubset(sample_manifest.columns),
        "sample manifest schema incomplete",
        failures,
    )

    positives = pd.DataFrame()
    controls = pd.DataFrame()
    if required_manifest.issubset(sample_manifest.columns):
        _require(
            not sample_manifest["sample_id"].astype(str).duplicated().any(),
            "duplicate sample IDs",
            failures,
        )
        _require(
            set(sample_manifest["label"].dropna().astype(int)) == {0, 1},
            "labels are not exactly {0,1}",
            failures,
        )
        _require(
            set(sample_manifest["split"].astype(str)).issubset({"train", "val", "test"}),
            "invalid split value",
            failures,
        )
        _require(
            not sample_manifest.groupby("episode_id")["split"].nunique().gt(1).any(),
            "episode split leakage",
            failures,
        )

        positives = sample_manifest[sample_manifest["label"] == 1]
        controls = sample_manifest[sample_manifest["label"] == 0]
        control_counts = controls.groupby("matched_positive_id").size()
        _require(len(positives) > 0, "no positive samples", failures)
        _require(
            len(controls) == controls_per_positive * len(positives),
            f"control/positive ratio is not {controls_per_positive}:1",
            failures,
        )
        _require(
            set(control_counts.index.astype(str)) == set(positives["sample_id"].astype(str)),
            "control linkage does not cover all positives",
            failures,
        )
        _require(
            control_counts.eq(controls_per_positive).all(),
            f"at least one positive does not have exactly {controls_per_positive} controls",
            failures,
        )
        _require(
            positives["matched_positive_id"].isna().all(),
            "positive sample has matched_positive_id",
            failures,
        )
        _require(
            controls["matched_positive_id"].notna().all(),
            "control missing matched_positive_id",
            failures,
        )

        reuse_columns = ["index", "lead", "split"]
        if "origin_pos" in controls.columns:
            reuse_columns.append("origin_pos")
        else:
            reuse_columns.append("origin_date")
        _require(
            not controls.duplicated(reuse_columns).any(),
            "control origin reused within index/lead/split",
            failures,
        )

    _require("X" in arrays and "sample_id" in arrays, "tensor archive missing X or sample_id", failures)
    if "X" in arrays and "sample_id" in arrays:
        tensor_ids = arrays["sample_id"].astype(str)
        manifest_ids = sample_manifest["sample_id"].astype(str).to_numpy()
        _require(len(tensor_ids) == len(sample_manifest), "tensor and manifest row counts differ", failures)
        _require(
            np.array_equal(tensor_ids, manifest_ids),
            "tensor and manifest IDs are not exactly aligned in order",
            failures,
        )
        _require(arrays["X"].ndim == 3, "X tensor is not rank 3", failures)
        if arrays["X"].ndim == 3:
            _require(
                arrays["X"].shape[1:] == (40, 1),
                "X tensor trailing shape is not (40,1)",
                failures,
            )
        _require(np.isfinite(arrays["X"]).all(), "X tensor contains nonfinite values", failures)

    if "reason" in corrections.columns:
        structural = corrections[corrections["reason"].astype(str) == "structural_bad_print"]
    else:
        structural = pd.DataFrame()
    counts_section = metadata.get("counts", {})
    expected_structural = int(counts_section.get("structural_removed_rows", -1))
    expected_indices = int(counts_section.get("affected_indices", -1))
    _require(
        len(structural) == expected_structural,
        "structural correction count disagrees with manifest",
        failures,
    )
    if len(structural):
        _require(
            structural["index"].nunique() == expected_indices,
            "affected-index count disagrees with manifest",
            failures,
        )
        _require(
            structural["action"].astype(str).eq("drop").all(),
            "structural correction action is not drop",
            failures,
        )

    rules = metadata.get("rules", {})
    for forbidden in (
        "interpolation",
        "forward_fill",
        "winsorization",
        "arbitrary_clipping",
        "synthetic_dates",
    ):
        _require(
            rules.get(forbidden) is False,
            f"forbidden transformation enabled or undocumented: {forbidden}",
            failures,
        )

    _require(metadata.get("test_evaluated") is False, "test_evaluated must be false", failures)

    counts = {
        "cleaned_ohlc_rows": int(len(cleaned)),
        "daily_volatility_rows": int(len(daily)),
        "transition_events": int(len(catalogue)),
        "samples": int(len(sample_manifest)),
        "positive_samples": int(len(positives)),
        "control_samples": int(len(controls)),
        "indices": int(cleaned["index"].nunique()) if "index" in cleaned else 0,
        "episodes": int(sample_manifest["episode_id"].nunique()) if "episode_id" in sample_manifest else 0,
        "controls_per_positive": controls_per_positive,
    }
    return {
        "passed": not failures,
        "failures": failures,
        "counts": counts,
        "files": files,
    }
