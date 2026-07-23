#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.data.validation import audit_processed_dataset

FOLD_FILES = (
    "rematched_rolling_manifest.csv",
    "rematched_rolling_tensors.npz",
    "control_match_audit.csv",
    "summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the canonical one-channel transition dataset, binary "
            "control candidate pool, and pre-control folds."
        )
    )
    parser.add_argument("--dataset-1d", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    return parser.parse_args()


def _validate_candidate_artifacts(
    dataset: Path,
    *,
    failures: list[str],
) -> dict[str, object]:
    manifest_path = dataset / "control_candidate_manifest.csv"
    tensor_path = dataset / "control_candidate_tensors.npz"
    summary_path = dataset / "candidate_pool_summary.json"
    missing = [
        str(path)
        for path in (manifest_path, tensor_path, summary_path)
        if not path.is_file()
    ]
    if missing:
        failures.extend(f"candidate artifact missing: {path}" for path in missing)
        return {"missing": missing}

    frame = pd.read_csv(manifest_path)
    required = {
        "candidate_id",
        "index",
        "lead",
        "origin_pos",
        "origin_date",
        "input_start_date",
        "target_end_date",
    }
    missing_columns = required.difference(frame.columns)
    if missing_columns:
        failures.append(
            f"candidate manifest missing columns: {sorted(missing_columns)}"
        )
    forbidden = {
        "control_stratum",
        "evaluation_stratum",
        "future_assessment_end_date",
        "future_persistent",
    }.intersection(frame.columns)
    if forbidden:
        failures.append(
            f"candidate manifest still contains redesigned-control columns: {sorted(forbidden)}"
        )
    if frame["candidate_id"].astype(str).duplicated().any():
        failures.append("candidate IDs are not unique")

    with np.load(tensor_path, allow_pickle=False) as archive:
        tensor = np.asarray(archive["X"])
        ids = archive["candidate_id"].astype(str)
    if tensor.shape != (len(frame), 40, 1):
        failures.append(
            f"candidate tensor shape is {tensor.shape}, expected ({len(frame)}, 40, 1)"
        )
    if not np.array_equal(frame["candidate_id"].astype(str).to_numpy(), ids):
        failures.append("candidate manifest and tensor IDs are misaligned")

    return {
        "rows": int(len(frame)),
        "unique_origins": int(
            frame[["index", "origin_date"]].drop_duplicates().shape[0]
        ),
        "leads": sorted(int(value) for value in frame["lead"].unique()),
        "tensor_shape": list(tensor.shape),
        "forbidden_columns_present": sorted(forbidden),
    }


def _validate_episode_separation(
    manifest: pd.DataFrame,
    *,
    failures: list[str],
) -> None:
    active = manifest.loc[manifest["fold_split"].isin(["train", "val", "test"])]
    split_counts = active.groupby(["fold", "episode_id"])["fold_split"].nunique()
    leaking = split_counts[split_counts.gt(1)]
    if len(leaking):
        failures.append(
            f"episodes span rolling-fold partitions: {len(leaking)} fold/episode pairs"
        )


def _validate_fold_artifacts(
    dataset: Path,
    *,
    controls_per_positive: int,
    failures: list[str],
) -> dict[str, object]:
    fold_dir = dataset / "purged_walk_forward_folds"
    missing = [name for name in FOLD_FILES if not (fold_dir / name).is_file()]
    if missing:
        failures.extend(f"fold artifact missing: {name}" for name in missing)
        return {"directory": str(fold_dir), "missing": missing}

    manifest = pd.read_csv(fold_dir / "rematched_rolling_manifest.csv")
    required = {
        "sample_id",
        "label",
        "episode_id",
        "fold",
        "fold_split",
        "index",
        "lead",
        "origin_date",
        "target_end_date",
    }
    missing_columns = required.difference(manifest.columns)
    if missing_columns:
        failures.append(f"fold manifest missing columns: {sorted(missing_columns)}")

    forbidden = {
        "control_stratum",
        "evaluation_stratum",
        "label_end_date",
        "future_assessment_end_date",
    }.intersection(manifest.columns)
    if forbidden:
        failures.append(
            f"fold manifest still contains redesigned-control columns: {sorted(forbidden)}"
        )

    if manifest.duplicated(["fold", "sample_id"]).any():
        failures.append("sample IDs are not unique within each rolling fold")
    if not manifest["fold_split"].isin(["train", "val", "test"]).all():
        failures.append("fold manifest contains invalid split values")
    if not manifest["fold_split"].eq("test").any():
        failures.append("fixed test assignment is missing")

    _validate_episode_separation(manifest, failures=failures)

    positives = manifest.loc[manifest["label"].eq(1)].copy()
    controls = manifest.loc[manifest["label"].eq(0)].copy()
    if controls.empty or "matched_positive_id" not in controls.columns:
        failures.append("fold controls lack matched-positive linkage")
    else:
        positive_keys = set(
            zip(
                positives["fold"].astype(int),
                positives["fold_split"].astype(str),
                positives["sample_id"].astype(str),
                strict=True,
            )
        )
        control_keys = set(
            zip(
                controls["fold"].astype(int),
                controls["fold_split"].astype(str),
                controls["matched_positive_id"].astype(str),
                strict=True,
            )
        )
        if positive_keys != control_keys:
            failures.append("kept positives and control linkage differ")
        counts = controls.groupby(
            ["fold", "fold_split", "matched_positive_id"], sort=True
        ).size()
        if not counts.eq(int(controls_per_positive)).all():
            failures.append(
                f"controls do not have exactly {controls_per_positive} rows per positive"
            )

    reuse_key = ["fold", "fold_split", "index", "origin_date"]
    if not controls.empty and controls.duplicated(reuse_key).any():
        failures.append("control origins are reused within a fold partition")

    with np.load(
        fold_dir / "rematched_rolling_tensors.npz",
        allow_pickle=False,
    ) as archive:
        tensor = np.asarray(archive["X"])
        ids = archive["sample_id"].astype(str)
        splits = archive["fold_split"].astype(str)
        channel_names = archive["channel_names"].astype(str)
    if len(manifest) != len(tensor):
        failures.append("fold manifest and tensor lengths differ")
    if not np.array_equal(manifest["sample_id"].astype(str).to_numpy(), ids):
        failures.append("fold sample IDs are misaligned")
    if not np.array_equal(
        manifest["fold_split"].astype(str).to_numpy(),
        splits,
    ):
        failures.append("fold split metadata are misaligned")
    if tensor.ndim != 3 or tensor.shape[1:] != (40, 1):
        failures.append(f"fold tensor trailing shape is not (40,1): {tensor.shape}")
    if channel_names.tolist() != ["log_volatility_level"]:
        failures.append(f"unexpected fold channels: {channel_names.tolist()}")

    summary = json.loads((fold_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("test_evaluated") is not False:
        failures.append("fold summary test_evaluated must be false")
    if summary.get("three_channel_dataset_built") is not False:
        failures.append("fold summary must record that 3D construction was skipped")

    return {
        "directory": str(fold_dir),
        "rows": int(len(manifest)),
        "positive_rows": int(len(positives)),
        "control_rows": int(len(controls)),
        "tensor_shape": list(tensor.shape),
        "channels": channel_names.tolist(),
        "forbidden_columns_present": sorted(forbidden),
    }


def validate(args: argparse.Namespace) -> dict[str, object]:
    failures: list[str] = []
    audit_1d = audit_processed_dataset(
        args.dataset_1d,
        controls_per_positive=args.controls_per_positive,
    )
    if not audit_1d.get("passed", False):
        failures.extend(f"1D: {item}" for item in audit_1d.get("failures", []))

    candidate_checks = _validate_candidate_artifacts(
        args.dataset_1d,
        failures=failures,
    )
    fold_checks = _validate_fold_artifacts(
        args.dataset_1d,
        controls_per_positive=args.controls_per_positive,
        failures=failures,
    )

    return {
        "passed": not failures,
        "failures": failures,
        "dataset_1d": audit_1d,
        "candidate_checks": candidate_checks,
        "fold_checks": fold_checks,
        "control_protocol": "precontrol_binary_matching",
        "three_channel_dataset_built": False,
        "test_evaluated": False,
    }


def main() -> int:
    args = parse_args()
    report = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
