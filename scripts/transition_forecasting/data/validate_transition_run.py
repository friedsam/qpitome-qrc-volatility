#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.data.three_channel import CHANNEL_NAMES, to_three_channel
from transition_forecasting.data.validation import audit_processed_dataset

FOLD_FILES = (
    "rematched_rolling_manifest.csv",
    "rematched_rolling_tensors.npz",
    "control_match_audit.csv",
    "summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate aligned transition datasets, candidate pools, and folds."
    )
    parser.add_argument("--dataset-1d", type=Path, required=True)
    parser.add_argument("--dataset-3d", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    return parser.parse_args()


def _load_tensor(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive["X"]), archive["sample_id"].astype(str)


def validate(args: argparse.Namespace) -> dict[str, object]:
    failures: list[str] = []
    audit_1d = audit_processed_dataset(
        args.dataset_1d,
        controls_per_positive=args.controls_per_positive,
    )
    if not audit_1d.get("passed", False):
        failures.extend(f"1D: {item}" for item in audit_1d.get("failures", []))

    manifest_1d = pd.read_csv(args.dataset_1d / "sample_manifest.csv")
    manifest_3d = pd.read_csv(args.dataset_3d / "sample_manifest.csv")
    x1, ids1 = _load_tensor(args.dataset_1d / "sequence_tensors.npz")
    x3, ids3 = _load_tensor(args.dataset_3d / "sequence_tensors.npz")

    if not manifest_1d.equals(manifest_3d):
        failures.append("1D and 3D sample manifests differ")
    if not np.array_equal(ids1, ids3):
        failures.append("1D and 3D tensor sample IDs differ")
    if not np.array_equal(x3, to_three_channel(x1)):
        failures.append("3D tensor is not the deterministic transform of the 1D tensor")
    if x3.shape[-1] != len(CHANNEL_NAMES):
        failures.append("3D tensor channel count is incorrect")

    candidate_checks: dict[str, object] = {}
    for label, dataset in (("1d", args.dataset_1d), ("3d", args.dataset_3d)):
        candidate_manifest = dataset / "control_candidate_manifest.csv"
        candidate_tensor = dataset / "control_candidate_tensors.npz"
        candidate_summary = dataset / "candidate_pool_summary.json"
        missing = [
            str(path)
            for path in (candidate_manifest, candidate_tensor, candidate_summary)
            if not path.is_file()
        ]
        candidate_checks[label] = {"missing": missing}
        failures.extend(f"{label} candidate artifact missing: {path}" for path in missing)

    fold_checks: dict[str, object] = {}
    fold_payloads: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for label, dataset in (("1d", args.dataset_1d), ("3d", args.dataset_3d)):
        fold_dir = dataset / "purged_walk_forward_folds"
        missing = [name for name in FOLD_FILES if not (fold_dir / name).is_file()]
        fold_checks[label] = {"directory": str(fold_dir), "missing": missing}
        failures.extend(f"{label} fold artifact missing: {name}" for name in missing)
        if not missing:
            manifest = pd.read_csv(fold_dir / "rematched_rolling_manifest.csv")
            with np.load(fold_dir / "rematched_rolling_tensors.npz", allow_pickle=False) as archive:
                tensor = np.asarray(archive["X"])
                ids = archive["sample_id"].astype(str)
                splits = archive["fold_split"].astype(str)
            if len(manifest) != len(tensor):
                failures.append(f"{label} fold manifest and tensor lengths differ")
            if not np.array_equal(manifest["sample_id"].astype(str).to_numpy(), ids):
                failures.append(f"{label} fold sample IDs are misaligned")
            if not np.array_equal(manifest["fold_split"].astype(str).to_numpy(), splits):
                failures.append(f"{label} fold split metadata are misaligned")
            if bool((manifest["fold_split"] == "test").sum() == 0):
                failures.append(f"{label} fixed test assignment is missing")
            fold_payloads[label] = (manifest, tensor, ids)

    if set(fold_payloads) == {"1d", "3d"}:
        manifest_fold_1d, tensor_fold_1d, ids_fold_1d = fold_payloads["1d"]
        manifest_fold_3d, tensor_fold_3d, ids_fold_3d = fold_payloads["3d"]
        identity_columns = ["sample_id", "fold", "fold_split"]
        if not manifest_fold_1d[identity_columns].equals(
            manifest_fold_3d[identity_columns]
        ):
            failures.append("1D and 3D fold assignments differ")
        if not np.array_equal(ids_fold_1d, ids_fold_3d):
            failures.append("1D and 3D fold tensor IDs differ")
        if not np.array_equal(tensor_fold_3d, to_three_channel(tensor_fold_1d)):
            failures.append("3D fold tensor is not the deterministic 1D transform")

    return {
        "passed": not failures,
        "failures": failures,
        "dataset_1d": audit_1d,
        "candidate_checks": candidate_checks,
        "fold_checks": fold_checks,
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
