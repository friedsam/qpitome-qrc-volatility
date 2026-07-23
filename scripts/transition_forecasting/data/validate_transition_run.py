#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.data.three_channel import CHANNEL_NAMES, to_three_channel
from transition_forecasting.data.validation import audit_processed_dataset
from transition_forecasting.modeling.control_strata import (
    CALM,
    HARD_NEGATIVE,
    TRANSITION,
    ControlStrataPolicy,
)

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


def _validate_candidate_manifest(
    path: Path,
    *,
    label: str,
    policy: ControlStrataPolicy,
    failures: list[str],
) -> dict[str, object]:
    frame = pd.read_csv(path)
    required = {
        "candidate_id",
        "index",
        "lead",
        "origin_date",
        "target_end_date",
        "future_assessment_end_date",
        "control_stratum",
        "future_threshold_crossings",
        "future_persistent",
    }
    missing = required.difference(frame.columns)
    if missing:
        failures.append(f"{label} candidate manifest missing columns: {sorted(missing)}")
        return {"rows": int(len(frame)), "missing_columns": sorted(missing)}

    observed = set(frame["control_stratum"].dropna().astype(str).unique())
    expected = {CALM, HARD_NEGATIVE}
    if observed != expected:
        failures.append(
            f"{label} candidate strata {sorted(observed)} do not equal {sorted(expected)}"
        )
    if frame["candidate_id"].astype(str).duplicated().any():
        failures.append(f"{label} candidate IDs are not unique")
    if frame["future_persistent"].astype(bool).any():
        failures.append(f"{label} persistent candidates leaked into controls")
    if frame.loc[
        frame["control_stratum"].eq(CALM),
        "future_threshold_crossings",
    ].ne(0).any():
        failures.append(f"{label} calm candidates contain forecast-horizon crossings")
    hard_crossings = frame.loc[
        frame["control_stratum"].eq(HARD_NEGATIVE),
        "future_threshold_crossings",
    ]
    if hard_crossings.le(0).any():
        failures.append(f"{label} hard negatives lack a forecast-horizon crossing")
    if hard_crossings.gt(policy.forecast_horizon).any():
        failures.append(f"{label} control crossing counts exceed the forecast horizon")

    target_end = pd.to_datetime(frame["target_end_date"], errors="coerce", utc=True)
    label_end = pd.to_datetime(
        frame["future_assessment_end_date"], errors="coerce", utc=True
    )
    if target_end.isna().any() or label_end.isna().any():
        failures.append(f"{label} candidate maturity dates are invalid")
    elif not target_end.le(label_end).all():
        failures.append(f"{label} candidate labels mature before targets end")

    return {
        "rows": int(len(frame)),
        "stratum_counts": (
            frame["control_stratum"].value_counts().sort_index().astype(int).to_dict()
        ),
        "unique_origins": int(
            frame[["index", "origin_date"]].drop_duplicates().shape[0]
        ),
    }


def _validate_episode_separation(
    manifest: pd.DataFrame,
    *,
    label: str,
    failures: list[str],
) -> None:
    required = {"fold", "episode_id", "fold_split"}
    missing = required.difference(manifest.columns)
    if missing:
        failures.append(
            f"{label} fold manifest cannot audit episodes; missing {sorted(missing)}"
        )
        return
    active = manifest.loc[manifest["fold_split"].isin(["train", "val", "test"])]
    split_counts = active.groupby(["fold", "episode_id"])["fold_split"].nunique()
    leaking = split_counts[split_counts.gt(1)]
    if len(leaking):
        failures.append(
            f"{label} episodes span rolling-fold partitions: {len(leaking)} fold/episode pairs"
        )


def _validate_fold_manifest(
    manifest: pd.DataFrame,
    *,
    label: str,
    policy: ControlStrataPolicy,
    failures: list[str],
) -> dict[str, object]:
    required = {
        "sample_id",
        "label",
        "fold",
        "fold_split",
        "index",
        "lead",
        "origin_date",
        "target_end_date",
        "label_end_date",
        "evaluation_stratum",
    }
    missing = required.difference(manifest.columns)
    if missing:
        failures.append(f"{label} fold manifest missing columns: {sorted(missing)}")
        return {"rows": int(len(manifest)), "missing_columns": sorted(missing)}

    observed = set(manifest["evaluation_stratum"].dropna().astype(str).unique())
    expected = {TRANSITION, CALM, HARD_NEGATIVE}
    if observed != expected:
        failures.append(
            f"{label} fold strata {sorted(observed)} do not equal {sorted(expected)}"
        )
    if manifest.duplicated(["fold", "sample_id"]).any():
        failures.append(f"{label} sample IDs are not unique within each rolling fold")

    target_end = pd.to_datetime(manifest["target_end_date"], errors="coerce", utc=True)
    label_end = pd.to_datetime(manifest["label_end_date"], errors="coerce", utc=True)
    if target_end.isna().any() or label_end.isna().any():
        failures.append(f"{label} fold maturity dates are invalid")
    elif not target_end.le(label_end).all():
        failures.append(f"{label} fold labels mature before targets end")

    _validate_episode_separation(manifest, label=label, failures=failures)

    positives = manifest.loc[manifest["label"].eq(1)].copy()
    controls = manifest.loc[manifest["label"].eq(0)].copy()
    if not positives["evaluation_stratum"].eq(TRANSITION).all():
        failures.append(f"{label} positive rows are not labeled transition")
    if controls.empty or "matched_positive_id" not in controls.columns:
        failures.append(f"{label} fold controls lack matched-positive linkage")
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
            failures.append(f"{label} kept positives and control linkage differ")

        counts = (
            controls.groupby(
                ["fold", "fold_split", "matched_positive_id", "evaluation_stratum"],
                sort=True,
            )
            .size()
            .unstack(fill_value=0)
        )
        for stratum, requested in policy.requested_counts.items():
            if stratum not in counts.columns or not counts[stratum].eq(requested).all():
                failures.append(
                    f"{label} controls do not have exactly {requested} {stratum} rows per positive"
                )

    reuse_key = ["fold", "fold_split", "index", "origin_date"]
    if not controls.empty and controls.duplicated(reuse_key).any():
        failures.append(f"{label} control origins are reused within a fold partition")

    return {
        "rows": int(len(manifest)),
        "positive_rows": int(len(positives)),
        "control_rows": int(len(controls)),
        "stratum_counts": (
            manifest["evaluation_stratum"]
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
        ),
    }


def validate(args: argparse.Namespace) -> dict[str, object]:
    failures: list[str] = []
    policy = ControlStrataPolicy.from_total_controls(args.controls_per_positive)
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
        if not missing:
            candidate_checks[label].update(
                _validate_candidate_manifest(
                    candidate_manifest,
                    label=label,
                    policy=policy,
                    failures=failures,
                )
            )

    fold_checks: dict[str, object] = {}
    fold_payloads: dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {}
    for label, dataset in (("1d", args.dataset_1d), ("3d", args.dataset_3d)):
        fold_dir = dataset / "purged_walk_forward_folds"
        missing = [name for name in FOLD_FILES if not (fold_dir / name).is_file()]
        fold_checks[label] = {"directory": str(fold_dir), "missing": missing}
        failures.extend(f"{label} fold artifact missing: {name}" for name in missing)
        if not missing:
            manifest = pd.read_csv(fold_dir / "rematched_rolling_manifest.csv")
            with np.load(
                fold_dir / "rematched_rolling_tensors.npz",
                allow_pickle=False,
            ) as archive:
                tensor = np.asarray(archive["X"])
                ids = archive["sample_id"].astype(str)
                splits = archive["fold_split"].astype(str)
            if len(manifest) != len(tensor):
                failures.append(f"{label} fold manifest and tensor lengths differ")
            if not np.array_equal(manifest["sample_id"].astype(str).to_numpy(), ids):
                failures.append(f"{label} fold sample IDs are misaligned")
            if not np.array_equal(
                manifest["fold_split"].astype(str).to_numpy(),
                splits,
            ):
                failures.append(f"{label} fold split metadata are misaligned")
            if bool((manifest["fold_split"] == "test").sum() == 0):
                failures.append(f"{label} fixed test assignment is missing")
            fold_checks[label].update(
                _validate_fold_manifest(
                    manifest,
                    label=label,
                    policy=policy,
                    failures=failures,
                )
            )
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
        if not np.array_equal(
            tensor_fold_3d,
            to_three_channel(tensor_fold_1d),
        ):
            failures.append("3D fold tensor is not the deterministic 1D transform")

    return {
        "passed": not failures,
        "failures": failures,
        "control_policy": policy.to_dict(),
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
