from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.chronological_splits import (
    IntervalColumns,
    rolling_origin_assignments,
)
from transition_forecasting.modeling.control_strata import (
    TRANSITION,
    ControlStrataPolicy,
    rematch_control_strata_within_partition,
)


def load_candidate_pool(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    manifest = pd.read_csv(
        run_dir / "control_candidate_manifest.csv",
        parse_dates=["origin_date", "input_start_date", "target_end_date"],
    )
    with np.load(run_dir / "control_candidate_tensors.npz", allow_pickle=True) as data:
        tensor = np.asarray(data["X"], dtype=float)
        ids = data["candidate_id"].astype(str)
    manifest_ids = manifest["candidate_id"].astype(str).to_numpy()
    if not np.array_equal(ids, manifest_ids):
        raise ValueError("candidate manifest and tensor IDs are not aligned")
    if len(manifest) != len(tensor):
        raise ValueError("candidate manifest and tensor lengths differ")
    manifest["_candidate_row"] = np.arange(len(manifest), dtype=int)
    return manifest, tensor


def _validate_positive_intervals(positives: pd.DataFrame) -> pd.DataFrame:
    required = {"origin_date", "input_start_date", "target_end_date"}
    missing = required.difference(positives.columns)
    if missing:
        raise ValueError(
            "Stage D must be rebuilt with exact interval columns before chronological rematching: "
            f"{sorted(missing)}"
        )
    frame = positives.copy()
    for column in required:
        frame[column] = pd.to_datetime(frame[column], errors="raise", utc=True)
    invalid = ~(
        (frame["input_start_date"] <= frame["origin_date"])
        & (frame["origin_date"] < frame["target_end_date"])
    )
    if invalid.any():
        raise ValueError(f"invalid positive trading-row intervals: {int(invalid.sum())}")
    return frame


def _candidate_partition(
    candidates: pd.DataFrame,
    fold_frame: pd.DataFrame,
) -> pd.DataFrame:
    active = fold_frame[fold_frame["fold_split"].isin(["train", "val", "test"])]
    train = active[active["fold_split"].eq("train")]
    val = active[active["fold_split"].eq("val")]
    test = active[active["fold_split"].eq("test")]
    if train.empty or val.empty or test.empty:
        raise ValueError("fold requires nonempty train, validation, and test positives")

    train_end = pd.to_datetime(train["interval_end"], utc=True).max()
    val_start = pd.to_datetime(val["interval_start"], utc=True).min()
    val_end = pd.to_datetime(val["interval_end"], utc=True).max()
    test_start = pd.to_datetime(test["interval_start"], utc=True).min()

    frame = candidates.copy()
    frame["input_start_date"] = pd.to_datetime(frame["input_start_date"], utc=True)
    frame["target_end_date"] = pd.to_datetime(frame["target_end_date"], utc=True)
    frame["fold_split"] = "unused"
    frame.loc[frame["target_end_date"] <= train_end, "fold_split"] = "train"
    frame.loc[
        (frame["input_start_date"] >= val_start)
        & (frame["target_end_date"] <= val_end),
        "fold_split",
    ] = "val"
    frame.loc[frame["input_start_date"] >= test_start, "fold_split"] = "test"
    return frame


def _complete_positive_ids(audit: pd.DataFrame) -> set[str]:
    if audit.empty:
        return set()
    return set(
        audit.loc[audit["complete_all_strata"], "positive_sample_id"].astype(str)
    )


def build_rematched_rolling_dataset(
    stage_d_manifest: pd.DataFrame,
    stage_d_tensor: np.ndarray,
    candidate_manifest: pd.DataFrame,
    candidate_tensor: np.ndarray,
    *,
    n_folds: int,
    test_fraction: float = 0.17,
    embargo_days: int = 10,
    controls_per_positive: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    control_policy = ControlStrataPolicy.from_total_controls(controls_per_positive)
    positives = stage_d_manifest[stage_d_manifest["label"].eq(1)].reset_index(
        drop=True
    ).copy()
    positive_source_rows = stage_d_manifest.index[
        stage_d_manifest["label"].eq(1)
    ].to_numpy(dtype=int)
    positives["_positive_tensor_row"] = positive_source_rows
    positives = _validate_positive_intervals(positives)

    assignments, fold_summaries, split_audit = rolling_origin_assignments(
        positives,
        n_folds=n_folds,
        test_fraction=test_fraction,
        embargo_days=embargo_days,
        columns=IntervalColumns(
            input_start="input_start_date",
            target_end="target_end_date",
        ),
    )

    output_frames: list[pd.DataFrame] = []
    output_tensors: list[np.ndarray] = []
    audit_frames: list[pd.DataFrame] = []

    for fold in range(1, n_folds + 1):
        fold_positives = assignments[
            assignments["fold"].eq(fold)
            & assignments["fold_split"].isin(["train", "val", "test"])
        ].copy()
        fold_candidates = _candidate_partition(
            candidate_manifest,
            assignments[assignments["fold"].eq(fold)],
        )
        fold_candidates["fold"] = fold
        fold_candidates = fold_candidates[
            fold_candidates["fold_split"].isin(["train", "val", "test"])
        ].copy()

        matched, match_audit = rematch_control_strata_within_partition(
            fold_positives,
            fold_candidates,
            policy=control_policy,
        )
        match_audit["fold"] = fold
        audit_frames.append(match_audit)

        complete_ids = _complete_positive_ids(match_audit)
        kept_positives = fold_positives[
            fold_positives["sample_id"].astype(str).isin(complete_ids)
        ].copy()
        matched = matched[
            matched["matched_positive_id"].astype(str).isin(complete_ids)
        ].copy()

        kept_positives["control_stratum"] = TRANSITION
        kept_positives["evaluation_stratum"] = TRANSITION
        kept_positives["tensor_source"] = "positive"
        kept_positives["tensor_row"] = kept_positives["_positive_tensor_row"].astype(int)
        matched["evaluation_stratum"] = matched["control_stratum"].astype(str)
        matched["tensor_source"] = "candidate"
        matched["tensor_row"] = matched["_candidate_row"].astype(int)

        combined = pd.concat([kept_positives, matched], ignore_index=True, sort=False)
        combined = combined.sort_values(
            [
                "fold_split",
                "index",
                "lead",
                "evaluation_stratum",
                "sample_id",
            ],
            kind="mergesort",
        ).reset_index(drop=True)

        tensors = []
        for _, row in combined.iterrows():
            if row["tensor_source"] == "positive":
                tensors.append(stage_d_tensor[int(row["tensor_row"])])
            else:
                tensors.append(candidate_tensor[int(row["tensor_row"])])
        fold_tensor = np.asarray(tensors, dtype=float)
        combined["_fold_tensor_row"] = np.arange(len(combined), dtype=int)
        output_frames.append(combined)
        output_tensors.append(fold_tensor)

    manifest = pd.concat(output_frames, ignore_index=True)
    tensor = np.concatenate(output_tensors, axis=0)
    manifest["_global_tensor_row"] = np.arange(len(manifest), dtype=int)
    audit = pd.concat(audit_frames, ignore_index=True)

    fold_quality = []
    for (fold, stratum), group in audit.groupby(
        ["fold", "control_stratum"], sort=True
    ):
        fold_quality.append(
            {
                "fold": int(fold),
                "control_stratum": str(stratum),
                "positives_requested": int(len(group)),
                "positives_complete_in_stratum": int(group["complete_match"].sum()),
                "positives_complete_all_strata": int(
                    group["complete_all_strata"].sum()
                ),
                "completion_rate_in_stratum": float(group["complete_match"].mean()),
                "median_max_selected_distance": float(
                    group["max_selected_distance"].median()
                ),
                "p95_max_selected_distance": float(
                    group["max_selected_distance"].quantile(0.95)
                ),
            }
        )

    stratum_counts = (
        manifest.groupby(["fold", "fold_split", "evaluation_stratum"], sort=True)
        .size()
        .rename("rows")
        .reset_index()
        .to_dict(orient="records")
    )
    summary = {
        "n_folds": int(n_folds),
        "test_fraction": float(test_fraction),
        "controls_per_positive": int(controls_per_positive),
        "control_policy": control_policy.to_dict(),
        "manifest_rows": int(len(manifest)),
        "tensor_shape": list(tensor.shape),
        "split_audit": split_audit,
        "fold_summaries": fold_summaries,
        "fold_match_quality_by_stratum": fold_quality,
        "fold_stratum_counts": stratum_counts,
        "positive_intervals": "exact_trading_rows",
        "candidate_intervals": "exact_trading_rows",
        "matching_information": "pre_origin_only",
        "negative_stratum_information": "future_persistence_label_only",
        "test_evaluated": False,
    }
    return manifest, tensor, audit, summary


def write_rematched_rolling_dataset(
    stage_d_run: Path,
    run_dir: Path,
    *,
    n_folds: int,
    test_fraction: float = 0.17,
    embargo_days: int = 10,
    controls_per_positive: int = 3,
) -> dict[str, object]:
    stage_d_manifest = pd.read_csv(stage_d_run / "sample_manifest.csv")
    with np.load(stage_d_run / "sequence_tensors.npz", allow_pickle=True) as data:
        stage_d_tensor = np.asarray(data["X"], dtype=float)
    candidate_manifest, candidate_tensor = load_candidate_pool(stage_d_run)

    manifest, tensor, audit, summary = build_rematched_rolling_dataset(
        stage_d_manifest,
        stage_d_tensor,
        candidate_manifest,
        candidate_tensor,
        n_folds=n_folds,
        test_fraction=test_fraction,
        embargo_days=embargo_days,
        controls_per_positive=controls_per_positive,
    )
    manifest.to_csv(run_dir / "rematched_rolling_manifest.csv", index=False)
    audit.to_csv(run_dir / "control_match_audit.csv", index=False)
    np.savez_compressed(
        run_dir / "rematched_rolling_tensors.npz",
        X=tensor,
        sample_id=manifest["sample_id"].astype(str).to_numpy(),
        fold=manifest["fold"].astype(int).to_numpy(),
        fold_split=manifest["fold_split"].astype(str).to_numpy(),
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
