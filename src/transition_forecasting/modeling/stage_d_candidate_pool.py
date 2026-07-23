from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.global_transition_catalogue import TRAIN_CUTOFF
from transition_forecasting.catalogue.transition_events import (
    HORIZON,
    LEADS,
    MATCH_FEATURES,
    NEG_EXCL,
    WINDOW,
    causal_features,
)
from transition_forecasting.modeling.control_strata import (
    PERSISTENT_EXCLUDED,
    ControlStrataPolicy,
    classify_control_future,
)
from transition_forecasting.modeling.global_stage_d_dataset import _load_series

TARGET_COLUMNS = tuple(f"target_x_h{i + 1}" for i in range(HORIZON))


def _resolve_threshold(series: pd.Series, threshold: float | None) -> float:
    if threshold is not None:
        value = float(threshold)
    else:
        training = series[series.index < TRAIN_CUTOFF]
        if training.empty:
            raise ValueError("cannot derive the frozen pre-2016 volatility threshold")
        value = float(training.quantile(0.80))
    if not np.isfinite(value):
        raise ValueError("control threshold must be finite")
    return value


def build_candidate_pool_from_series(
    *,
    index_name: str,
    series: pd.Series,
    onset_positions: np.ndarray,
    leads: tuple[int, ...] = LEADS,
    event_exclusion: int = NEG_EXCL,
    threshold: float | None = None,
    control_policy: ControlStrataPolicy = ControlStrataPolicy(),
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build every eligible stratified negative-control candidate.

    Candidate eligibility excludes persistent transitions whose onset begins within
    the forecast horizon. Outcome labels mirror the catalogue onset rule; all matching
    features are computed from the pre-origin history only.
    """

    control_policy.validate()
    resolved_threshold = _resolve_threshold(series, threshold)
    rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    onsets = np.asarray(onset_positions, dtype=int)
    onsets = onsets[onsets >= 0]
    required_future = control_policy.future_assessment_rows

    for position in range(WINDOW - 1, len(series) - required_future):
        if len(onsets) and int(np.min(np.abs(onsets - position))) <= int(event_exclusion):
            continue
        features = causal_features(series, position)
        if features is None:
            continue
        prior_history = series.iloc[
            position - control_policy.prior_window + 1 : position + 1
        ].to_numpy(dtype=float)
        assessment = series.iloc[
            position + 1 : position + 1 + required_future
        ].to_numpy(dtype=float)
        stratum_payload = classify_control_future(
            assessment,
            prior_history=prior_history,
            threshold=resolved_threshold,
            policy=control_policy,
        )
        if stratum_payload["control_stratum"] == PERSISTENT_EXCLUDED:
            continue

        target = series.iloc[position + 1 : position + 1 + HORIZON].to_numpy(dtype=float)
        sequence = series.iloc[position - WINDOW + 1 : position + 1].to_numpy(dtype=float)[:, None]
        input_start_date = series.index[position - WINDOW + 1]
        target_end_date = series.index[position + HORIZON]
        assessment_end_date = series.index[position + required_future]

        for lead in leads:
            candidate_id = f"C_{index_name}_L{int(lead)}_P{position}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "index": str(index_name),
                    "lead": int(lead),
                    "origin_pos": int(position),
                    "origin_date": series.index[position],
                    "input_start_date": input_start_date,
                    "target_end_date": target_end_date,
                    "future_assessment_end_date": assessment_end_date,
                    "control_threshold": resolved_threshold,
                    **stratum_payload,
                    **{feature: float(features[feature]) for feature in MATCH_FEATURES},
                    **{
                        column: float(value)
                        for column, value in zip(TARGET_COLUMNS, target, strict=True)
                    },
                }
            )
            sequences.append(sequence)

    frame = pd.DataFrame(rows)
    tensor = (
        np.asarray(sequences, dtype=float)
        if sequences
        else np.empty((0, WINDOW, 1), dtype=float)
    )
    if not frame.empty:
        if frame["candidate_id"].duplicated().any():
            raise ValueError("candidate IDs are not unique")
        expected = (len(frame), WINDOW, 1)
        if tensor.shape != expected:
            raise ValueError(
                f"candidate tensor shape mismatch: expected {expected}, got {tensor.shape}"
            )
        if frame["future_persistent"].astype(bool).any():
            raise ValueError("persistent-transition candidates leaked into the control pool")
    return frame, tensor


def build_global_candidate_pool(
    representative_catalogue_path: Path,
    inventory_path: Path,
    *,
    control_policy: ControlStrataPolicy = ControlStrataPolicy(),
) -> tuple[pd.DataFrame, np.ndarray, dict[str, object]]:
    catalogue = pd.read_csv(
        representative_catalogue_path,
        parse_dates=["onset_date", "effective_start"],
    )
    inventory = pd.read_csv(inventory_path)
    series_by_index = _load_series(catalogue, inventory)

    frames: list[pd.DataFrame] = []
    tensors: list[np.ndarray] = []
    per_index: dict[str, int] = {}
    thresholds = catalogue.groupby("index", sort=False)["threshold"].first().astype(float)
    for index_name, group in catalogue.groupby("index", sort=True):
        series = series_by_index[str(index_name)]
        positions = series.index.get_indexer(pd.to_datetime(group["onset_date"]))
        frame, tensor = build_candidate_pool_from_series(
            index_name=str(index_name),
            series=series,
            onset_positions=positions,
            threshold=float(thresholds.loc[index_name]),
            control_policy=control_policy,
        )
        frames.append(frame)
        tensors.append(tensor)
        per_index[str(index_name)] = int(len(frame))

    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tensor = (
        np.concatenate(tensors, axis=0)
        if tensors
        else np.empty((0, WINDOW, 1), dtype=float)
    )
    if len(manifest) != len(tensor):
        raise ValueError("candidate manifest and tensor lengths differ")

    strata_counts = (
        manifest["control_stratum"].value_counts().sort_index().astype(int).to_dict()
        if len(manifest)
        else {}
    )
    summary = {
        "candidate_rows": int(len(manifest)),
        "unique_control_origins": int(
            manifest[["index", "origin_date"]].drop_duplicates().shape[0]
        )
        if len(manifest)
        else 0,
        "indices": int(manifest["index"].nunique()) if len(manifest) else 0,
        "leads": sorted(int(value) for value in manifest["lead"].unique())
        if len(manifest)
        else [],
        "control_strata_rows": strata_counts,
        "control_policy": control_policy.to_dict(),
        "sequence_shape": list(tensor.shape),
        "window_trading_rows": int(WINDOW),
        "target_horizon_trading_rows": int(HORIZON),
        "future_assessment_trading_rows": int(
            control_policy.future_assessment_rows
        ),
        "event_exclusion_trading_rows": int(NEG_EXCL),
        "rows_by_index": per_index,
    }
    return manifest, tensor, summary


def write_global_candidate_pool(
    representative_catalogue_path: Path,
    inventory_path: Path,
    run_dir: Path,
    *,
    control_policy: ControlStrataPolicy = ControlStrataPolicy(),
) -> dict[str, object]:
    manifest, tensor, summary = build_global_candidate_pool(
        representative_catalogue_path,
        inventory_path,
        control_policy=control_policy,
    )
    manifest.to_csv(run_dir / "control_candidate_manifest.csv", index=False)
    np.savez_compressed(
        run_dir / "control_candidate_tensors.npz",
        X=tensor,
        candidate_id=manifest["candidate_id"].astype(str).to_numpy(),
        index=manifest["index"].astype(str).to_numpy(),
        lead=manifest["lead"].astype(int).to_numpy(),
    )
    (run_dir / "candidate_pool_summary.json").write_text(
        pd.Series(summary).to_json(indent=2) + "\n"
    )
    return summary
