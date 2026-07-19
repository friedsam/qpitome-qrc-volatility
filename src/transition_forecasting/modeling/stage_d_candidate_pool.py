from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.transition_events import (
    HORIZON,
    LEADS,
    MATCH_FEATURES,
    NEG_EXCL,
    WINDOW,
    causal_features,
)
from transition_forecasting.modeling.global_stage_d_dataset import _load_series

TARGET_COLUMNS = tuple(f"target_x_h{i + 1}" for i in range(HORIZON))


def build_candidate_pool_from_series(
    *,
    index_name: str,
    series: pd.Series,
    onset_positions: np.ndarray,
    leads: tuple[int, ...] = LEADS,
    event_exclusion: int = NEG_EXCL,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build every eligible negative-control candidate with trading-row intervals."""

    rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    onsets = np.asarray(onset_positions, dtype=int)
    onsets = onsets[onsets >= 0]

    for position in range(WINDOW - 1, len(series) - HORIZON):
        if len(onsets) and int(np.min(np.abs(onsets - position))) <= int(event_exclusion):
            continue
        features = causal_features(series, position)
        if features is None:
            continue
        target = series.iloc[position + 1 : position + 1 + HORIZON].to_numpy(dtype=float)
        sequence = series.iloc[position - WINDOW + 1 : position + 1].to_numpy(dtype=float)[:, None]
        input_start_date = series.index[position - WINDOW + 1]
        target_end_date = series.index[position + HORIZON]

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
                    **{feature: float(features[feature]) for feature in MATCH_FEATURES},
                    **{column: float(value) for column, value in zip(TARGET_COLUMNS, target, strict=True)},
                }
            )
            sequences.append(sequence)

    frame = pd.DataFrame(rows)
    tensor = np.asarray(sequences, dtype=float)
    if not frame.empty:
        if frame["candidate_id"].duplicated().any():
            raise ValueError("candidate IDs are not unique")
        expected = (len(frame), WINDOW, 1)
        if tensor.shape != expected:
            raise ValueError(f"candidate tensor shape mismatch: expected {expected}, got {tensor.shape}")
    return frame, tensor


def build_global_candidate_pool(
    representative_catalogue_path: Path,
    inventory_path: Path,
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
    for index_name, group in catalogue.groupby("index", sort=True):
        series = series_by_index[str(index_name)]
        positions = series.index.get_indexer(pd.to_datetime(group["onset_date"]))
        frame, tensor = build_candidate_pool_from_series(
            index_name=str(index_name),
            series=series,
            onset_positions=positions,
        )
        frames.append(frame)
        tensors.append(tensor)
        per_index[str(index_name)] = int(len(frame))

    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tensor = np.concatenate(tensors, axis=0) if tensors else np.empty((0, WINDOW, 1), dtype=float)
    if len(manifest) != len(tensor):
        raise ValueError("candidate manifest and tensor lengths differ")

    summary = {
        "candidate_rows": int(len(manifest)),
        "unique_control_origins": int(manifest[["index", "origin_date"]].drop_duplicates().shape[0])
        if len(manifest)
        else 0,
        "indices": int(manifest["index"].nunique()) if len(manifest) else 0,
        "leads": sorted(int(value) for value in manifest["lead"].unique()) if len(manifest) else [],
        "sequence_shape": list(tensor.shape),
        "window_trading_rows": int(WINDOW),
        "target_horizon_trading_rows": int(HORIZON),
        "event_exclusion_trading_rows": int(NEG_EXCL),
        "rows_by_index": per_index,
    }
    return manifest, tensor, summary


def write_global_candidate_pool(
    representative_catalogue_path: Path,
    inventory_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    manifest, tensor, summary = build_global_candidate_pool(
        representative_catalogue_path,
        inventory_path,
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
