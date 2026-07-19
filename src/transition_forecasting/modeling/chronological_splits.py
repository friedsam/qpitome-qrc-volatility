from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IntervalColumns:
    origin: str = "origin_date"
    index: str = "index"
    label: str = "label"
    episode: str = "episode_id"
    sample_id: str = "sample_id"


def _as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="raise", utc=True)


def _deduplicate_samples(frame: pd.DataFrame, columns: IntervalColumns) -> tuple[pd.DataFrame, int]:
    key = [columns.index, columns.origin]
    duplicated = frame.duplicated(key, keep="first")
    return frame.loc[~duplicated].copy(), int(duplicated.sum())


def _chronology_groups(frame: pd.DataFrame, columns: IntervalColumns) -> pd.Series:
    labels = frame[columns.label].astype(int)
    positive_group = "P:" + frame[columns.episode].astype(str)
    control_group = "C:" + frame[columns.sample_id].astype(str)
    return pd.Series(np.where(labels.eq(1), positive_group, control_group), index=frame.index, dtype=object)


def _interval_bounds(
    frame: pd.DataFrame,
    columns: IntervalColumns,
    *,
    input_lookback_days: int,
    target_horizon_days: int,
) -> tuple[pd.Series, pd.Series]:
    origin = _as_utc(frame[columns.origin])
    start = origin - pd.to_timedelta(int(input_lookback_days), unit="D")
    end = origin + pd.to_timedelta(int(target_horizon_days), unit="D")
    return start, end


def _purge_cross_partition_overlaps(
    frame: pd.DataFrame,
    *,
    split_column: str,
    interval_start: str,
    interval_end: str,
    index_column: str,
) -> int:
    purged = 0
    order = {"train": 0, "val": 1, "test": 2}
    active = frame[frame[split_column].isin(order)].copy()
    for _, market in active.groupby(index_column, sort=False):
        rows = market.sort_values(interval_start)
        indices = rows.index.to_list()
        for left_position, left_index in enumerate(indices):
            left_split = frame.at[left_index, split_column]
            if left_split not in order:
                continue
            left_end = frame.at[left_index, interval_end]
            for right_index in indices[left_position + 1 :]:
                if frame.at[right_index, interval_start] > left_end:
                    break
                right_split = frame.at[right_index, split_column]
                if right_split not in order or left_split == right_split:
                    continue
                later_index = right_index if order[right_split] >= order[left_split] else left_index
                if frame.at[later_index, split_column] != "purged_overlap":
                    frame.at[later_index, split_column] = "purged_overlap"
                    purged += 1
    return purged


def rolling_origin_assignments(
    manifest: pd.DataFrame,
    *,
    n_folds: int = 3,
    test_fraction: float = 0.17,
    embargo_days: int = 10,
    input_lookback_days: int = 60,
    target_horizon_days: int = 20,
    columns: IntervalColumns = IntervalColumns(),
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object]]:
    required = {
        columns.origin,
        columns.index,
        columns.label,
        columns.episode,
        columns.sample_id,
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing required chronology columns: {sorted(missing)}")
    if n_folds < 1:
        raise ValueError("n_folds must be positive")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")

    frame = manifest.reset_index(drop=True).copy()
    frame["_source_row"] = np.arange(len(frame), dtype=int)
    frame, duplicate_count = _deduplicate_samples(frame, columns)
    frame["_origin"] = _as_utc(frame[columns.origin])
    frame["chronology_group"] = _chronology_groups(frame, columns)
    frame["interval_start"], frame["interval_end"] = _interval_bounds(
        frame,
        columns,
        input_lookback_days=input_lookback_days,
        target_horizon_days=target_horizon_days,
    )

    group_dates = (
        frame.groupby("chronology_group", as_index=False)["_origin"]
        .min()
        .sort_values(["_origin", "chronology_group"])
        .reset_index(drop=True)
    )
    n_groups = len(group_dates)
    n_test = max(1, int(round(n_groups * test_fraction)))
    n_dev = n_groups - n_test
    if n_dev < n_folds + 1:
        raise ValueError("not enough chronological groups for requested rolling folds")

    blocks = [np.asarray(block, dtype=int) for block in np.array_split(np.arange(n_dev), n_folds + 1)]
    test_groups = set(group_dates.loc[n_dev:, "chronology_group"])
    embargo = pd.Timedelta(days=int(embargo_days))
    assignments: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for fold in range(n_folds):
        train_groups = set(group_dates.loc[np.concatenate(blocks[: fold + 1]), "chronology_group"])
        val_indices = blocks[fold + 1]
        val_groups = set(group_dates.loc[val_indices, "chronology_group"])
        boundary = group_dates.loc[val_indices[0], "_origin"]

        fold_frame = frame.copy()
        fold_frame["fold"] = fold + 1
        fold_frame["fold_split"] = "unused"
        fold_frame.loc[fold_frame["chronology_group"].isin(train_groups), "fold_split"] = "train"
        fold_frame.loc[fold_frame["chronology_group"].isin(val_groups), "fold_split"] = "val"
        fold_frame.loc[fold_frame["chronology_group"].isin(test_groups), "fold_split"] = "test"

        near_boundary = (fold_frame["interval_end"] >= boundary - embargo) & (
            fold_frame["interval_start"] <= boundary + embargo
        )
        fold_frame.loc[
            near_boundary & fold_frame["fold_split"].isin(["train", "val"]),
            "fold_split",
        ] = "purged_embargo"

        overlap_purged = _purge_cross_partition_overlaps(
            fold_frame,
            split_column="fold_split",
            interval_start="interval_start",
            interval_end="interval_end",
            index_column=columns.index,
        )

        train = fold_frame[fold_frame["fold_split"].eq("train")]
        val = fold_frame[fold_frame["fold_split"].eq("val")]
        if train.empty or val.empty:
            raise ValueError(f"fold {fold + 1} has an empty train or validation partition")
        if train["interval_end"].max() >= val["interval_start"].min():
            raise ValueError(f"fold {fold + 1} is not strictly chronological after purging")

        summaries.append(
            {
                "fold": fold + 1,
                "boundary": str(boundary),
                "train_groups": int(train["chronology_group"].nunique()),
                "val_groups": int(val["chronology_group"].nunique()),
                "test_groups": int(
                    fold_frame.loc[fold_frame["fold_split"].eq("test"), "chronology_group"].nunique()
                ),
                "train_samples": int(len(train)),
                "val_samples": int(len(val)),
                "purged_embargo_samples": int(fold_frame["fold_split"].eq("purged_embargo").sum()),
                "purged_overlap_samples": int(overlap_purged),
            }
        )
        assignments.append(fold_frame)

    combined = pd.concat(assignments, ignore_index=True).drop(columns="_origin")
    audit = {
        "input_samples": int(len(manifest)),
        "deduplicated_samples": int(len(frame)),
        "duplicate_index_origin_removed": duplicate_count,
        "input_lookback_days": int(input_lookback_days),
        "target_horizon_days": int(target_horizon_days),
        "embargo_days": int(embargo_days),
        "control_groups_use_actual_origin": True,
        "positive_groups_use_episode_id": True,
    }
    return combined, summaries, audit
