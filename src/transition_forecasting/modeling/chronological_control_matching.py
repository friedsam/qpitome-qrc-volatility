from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.transition_events import MATCH_FEATURES


@dataclass(frozen=True)
class MatchConfig:
    controls_per_positive: int = 3
    max_match_distance: float | None = None


def _validate_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def rematch_controls_within_partition(
    positives: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    partition_columns: tuple[str, ...] = ("fold", "fold_split", "index", "lead"),
    config: MatchConfig = MatchConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match controls without reuse inside each strict chronological partition.

    Candidates must already satisfy interval, event-exclusion, and partition-eligibility
    constraints. Matching is deterministic: positives are ordered by sample_id and ties
    are broken by origin position and origin date.
    """

    required_positive = {
        "sample_id",
        "episode_id",
        "market_group",
        "event_onset",
        "label",
        *partition_columns,
        *MATCH_FEATURES,
    }
    required_candidate = {
        "origin_pos",
        "origin_date",
        "input_start_date",
        "target_end_date",
        *partition_columns,
        *MATCH_FEATURES,
    }
    _validate_columns(positives, required_positive, "positives")
    _validate_columns(candidates, required_candidate, "candidates")

    if not positives["label"].eq(1).all():
        raise ValueError("positives must contain label == 1 only")
    if config.controls_per_positive < 1:
        raise ValueError("controls_per_positive must be positive")

    matched_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    group_key: str | list[str]
    group_key = list(partition_columns) if len(partition_columns) > 1 else partition_columns[0]
    for key, positive_group in positives.groupby(group_key, sort=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        candidate_mask = np.ones(len(candidates), dtype=bool)
        for column, value in zip(partition_columns, key_tuple, strict=True):
            candidate_mask &= candidates[column].eq(value).to_numpy()
        candidate_group = candidates.loc[candidate_mask].copy()
        candidate_group = candidate_group.sort_values(["origin_pos", "origin_date"]).reset_index(drop=True)

        used_positions: set[int] = set()
        if candidate_group.empty:
            for _, positive in positive_group.iterrows():
                audit_rows.append(
                    {
                        **{column: value for column, value in zip(partition_columns, key_tuple, strict=True)},
                        "positive_sample_id": str(positive["sample_id"]),
                        "available_candidates": 0,
                        "controls_selected": 0,
                        "complete_match": False,
                        "max_selected_distance": np.nan,
                    }
                )
            continue

        feature_values = candidate_group[list(MATCH_FEATURES)].astype(float)
        mean = feature_values.mean()
        scale = feature_values.std(ddof=0).replace(0.0, 1.0)
        standardized_candidates = (feature_values - mean) / scale

        for _, positive in positive_group.sort_values("sample_id").iterrows():
            standardized_positive = (positive[list(MATCH_FEATURES)].astype(float) - mean) / scale
            distances = np.sqrt(
                ((standardized_candidates - standardized_positive.to_numpy(dtype=float)) ** 2).sum(axis=1)
            )
            order = np.lexsort(
                (
                    candidate_group["origin_date"].astype(str).to_numpy(),
                    candidate_group["origin_pos"].to_numpy(dtype=int),
                    distances.to_numpy(dtype=float),
                )
            )

            selected_distances: list[float] = []
            rank = 0
            for candidate_index in order:
                candidate = candidate_group.iloc[int(candidate_index)]
                origin_pos = int(candidate["origin_pos"])
                distance = float(distances.iloc[int(candidate_index)])
                if origin_pos in used_positions:
                    continue
                if config.max_match_distance is not None and distance > config.max_match_distance:
                    continue
                used_positions.add(origin_pos)
                rank += 1
                selected_distances.append(distance)
                row = candidate.to_dict()
                row.update(
                    {
                        "sample_id": f"N_{positive['sample_id']}_{rank}",
                        "label": 0,
                        "episode_id": positive["episode_id"],
                        "market_group": positive["market_group"],
                        "event_onset": positive["event_onset"],
                        "matched_positive_id": positive["sample_id"],
                        "match_distance": distance,
                    }
                )
                matched_rows.append(row)
                if rank >= config.controls_per_positive:
                    break

            audit_rows.append(
                {
                    **{column: value for column, value in zip(partition_columns, key_tuple, strict=True)},
                    "positive_sample_id": str(positive["sample_id"]),
                    "available_candidates": int(len(candidate_group)),
                    "controls_selected": int(rank),
                    "complete_match": bool(rank == config.controls_per_positive),
                    "max_selected_distance": max(selected_distances) if selected_distances else np.nan,
                }
            )

    matched = pd.DataFrame(matched_rows)
    audit = pd.DataFrame(audit_rows)
    if not matched.empty and matched.duplicated([*partition_columns, "index", "origin_date"]).any():
        raise ValueError("control origin reuse detected after matching")
    return matched, audit
