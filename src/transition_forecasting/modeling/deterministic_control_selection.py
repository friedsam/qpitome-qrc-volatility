from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.modeling.control_strata import (
    CALM,
    HARD_NEGATIVE,
    TRANSITION,
)

DEVELOPMENT_SPLITS = ("train", "val")


def _stable_bin_midpoints(length: int, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    if length <= count:
        return np.arange(length, dtype=int)
    return np.floor((np.arange(count, dtype=float) + 0.5) * length / count).astype(int)


def select_deterministic_control_panel(
    manifest: pd.DataFrame,
    *,
    fold: int,
    leads: tuple[int, ...] = (1, 5, 10),
    max_episodes_per_split_lead: int | None = None,
    splits: tuple[str, ...] = DEVELOPMENT_SPLITS,
) -> pd.DataFrame:
    """Select an equal transition/calm/hard panel without random sampling.

    Every eligible transition episode is considered before an optional deterministic
    chronological cap. One representative positive row is retained per global episode,
    then the closest pre-origin-matched calm and hard controls linked to that positive
    are selected. The result has equal counts in all three evaluation strata.
    """

    required = {
        "sample_id",
        "label",
        "episode_id",
        "event_onset",
        "origin_date",
        "lead",
        "fold",
        "fold_split",
        "index",
        "evaluation_stratum",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if "test" in splits:
        raise ValueError("test split selection is prohibited for development assays")
    if max_episodes_per_split_lead is not None and max_episodes_per_split_lead < 1:
        raise ValueError("max_episodes_per_split_lead must be positive")

    frame = manifest.loc[
        manifest["fold"].eq(int(fold))
        & manifest["fold_split"].isin(splits)
        & manifest["lead"].isin(leads)
    ].copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame.empty:
        raise RuntimeError(f"fold {fold}: no eligible development rows")

    selected_parts: list[pd.DataFrame] = []
    for fold_split in splits:
        for lead in leads:
            block = frame.loc[
                frame["fold_split"].eq(fold_split) & frame["lead"].eq(int(lead))
            ].copy()
            positives = block.loc[
                block["label"].eq(1)
                & block["evaluation_stratum"].astype(str).eq(TRANSITION)
            ].sort_values(
                ["event_onset", "episode_id", "index", "origin_date", "sample_id"],
                kind="mergesort",
            )
            positives = positives.drop_duplicates("episode_id", keep="first").reset_index(
                drop=True
            )
            if max_episodes_per_split_lead is not None:
                positions = _stable_bin_midpoints(
                    len(positives),
                    max_episodes_per_split_lead,
                )
                positives = positives.iloc[positions].reset_index(drop=True)

            controls = block.loc[block["label"].eq(0)].copy()
            if "matched_positive_id" not in controls.columns or "match_distance" not in controls.columns:
                raise ValueError("control rows require matched_positive_id and match_distance")
            controls["matched_positive_id"] = controls["matched_positive_id"].astype(str)

            complete_positive_ids: list[str] = []
            selected_controls: list[pd.DataFrame] = []
            for _, positive in positives.iterrows():
                positive_id = str(positive["sample_id"])
                linked = controls.loc[
                    controls["matched_positive_id"].eq(positive_id)
                ].copy()
                local_parts: list[pd.DataFrame] = []
                for stratum in (CALM, HARD_NEGATIVE):
                    local = linked.loc[
                        linked["evaluation_stratum"].astype(str).eq(stratum)
                    ].sort_values(
                        ["match_distance", "origin_date", "index", "sample_id"],
                        kind="mergesort",
                    )
                    if local.empty:
                        local_parts = []
                        break
                    local_parts.append(local.head(1))
                if len(local_parts) == 2:
                    complete_positive_ids.append(positive_id)
                    selected_controls.extend(local_parts)

            kept_positives = positives.loc[
                positives["sample_id"].isin(complete_positive_ids)
            ]
            if kept_positives.empty:
                raise RuntimeError(
                    f"fold {fold} split {fold_split} lead {lead}: no complete stratified episodes"
                )
            selected_parts.append(kept_positives)
            selected_parts.extend(selected_controls)

    selected = pd.concat(selected_parts, ignore_index=True, sort=False)
    counts = selected.groupby(
        ["fold_split", "lead", "evaluation_stratum"], sort=True
    ).size()
    for (fold_split, lead), group in counts.groupby(level=[0, 1]):
        local = group.droplevel([0, 1]).to_dict()
        expected = local.get(TRANSITION, 0)
        if expected < 1 or any(local.get(stratum, 0) != expected for stratum in (CALM, HARD_NEGATIVE)):
            raise RuntimeError(
                f"fold {fold} split {fold_split} lead {lead}: unbalanced strata {local}"
            )
    if selected["fold_split"].eq("test").any():
        raise RuntimeError("deterministic control panel contains test rows")
    if "_tensor_row" in selected.columns and selected["_tensor_row"].duplicated().any():
        raise RuntimeError("deterministic control panel contains duplicate tensor rows")

    return selected.sort_values(
        [
            "fold_split",
            "lead",
            "evaluation_stratum",
            "event_onset",
            "episode_id",
            "origin_date",
            "sample_id",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
