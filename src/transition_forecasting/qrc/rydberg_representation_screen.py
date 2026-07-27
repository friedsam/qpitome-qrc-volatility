"""Exact balanced-row selection used by the canonical Case151 assay."""
from __future__ import annotations

import pandas as pd

from transition_forecasting.modeling.fold_selection import (
    DEVELOPMENT_FOLD_SPLITS,
    select_balanced_episode_rows,
)


def _select_rows_for_fold(
    manifest: pd.DataFrame,
    *,
    fold: int,
    leads: tuple[int, ...],
    max_per_class: int,
    seed: int,
) -> pd.DataFrame:
    parts = [
        select_balanced_episode_rows(
            manifest,
            fold=fold,
            lead=int(lead),
            max_per_class=max_per_class,
            splits=DEVELOPMENT_FOLD_SPLITS,
            seed=seed + int(lead) * 1009,
        )
        for lead in leads
    ]
    selected = pd.concat(parts, ignore_index=True)
    if selected["_tensor_row"].duplicated().any():
        raise RuntimeError(f"fold {fold}: duplicate selected tensor rows")
    if selected["fold_split"].eq("test").any():
        raise RuntimeError("representation screen must not receive test rows")
    return selected.sort_values(
        [
            "fold_split",
            "lead",
            "label",
            "episode_id",
            "origin_date",
            "sample_id",
        ]
    ).reset_index(drop=True)
