from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

CANONICAL_FOLD_SPLITS = ("train", "val", "test")
DEVELOPMENT_FOLD_SPLITS = ("train", "val")


def validate_fold_manifest_schema(manifest: pd.DataFrame) -> None:
    required = {
        "sample_id",
        "label",
        "episode_id",
        "origin_date",
        "lead",
        "fold",
        "fold_split",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    observed = set(manifest["fold_split"].dropna().astype(str).unique())
    expected = set(CANONICAL_FOLD_SPLITS)
    if observed != expected:
        raise ValueError(
            f"unexpected fold_split values {sorted(observed)}; "
            f"expected {sorted(expected)}"
        )


def select_balanced_episode_rows(
    manifest: pd.DataFrame,
    *,
    fold: int,
    lead: int,
    max_per_class: int,
    excluded_sample_ids: Iterable[str] = (),
    splits: tuple[str, ...] = DEVELOPMENT_FOLD_SPLITS,
    seed: int = 0,
) -> pd.DataFrame:
    validate_fold_manifest_schema(manifest)

    unknown_splits = set(splits).difference(CANONICAL_FOLD_SPLITS)
    if unknown_splits:
        raise ValueError(f"unknown requested splits: {sorted(unknown_splits)}")
    if "test" in splits:
        raise ValueError("test split selection is prohibited for development assays")
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive")

    excluded = {str(value) for value in excluded_sample_ids}
    rows = manifest.loc[
        manifest["fold"].eq(fold)
        & manifest["fold_split"].isin(splits)
        & manifest["lead"].eq(lead)
    ].copy()
    rows["sample_id"] = rows["sample_id"].astype(str)
    rows = rows.loc[~rows["sample_id"].isin(excluded)]
    rows = rows.sort_values(
        ["fold_split", "label", "episode_id", "origin_date", "sample_id"]
    )
    rows = rows.drop_duplicates(
        ["fold_split", "label", "episode_id"], keep="first"
    )

    selected_parts: list[pd.DataFrame] = []
    for split_offset, fold_split in enumerate(splits):
        split_rows = rows.loc[rows["fold_split"].eq(fold_split)]
        class_parts: dict[int, pd.DataFrame] = {}
        for label in (0, 1):
            group = split_rows.loc[split_rows["label"].eq(label)].copy()
            rng = np.random.default_rng(seed + 1000 * fold + 100 * split_offset + label)
            if len(group) > max_per_class:
                positions = np.sort(
                    rng.choice(len(group), size=max_per_class, replace=False)
                )
                group = group.iloc[positions]
            class_parts[label] = group

        counts = {label: len(class_parts[label]) for label in (0, 1)}
        if counts[0] == 0 or counts[1] == 0:
            raise RuntimeError(
                f"fold {fold} split {fold_split}: selection lacks one class: {counts}"
            )

        n_balanced = min(counts.values())
        selected_parts.extend(
            [class_parts[label].head(n_balanced) for label in (0, 1)]
        )

    selected = pd.concat(selected_parts, ignore_index=True)
    return selected.sort_values(
        ["fold_split", "label", "episode_id", "sample_id"]
    ).reset_index(drop=True)
