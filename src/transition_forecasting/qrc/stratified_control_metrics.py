from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from transition_forecasting.qrc.representation_screen_analysis import _metric_payload


def attach_evaluation_strata(
    predictions: pd.DataFrame,
    fold_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach frozen control strata to validation predictions without guessing."""

    prediction_required = {
        "fold",
        "sample_id",
        "model_name",
        "lead",
        "y_true",
        "y_pred",
    }
    manifest_required = {"fold", "sample_id", "evaluation_stratum"}
    prediction_missing = prediction_required.difference(predictions.columns)
    manifest_missing = manifest_required.difference(fold_manifest.columns)
    if prediction_missing:
        raise ValueError(
            f"predictions are missing columns: {sorted(prediction_missing)}"
        )
    if manifest_missing:
        raise ValueError(
            f"fold manifest is missing columns: {sorted(manifest_missing)}"
        )

    frame = predictions.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["fold"] = frame["fold"].astype(int)

    lookup = fold_manifest[["fold", "sample_id", "evaluation_stratum"]].copy()
    lookup["sample_id"] = lookup["sample_id"].astype(str)
    lookup["fold"] = lookup["fold"].astype(int)
    duplicate_groups = (
        lookup.groupby(["fold", "sample_id"], sort=False)["evaluation_stratum"]
        .nunique(dropna=False)
    )
    if duplicate_groups.gt(1).any():
        raise ValueError("fold/sample IDs map to multiple evaluation strata")
    lookup = lookup.drop_duplicates(["fold", "sample_id"], keep="first")

    if "evaluation_stratum" in frame.columns:
        frame = frame.drop(columns="evaluation_stratum")
    merged = frame.merge(
        lookup,
        on=["fold", "sample_id"],
        how="left",
        validate="many_to_one",
    )
    if merged["evaluation_stratum"].isna().any():
        missing = merged.loc[
            merged["evaluation_stratum"].isna(), ["fold", "sample_id"]
        ].drop_duplicates()
        raise ValueError(
            "predictions lack frozen stratum linkage for "
            f"{len(missing)} fold/sample pairs"
        )
    return merged


def stratified_metric_table(
    predictions: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Compute the incumbent metric payload for explicit frozen strata."""

    columns = tuple(str(value) for value in group_columns)
    if not columns:
        raise ValueError("group_columns cannot be empty")
    required = {"fold", "sample_id", "y_true", "y_pred", *columns}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    grouping: str | list[str] = list(columns) if len(columns) > 1 else columns[0]
    for keys, local in predictions.groupby(grouping, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        observed = local["y_true"].to_numpy(dtype=float)[:, None]
        forecast = local["y_pred"].to_numpy(dtype=float)[:, None]
        payload = _metric_payload(
            observed,
            forecast,
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                **dict(zip(columns, key_tuple, strict=True)),
                "folds": int(local["fold"].nunique()),
                "samples": int(local[["fold", "sample_id"]].drop_duplicates().shape[0]),
                "rows": int(len(local)),
                **{key: float(value) for key, value in payload.items()},
            }
        )
    return pd.DataFrame(rows)


def build_stratified_result_tables(
    predictions: pd.DataFrame,
    fold_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    linked = attach_evaluation_strata(predictions, fold_manifest)
    pooled = stratified_metric_table(
        linked,
        group_columns=("model_name", "evaluation_stratum"),
    )
    by_lead = stratified_metric_table(
        linked,
        group_columns=("model_name", "evaluation_stratum", "lead"),
    )
    return linked, pooled, by_lead
