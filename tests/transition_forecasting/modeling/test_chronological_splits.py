from __future__ import annotations

import pandas as pd

from transition_forecasting.modeling.chronological_splits import (
    IntervalColumns,
    rolling_origin_assignments,
)


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sample = 0
    for episode in range(24):
        onset = pd.Timestamp("2000-01-01") + pd.Timedelta(days=180 * episode)
        for lead in (1, 5, 10):
            sample += 1
            positive_origin = onset - pd.Timedelta(days=lead)
            rows.append(
                {
                    "sample_id": f"P{sample}",
                    "label": 1,
                    "index": "IDX",
                    "episode_id": f"E{episode:02d}",
                    "origin_date": positive_origin,
                }
            )
            for control in range(3):
                sample += 1
                rows.append(
                    {
                        "sample_id": f"C{sample}",
                        "label": 0,
                        "index": "IDX",
                        "episode_id": f"E{episode:02d}",
                        "origin_date": positive_origin
                        + pd.Timedelta(days=45 + 30 * control + 365 * (episode % 3)),
                    }
                )
    return pd.DataFrame(rows)


def test_assignments_use_control_origin_not_positive_episode_date() -> None:
    manifest = _manifest()
    assignments, summaries, audit = rolling_origin_assignments(
        manifest,
        n_folds=3,
        test_fraction=0.17,
        embargo_days=10,
        input_lookback_days=40,
        target_horizon_days=15,
    )

    assert len(summaries) == 3
    assert audit["control_groups_use_actual_origin"] is True
    controls = assignments[assignments["label"].eq(0)]
    assert controls["chronology_group"].str.startswith("C:").all()
    positives = assignments[assignments["label"].eq(1)]
    assert positives["chronology_group"].str.startswith("P:").all()

    for fold in sorted(assignments["fold"].unique()):
        frame = assignments[assignments["fold"].eq(fold)]
        train = frame[frame["fold_split"].eq("train")]
        val = frame[frame["fold_split"].eq("val")]
        assert not train.empty
        assert not val.empty
        assert train["interval_end"].max() < val["interval_start"].min()


def test_duplicate_index_origin_rows_are_removed_and_source_rows_preserved() -> None:
    manifest = _manifest()
    duplicate = manifest.iloc[[0]].copy()
    duplicate["sample_id"] = "DUPLICATE"
    manifest = pd.concat([manifest, duplicate], ignore_index=True)

    assignments, _, audit = rolling_origin_assignments(
        manifest,
        n_folds=3,
        test_fraction=0.17,
        embargo_days=10,
        input_lookback_days=40,
        target_horizon_days=15,
    )

    assert audit["duplicate_index_origin_removed"] >= 1
    assert "_source_row" in assignments.columns
    assert assignments["_source_row"].between(0, len(manifest) - 1).all()
    for _, fold in assignments.groupby("fold"):
        assert not fold.duplicated(["index", "origin_date"]).any()


def test_exact_interval_columns_replace_calendar_approximation() -> None:
    manifest = _manifest()
    manifest["input_start_date"] = pd.to_datetime(manifest["origin_date"]) - pd.offsets.BDay(39)
    manifest["target_end_date"] = pd.to_datetime(manifest["origin_date"]) + pd.offsets.BDay(10)

    assignments, _, audit = rolling_origin_assignments(
        manifest,
        n_folds=3,
        test_fraction=0.17,
        embargo_days=10,
        columns=IntervalColumns(
            input_start="input_start_date",
            target_end="target_end_date",
        ),
    )

    assert audit["interval_source"] == "exact_columns"
    first = assignments.iloc[0]
    assert pd.Timestamp(first["interval_start"]) == pd.Timestamp(first["input_start_date"], tz="UTC")
    assert pd.Timestamp(first["interval_end"]) == pd.Timestamp(first["target_end_date"], tz="UTC")
