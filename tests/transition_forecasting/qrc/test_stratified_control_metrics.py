from __future__ import annotations

import pandas as pd

from transition_forecasting.qrc.stratified_control_metrics import (
    attach_evaluation_strata,
    build_stratified_result_tables,
)


def _predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_id, lead, truth, forecast in (
        ("P1", 5, (1.0, 1.2), (1.1, 1.1)),
        ("C1", 5, (0.2, 0.3), (0.25, 0.25)),
        ("H1", 5, (0.7, 0.8), (0.75, 0.85)),
    ):
        for horizon, (observed, predicted) in enumerate(
            zip(truth, forecast, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "fold": 1,
                    "sample_id": sample_id,
                    "model_name": "har",
                    "lead": lead,
                    "horizon": horizon,
                    "y_true": observed,
                    "y_pred": predicted,
                }
            )
    return pd.DataFrame(rows)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"fold": 1, "sample_id": "P1", "evaluation_stratum": "transition"},
            {"fold": 1, "sample_id": "C1", "evaluation_stratum": "calm"},
            {"fold": 1, "sample_id": "H1", "evaluation_stratum": "hard_negative"},
        ]
    )


def test_stratum_linkage_and_metric_tables_are_complete() -> None:
    linked, pooled, by_lead = build_stratified_result_tables(
        _predictions(),
        _manifest(),
    )

    assert set(linked["evaluation_stratum"]) == {
        "transition",
        "calm",
        "hard_negative",
    }
    assert set(pooled["evaluation_stratum"]) == {
        "transition",
        "calm",
        "hard_negative",
    }
    assert pooled["samples"].eq(1).all()
    assert pooled["rows"].eq(2).all()
    assert by_lead["lead"].eq(5).all()
    assert by_lead["qlike"].notna().all()
    assert by_lead["rmse"].notna().all()


def test_missing_fold_sample_linkage_is_rejected() -> None:
    manifest = _manifest().loc[lambda frame: frame["sample_id"].ne("H1")]
    try:
        attach_evaluation_strata(_predictions(), manifest)
    except ValueError as exc:
        assert "lack frozen stratum linkage" in str(exc)
    else:
        raise AssertionError("missing stratum linkage was accepted")
