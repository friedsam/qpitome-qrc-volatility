from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.forecast_warning_tools import (
    FrozenPathWarningConfig,
    build_path_score_frame,
    paired_cluster_bootstrap_ap,
    warning_score_metrics,
)


def _prediction_rows() -> pd.DataFrame:
    rows = []
    model_names = (
        "har",
        "chain_interaction_off",
        "chain_interacting_1p00",
        "frozen_ladder_symmetric_1p25",
    )
    for fold in (1, 2):
        source = "development" if fold == 1 else "confirmation"
        for episode in range(6):
            for label in (0, 1):
                sample_id = f"fold{fold}_episode{episode}_label{label}"
                har_path = -4.0 + 0.02 * np.arange(10)
                true_path = har_path + label * 0.08 * np.arange(10)
                for model in model_names:
                    model_signal = {
                        "har": 0.0,
                        "chain_interaction_off": 0.01,
                        "chain_interacting_1p00": 0.03,
                        "frozen_ladder_symmetric_1p25": 0.05,
                    }[model]
                    prediction = har_path + label * model_signal * np.arange(10)
                    for horizon in range(1, 11):
                        rows.append(
                            {
                                "fold": fold,
                                "sample_id": sample_id,
                                "lead": 5,
                                "label": label,
                                "episode_id": f"fold{fold}_episode{episode}",
                                "origin_date": f"200{fold}-01-01",
                                "model": model,
                                "source": source,
                                "horizon": horizon,
                                "y_true": true_path[horizon - 1],
                                "y_pred": prediction[horizon - 1],
                                "har_pred": har_path[horizon - 1],
                            }
                        )
    return pd.DataFrame(rows)


def test_config_rejects_onset_before_baseline() -> None:
    config = FrozenPathWarningConfig(
        baseline_horizon=5,
        onset_horizon=1,
        development_folds=(1,),
        confirmation_folds=(2,),
    )
    try:
        config.validate()
    except ValueError as exc:
        assert "must follow" in str(exc)
    else:
        raise AssertionError("invalid onset protocol was accepted")


def test_path_scores_encode_forecast_rise_and_correction() -> None:
    config = FrozenPathWarningConfig(
        development_folds=(1,),
        confirmation_folds=(2,),
        bootstrap_replicates=50,
    )
    scores = build_path_score_frame(_prediction_rows(), config)
    assert len(scores) == 2 * 6 * 2 * 4
    selected = scores.loc[
        scores["model"].eq("frozen_ladder_symmetric_1p25")
        & scores["label"].eq(1)
    ]
    assert np.allclose(selected["onset_rise"], 4 * (0.02 + 0.05))
    assert np.allclose(selected["qrc_mean_correction"], 0.05 * 4.5)
    controls = scores.loc[
        scores["model"].eq("frozen_ladder_symmetric_1p25")
        & scores["label"].eq(0)
    ]
    assert np.allclose(controls["qrc_mean_correction"], 0.0)


def test_warning_metrics_and_paired_bootstrap() -> None:
    config = FrozenPathWarningConfig(
        development_folds=(1,),
        confirmation_folds=(2,),
        bootstrap_replicates=100,
    )
    scores = build_path_score_frame(_prediction_rows(), config)
    ladder = scores.loc[
        scores["model"].eq("frozen_ladder_symmetric_1p25")
    ]
    payload = warning_score_metrics(ladder, score_column="onset_rise")
    assert payload["average_precision"] > 0.9
    bootstrap = paired_cluster_bootstrap_ap(
        scores,
        model_a="frozen_ladder_symmetric_1p25",
        model_b="har",
        score_column="onset_rise",
        folds=(1, 2),
        replicates=100,
        seed=7,
    )
    assert bootstrap["observed_delta_ap"] > 0.0
    assert bootstrap["valid_replicates"] > 0
