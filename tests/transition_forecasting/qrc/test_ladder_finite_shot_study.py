from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.ladder_finite_shot_study import (
    LadderFiniteShotStudyConfig,
    protocol_feature_matrix,
    summarize_shot_metrics,
)


def test_study_defaults_cover_five_shot_levels_and_five_seeds() -> None:
    config = LadderFiniteShotStudyConfig()
    config.validate()

    assert config.folds == tuple(range(1, 9))
    assert config.shot_counts == (250, 500, 1000, 2000, 5000)
    assert len(config.shot_seeds) == 5
    assert config.protocols == (
        "shot_consistent",
        "exact_train_shot_val",
    )


def test_protocol_feature_matrix_uses_exact_training_only_for_diagnostic() -> None:
    exact = np.arange(18, dtype=float).reshape(3, 6)
    sampled = exact + 100.0
    train = np.asarray([True, True, False])

    primary = protocol_feature_matrix(
        exact_modes=exact,
        sampled_modes=sampled,
        train_mask=train,
        protocol="shot_consistent",
    )
    diagnostic = protocol_feature_matrix(
        exact_modes=exact,
        sampled_modes=sampled,
        train_mask=train,
        protocol="exact_train_shot_val",
    )

    assert np.array_equal(primary, sampled)
    assert np.array_equal(diagnostic[train], exact[train])
    assert np.array_equal(diagnostic[~train], sampled[~train])

    with pytest.raises(ValueError, match="unsupported"):
        protocol_feature_matrix(
            exact_modes=exact,
            sampled_modes=sampled,
            train_mask=train,
            protocol="unknown",  # type: ignore[arg-type]
        )


def test_shot_summary_aggregates_across_replicate_seeds() -> None:
    frame = pd.DataFrame(
        {
            "model_name": ["m", "m", "m"],
            "protocol": ["shot_consistent"] * 3,
            "shots": [250] * 3,
            "shot_seed": [1, 2, 3],
            "qlike": [0.8, 0.7, 0.9],
            "rmse": [0.5, 0.4, 0.6],
            "prediction_pearson_exact": [0.9, 0.8, 1.0],
            "prediction_spearman_exact": [0.85, 0.75, 0.95],
        }
    )
    summary = summarize_shot_metrics(frame)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["seeds"] == 3
    assert np.isclose(row["qlike_median"], 0.8)
    assert np.isclose(row["qlike_minimum"], 0.7)
    assert np.isclose(row["qlike_maximum"], 0.9)
