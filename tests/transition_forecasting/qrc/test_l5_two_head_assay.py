from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.qrc.l5_two_head_assay import (
    L5TwoHeadAssayConfig,
    event_score_table,
    fit_l5_two_head,
    qlike_loss,
    rmse_loss,
)


def test_config_is_bounded_to_l5_and_three_seeds() -> None:
    config = L5TwoHeadAssayConfig()
    config.validate()
    assert config.target_lead == 5
    assert config.selection_seeds == (20260721, 20260722, 20260723)
    assert config.split_horizon == 4


def test_losses_are_zero_for_exact_forecast() -> None:
    y = np.array([[1.0, 2.0], [2.0, 3.0]])
    assert rmse_loss(y, y) == 0.0
    assert qlike_loss(y, y) == 0.0


def test_event_score_detects_negative_to_positive_pattern() -> None:
    horizons = np.arange(1, 11)
    correction = np.array([-0.2] * 4 + [0.3] * 6)
    frame = pd.DataFrame(
        {
            "selection_seed": 20260722,
            "fold": 4,
            "sample_id": "sample",
            "episode_id": "episode",
            "label": 1,
            "model_name": "l5_transition_two_head",
            "horizon": horizons,
            "har_pred": np.zeros(10),
            "y_pred": correction,
            "y_true": correction,
        }
    )
    scores = event_score_table(frame, split_horizon=4)
    assert len(scores) == 1
    assert bool(scores.loc[0, "desired_sign_pattern"])
    assert scores.loc[0, "early_correction"] < 0.0
    assert scores.loc[0, "late_correction"] > 0.0
    assert scores.loc[0, "expansion"] > 0.0


def test_two_head_fit_returns_complete_signed_correction() -> None:
    rng = np.random.default_rng(3)
    rows = 40
    matrix = rng.normal(size=(rows, 9))
    residuals = np.zeros((rows, 10), dtype=float)
    residuals[:, :4] = -0.2 * matrix[:, [0]]
    residuals[:, 4:] = 0.3 * matrix[:, [1]]
    har = np.zeros_like(residuals)
    y = har + residuals
    specialist_train = np.zeros(rows, dtype=bool)
    specialist_train[:30] = True
    origin_date = np.asarray([f"2020-01-{value + 1:02d}" for value in range(rows)])
    config = L5TwoHeadAssayConfig(
        alpha_grid=(1.0, 10.0),
        minimum_specialist_rows=4,
    )

    correction, diagnostics, candidates = fit_l5_two_head(
        matrix,
        residuals=residuals,
        har=har,
        y=y,
        specialist_train=specialist_train,
        origin_date=origin_date,
        config=config,
    )

    assert correction.shape == residuals.shape
    assert np.isfinite(correction).all()
    assert diagnostics["alpha_early"] in config.alpha_grid
    assert diagnostics["alpha_late"] in config.alpha_grid
    assert set(candidates["head"]) == {"early", "late"}
