from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.modeling import cross_market_parameter_screen as MODULE


def test_feature_generation_and_summary_are_finite() -> None:
    rng = np.random.default_rng(5)
    sequences = rng.normal(size=(18, 40, 9))
    config = {"id": "tiny", "n": 12, "conn": 0.2, "sr": 0.7, "inp": 0.2, "leak": 0.5}
    features = MODULE._features(sequences, config, seed=3)

    assert features.shape == (18, 36)
    assert np.isfinite(features).all()

    rows = []
    for fold in (1, 2):
        for seed in (1, 2):
            rows.append({
                "config_id": "tiny",
                "order": "ordered",
                "fold": fold,
                "seed": seed,
                "alpha": 100.0,
                "val_qlike": 0.8 + 0.01 * fold,
                "val_rmse": 0.5,
                "mz_alpha": 0.0,
                "mz_beta": 1.0,
                "mz_r2": 0.2,
                "n": 12,
                "conn": 0.2,
                "sr": 0.7,
                "inp": 0.2,
                "leak": 0.5,
            })
    best, leaderboard = MODULE.summarize(pd.DataFrame(rows))

    assert len(best) == 4
    assert len(leaderboard) == 1
    assert np.isfinite(leaderboard.select_dtypes(include=[float, int])).all().all()
