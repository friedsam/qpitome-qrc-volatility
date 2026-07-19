from __future__ import annotations

import numpy as np
import pandas as pd

from transition_forecasting.modeling.chronological_splits import rolling_origin_assignments
from transition_forecasting.modeling.stage_e_sequence_models import evaluate_fold


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    sequences = []
    rng = np.random.default_rng(5)
    sample = 0
    for episode in range(24):
        onset = pd.Timestamp("2005-01-01") + pd.Timedelta(days=180 * episode)
        for local in range(2):
            sample += 1
            level = -4.2 + 0.01 * sample
            sequence = level + np.linspace(-0.04, 0.0, 40) + rng.normal(0.0, 0.003, 40)
            rows.append(
                {
                    "sample_id": f"S{sample}",
                    "episode_id": f"E{episode:02d}",
                    "event_onset": onset,
                    "origin_date": onset - pd.Timedelta(days=(1, 5)[local]),
                    "index": f"IDX{episode % 4}",
                    "market_group": f"M{episode % 4}",
                    "label": local % 2,
                    "lead": (1, 5)[local],
                    "level": level,
                    "mean5": float(sequence[-5:].mean()),
                    "mean20": float(sequence[-20:].mean()),
                    **{f"target_x_h{h + 1}": level + 0.01 * (h + 1) for h in range(10)},
                }
            )
            sequences.append(sequence[:, None])
    return pd.DataFrame(rows), np.asarray(sequences)


def test_rolling_origin_folds_are_forward_only_and_keep_test_untouched() -> None:
    manifest, _ = _data()
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
    assert set(assignments["fold"]) == {1, 2, 3}
    for fold in (1, 2, 3):
        frame = assignments[assignments["fold"].eq(fold)]
        train = frame[frame["fold_split"].eq("train")]
        val = frame[frame["fold_split"].eq("val")]
        assert train["interval_end"].max() < val["interval_start"].min()
        assert frame["fold_split"].eq("test").sum() > 0


def test_rolling_origin_fold_evaluation_scores_validation_only() -> None:
    manifest, sequences = _data()
    assignments, _, _ = rolling_origin_assignments(
        manifest,
        n_folds=3,
        test_fraction=0.17,
        embargo_days=10,
        input_lookback_days=40,
        target_horizon_days=15,
    )
    fold_manifest = assignments[assignments["fold"].eq(1)].reset_index(drop=True)

    results = evaluate_fold(
        fold_manifest,
        sequences,
        fold=1,
        seeds=(1,),
        alphas=(1.0,),
        pca_components=3,
        config={"name": "test", "n": 8, "sr": 0.7, "inp": 0.2, "leak": 0.3},
    )

    assert set(results["model"]) == {"har", "sequence_ridge", "pca10_esn", "shuffled_esn"}
    assert "test_qlike" not in results.columns
    assert np.isfinite(results[["val_qlike", "val_rmse"]]).all().all()
