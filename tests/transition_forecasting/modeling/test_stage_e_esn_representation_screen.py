from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.esn_representation import pool_trajectory, reservoir_trajectory, transform_input

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_esn_representation_screen.py")
SPEC = importlib.util.spec_from_file_location("stage_e_esn_representation_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_input_transforms_and_pooling_shapes() -> None:
    X = np.arange(2 * 6, dtype=float).reshape(2, 6, 1)
    assert transform_input(X, "level").shape == (2, 6, 1)
    assert transform_input(X, "level_diff").shape == (2, 6, 2)
    assert transform_input(X, "level_diff_time").shape == (2, 6, 3)

    W_in = np.ones((4, 1)) * 0.1
    W = np.eye(4) * 0.2
    states = reservoir_trajectory(X, W_in, W, leak=0.3)
    assert states.shape == (2, 6, 4)
    assert pool_trajectory(states, "final", washout=0).shape == (2, 4)
    assert pool_trajectory(states, "mean", washout=1).shape == (2, 4)
    assert pool_trajectory(states, "std", washout=1).shape == (2, 4)
    assert pool_trajectory(states, "final_mean_std", washout=1).shape == (2, 12)


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    sequences = []
    rng = np.random.default_rng(7)
    for sample in range(30):
        level = -4.0 + 0.01 * sample
        sequence = level + np.linspace(-0.05, 0.0, 40) + rng.normal(0.0, 0.002, 40)
        rows.append({
            "sample_id": f"S{sample}",
            "episode_id": f"E{sample // 3}",
            "fold_split": "train" if sample < 20 else "val",
            "level": level,
            "mean5": float(sequence[-5:].mean()),
            "mean20": float(sequence[-20:].mean()),
            **{f"target_x_h{h + 1}": level + 0.01 * (h + 1) for h in range(10)},
        })
        sequences.append(sequence[:, None])
    return pd.DataFrame(rows), np.asarray(sequences)


def test_screen_scores_ordered_and_shuffled_without_test_metrics() -> None:
    manifest, sequences = _data()
    results = MODULE.evaluate_fold(
        manifest,
        sequences,
        fold=1,
        seeds=(1,),
        alphas=(10.0,),
        washouts=(0, 5),
        representations=("level", "level_diff"),
        poolings=("final", "mean"),
        n_reservoir=8,
    )
    assert set(results["order"]) == {"ordered", "shuffled"}
    assert set(results["representation"]) == {"level", "level_diff"}
    assert set(results["pooling"]) == {"final", "mean"}
    assert set(results["washout"]) == {0, 5}
    assert "test_qlike" not in results.columns
    assert np.isfinite(results[["val_qlike", "val_rmse"]]).all().all()
