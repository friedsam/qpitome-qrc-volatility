from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.numpy_esn import make_esn_weights
from transition_forecasting.modeling import esn_parameter_screen as MODULE


def test_connectivity_is_configurable_and_deterministic() -> None:
    _, sparse_a = make_esn_weights(2, 40, 0.9, 0.3, seed=4, connectivity=0.02)
    _, sparse_b = make_esn_weights(2, 40, 0.9, 0.3, seed=4, connectivity=0.02)
    _, dense = make_esn_weights(2, 40, 0.9, 0.3, seed=4, connectivity=0.20)

    assert np.array_equal(sparse_a, sparse_b)
    assert np.count_nonzero(sparse_a) < np.count_nonzero(dense)
    assert np.isclose(np.max(np.abs(np.linalg.eigvals(sparse_a))), 0.9)


def test_refinement_profile_is_curated_and_unique() -> None:
    ids = [config["id"] for config in MODULE.REFINEMENT_CONFIGS]
    parameter_sets = {
        (config["n"], config["conn"], config["sr"], config["inp"], config["leak"])
        for config in MODULE.REFINEMENT_CONFIGS
    }

    assert 12 <= len(ids) <= 18
    assert len(ids) == len(set(ids)) == len(parameter_sets)
    assert MODULE.REFINEMENT_ALPHAS == (300.0, 1000.0, 3000.0, 10000.0)


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    sequences = []
    rng = np.random.default_rng(9)
    for fold in (1, 2):
        for sample in range(24):
            level = -4.0 + 0.01 * sample
            sequence = level + np.linspace(-0.03, 0.0, 40) + rng.normal(0.0, 0.002, 40)
            rows.append({
                "fold": fold,
                "sample_id": f"F{fold}S{sample}",
                "episode_id": f"F{fold}E{sample // 3}",
                "fold_split": "train" if sample < 16 else "val",
                "level": level,
                "mean5": float(sequence[-5:].mean()),
                "mean20": float(sequence[-20:].mean()),
                **{f"target_x_h{h + 1}": level + 0.01 * (h + 1) for h in range(10)},
            })
            sequences.append(sequence[:, None])
    return pd.DataFrame(rows), np.asarray(sequences[:24])


def test_small_config_evaluation_produces_finite_scores_and_mz() -> None:
    assignments, sequences = _data()
    config = {"id": "tiny", "n": 8, "conn": 0.2, "sr": 0.7, "inp": 0.2, "leak": 0.5}
    results = MODULE.evaluate_config(
        assignments,
        sequences,
        config,
        seeds=(1,),
        alphas=(10.0,),
    )

    assert set(results["fold"]) == {1, 2}
    assert set(results["config_id"]) == {"tiny"}
    assert "test_qlike" not in results.columns
    assert np.isfinite(
        results[["val_qlike", "val_rmse", "mz_alpha", "mz_beta", "mz_r2"]]
    ).all().all()
