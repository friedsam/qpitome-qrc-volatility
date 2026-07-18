from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.modeling.stage_e_classical_baselines import (
    StageEData,
    qlike_loss,
    run_classical_sanity_ladder,
    validate_split_integrity,
)


def _synthetic_data() -> StageEData:
    rng = np.random.default_rng(7)
    rows = []
    sequences = []
    split_specs = (("train", 12), ("val", 6), ("test", 6))
    sample_number = 0
    for split, count in split_specs:
        for local in range(count):
            sample_number += 1
            level = -4.0 + 0.03 * sample_number
            sequence = level + np.linspace(-0.2, 0.0, 40) + rng.normal(0.0, 0.01, 40)
            target = level + np.linspace(0.01, 0.10, 10)
            rows.append(
                {
                    "sample_id": f"S{sample_number}",
                    "label": local % 2,
                    "lead": (1, 5, 10)[local % 3],
                    "episode_id": f"E_{split}_{local}",
                    "split": split,
                    "level": level,
                    "mean5": float(sequence[-5:].mean()),
                    "mean20": float(sequence[-20:].mean()),
                    "slope5": 0.01,
                    "slope20": 0.005,
                    "std20": float(sequence[-20:].std(ddof=1)),
                    "max20": float(sequence[-20:].max()),
                    **{f"target_x_h{h + 1}": float(target[h]) for h in range(10)},
                }
            )
            sequences.append(sequence[:, None])
    return StageEData(pd.DataFrame(rows), np.asarray(sequences))


def test_qlike_is_zero_for_exact_forecast() -> None:
    values = np.array([[-4.0, -3.5], [-2.0, -1.0]])
    assert np.allclose(qlike_loss(values, values), 0.0)


def test_split_integrity_rejects_episode_leakage() -> None:
    manifest = _synthetic_data().manifest.copy()
    manifest.loc[manifest.index[-1], "episode_id"] = manifest.loc[0, "episode_id"]
    with pytest.raises(ValueError, match="episodes span multiple splits"):
        validate_split_integrity(manifest)


def test_classical_sanity_ladder_uses_validation_only() -> None:
    data = _synthetic_data()
    metrics, predictions, tuning, summary = run_classical_sanity_ladder(
        data,
        alphas=(0.01, 1.0),
        seed=11,
    )
    assert set(predictions["split"]) == {"val"}
    assert set(predictions["model"]) == {
        "persistence",
        "level_ridge",
        "har_ridge",
        "extended_har_ridge",
        "sequence_ridge",
        "shuffled_sequence_ridge",
    }
    assert set(tuning["alpha"]) == {0.01, 1.0}
    assert summary["selection_split"] == "val"
    assert summary["test_evaluated"] is False
    assert len(summary["validation_ranking"]) == 6
    pooled_path = metrics[
        (metrics["group_type"] == "pooled") & (metrics["horizon"] == "path")
    ]
    assert len(pooled_path) == 6
