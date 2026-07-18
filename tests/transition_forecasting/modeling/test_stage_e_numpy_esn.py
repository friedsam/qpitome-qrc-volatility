from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.numpy_esn import fit_continuous_ridge_scores
from transition_forecasting.modeling.stage_e_classical_baselines import StageEData

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_numpy_esn.py")
SPEC = importlib.util.spec_from_file_location("stage_e_numpy_esn", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _data() -> StageEData:
    rng = np.random.default_rng(5)
    rows = []
    sequences = []
    sample = 0
    for split, count in (("train", 12), ("val", 6), ("test", 6)):
        for local in range(count):
            sample += 1
            level = -4.0 + 0.02 * sample
            sequence = level + np.linspace(-0.1, 0.0, 40) + rng.normal(0.0, 0.005, 40)
            target = level + np.linspace(0.01, 0.10, 10)
            rows.append(
                {
                    "sample_id": f"S{sample}",
                    "episode_id": f"E_{split}_{local}",
                    "label": local % 2,
                    "lead": (1, 5, 10)[local % 3],
                    "split": split,
                    "level": level,
                    "mean5": float(sequence[-5:].mean()),
                    "mean20": float(sequence[-20:].mean()),
                    **{f"target_x_h{h + 1}": float(target[h]) for h in range(10)},
                }
            )
            sequences.append(sequence[:, None])
    return StageEData(pd.DataFrame(rows), np.asarray(sequences))


def test_continuous_readout_does_not_log_targets() -> None:
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = np.column_stack([x[:, 0], x[:, 1]])
    result = fit_continuous_ridge_scores(x, y, {"train": x}, alpha=1e-8)["train"]
    assert np.allclose(result, y, atol=1e-5)


def test_stage_e_numpy_esn_runs_direct_and_har_residual(monkeypatch) -> None:
    monkeypatch.setattr(
        MODULE,
        "historical_numpy_esn_grid",
        lambda seeds: [
            {
                "n": 8,
                "sr": 0.7,
                "inp": 0.2,
                "leak": 0.3,
                "alpha": 1.0,
                "seed": seeds[0],
                "config_id": "tiny",
            }
        ],
    )
    metrics, predictions, tuning, summary = MODULE.run_stage_e_numpy_esn(
        _data(), seeds=[1], har_alpha=1.0
    )
    assert set(predictions["model"]) == {
        "numpy_esn_direct",
        "numpy_esn_har_residual",
    }
    assert set(predictions["split"]) == {"val"}
    assert set(tuning["formulation"]) == {"direct", "har_residual"}
    assert summary["test_evaluated"] is False
    pooled = metrics[(metrics["group_type"] == "pooled") & (metrics["horizon"] == "path")]
    assert len(pooled) == 2
