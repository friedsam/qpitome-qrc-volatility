from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/bootstrap_stage_e_model_differences.py")
SPEC = importlib.util.spec_from_file_location("stage_e_bootstrap", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
paired_episode_bootstrap = MODULE.paired_episode_bootstrap


def _predictions() -> pd.DataFrame:
    rows = []
    for model, offset in (("har_ridge", 0.20), ("sequence_ridge", 0.10)):
        for episode, label, lead in (("E1", 1, 1), ("E2", 1, 5), ("E3", 0, 10)):
            row = {
                "model": model,
                "sample_id": episode,
                "episode_id": episode,
                "label": label,
                "lead": lead,
                "split": "val",
            }
            actual = np.linspace(-3.0, -2.1, 10)
            for h, value in enumerate(actual, start=1):
                row[f"actual_h{h}"] = value
                row[f"predicted_h{h}"] = value + offset
            rows.append(row)
    return pd.DataFrame(rows)


def test_paired_episode_bootstrap_detects_better_challenger() -> None:
    result = paired_episode_bootstrap(_predictions(), n_bootstrap=500, seed=3)
    pooled = result[(result["group_type"] == "pooled") & (result["group_value"] == "all")].iloc[0]
    assert pooled["n_episodes"] == 3
    assert pooled["mean_delta_qlike"] < 0.0
    assert pooled["probability_challenger_better"] == 1.0
