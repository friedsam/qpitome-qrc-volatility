from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_chronological_robustness.py")
if not SCRIPT.is_file():
    pytest.skip(
        "Legacy Stage E chronological robustness script is not present on this branch",
        allow_module_level=True,
    )
SPEC = importlib.util.spec_from_file_location("stage_e_chronological_robustness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    sequences = []
    rng = np.random.default_rng(4)
    sample = 0
    for group in range(30):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=14 * group)
        for local in range(2):
            sample += 1
            level = -4.5 + 0.01 * sample
            sequence = level + np.linspace(-0.05, 0.0, 40) + rng.normal(0.0, 0.003, 40)
            rows.append({
                "sample_id": f"S{sample}",
                "global_cluster_id": f"G{group:02d}",
                "episode_id": f"E{group:02d}_{local}",
                "event_date": date,
                "split": "train" if group % 2 == 0 else "val",
                "level": level,
                "mean5": float(sequence[-5:].mean()),
                "mean20": float(sequence[-20:].mean()),
                **{f"target_x_h{h + 1}": level + 0.01 * (h + 1) for h in range(10)},
            })
            sequences.append(sequence[:, None])
    return pd.DataFrame(rows), np.asarray(sequences)


def test_chronological_split_is_ordered_grouped_and_purged() -> None:
    manifest, _ = _data()
    split, summary = MODULE.chronological_split(
        manifest,
        train_fraction=0.6,
        val_fraction=0.2,
        embargo_days=7,
    )
    assert summary["date_column"] == "event_date"
    assert summary["group_column"] == "global_cluster_id"
    assert {"train", "val", "test", "purged"}.issubset(set(split["split"]))
    assert split.groupby("global_cluster_id")["split"].nunique().max() == 1
    assert split.groupby("episode_id")["split"].nunique().max() == 1
    dates = pd.to_datetime(split["event_date"])
    assert dates[split["split"].eq("train")].max() < dates[split["split"].eq("val")].min()
    assert dates[split["split"].eq("val")].max() < dates[split["split"].eq("test")].min()


def test_chronological_study_never_scores_test() -> None:
    manifest, sequences = _data()
    split, _ = MODULE.chronological_split(
        manifest,
        train_fraction=0.6,
        val_fraction=0.2,
        embargo_days=7,
    )
    config = ({"name": "tiny", "n": 8, "sr": 0.7, "inp": 0.2, "leak": 0.3},)
    results = MODULE.run_study(
        split,
        sequences,
        configs=config,
        seeds=(1,),
        alphas=(1.0,),
        pca_components=3,
    )
    assert set(results["model"]) == {
        "har", "sequence_ridge", "full_esn", "pca10_esn", "shuffled_esn", "random_tanh"
    }
    assert "test_qlike" not in results.columns
    assert np.isfinite(results[["val_qlike", "val_rmse"]]).all().all()
