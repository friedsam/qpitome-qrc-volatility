from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_esn_control_study.py")
SPEC = importlib.util.spec_from_file_location("stage_e_esn_control_study", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _manifest() -> pd.DataFrame:
    rows = []
    sample = 0
    for split, count in (("train", 20), ("val", 8), ("test", 8)):
        for local in range(count):
            sample += 1
            level = -4.0 + 0.02 * sample
            rows.append({
                "sample_id": f"S{sample}",
                "episode_id": f"E_{split}_{local}",
                "label": local % 2,
                "lead": (1, 5, 10)[local % 3],
                "split": split,
                "level": level,
                "mean5": level + 0.01,
                "mean20": level - 0.02,
                **{f"target_x_h{h + 1}": level + 0.01 * (h + 1) for h in range(10)},
            })
    return pd.DataFrame(rows)


def test_control_study_builds_and_scores_matched_features() -> None:
    manifest = _manifest()
    rng = np.random.default_rng(7)
    sequences = rng.normal(size=(len(manifest), 40, 1))
    states = rng.normal(size=(len(manifest), 9))
    metadata = {"name": "tiny", "n": 8, "sr": 0.7, "inp": 0.2, "leak": 0.3, "seed": 1}
    train_mask = manifest["split"].eq("train").to_numpy()
    y = manifest[[f"target_x_h{h + 1}" for h in range(10)]].to_numpy(dtype=float)
    residual = y - MODULE._har_predictions(manifest, y, train_mask)

    feature_sets = MODULE.build_feature_sets(
        states,
        sequences,
        train_mask,
        residual,
        metadata,
        components=(1, 2),
    )

    assert {"full_esn", "final_input", "random_tanh", "shuffled_esn"}.issubset(feature_sets)
    assert feature_sets["full_esn"].shape == (len(manifest), 9)
    assert feature_sets["random_tanh"].shape == (len(manifest), 8)
    assert feature_sets["shuffled_esn"].shape == (len(manifest), 9)
    assert feature_sets["pca_variance_2"].shape[1] == 2
    assert feature_sets["pca_residual_2"].shape[1] == 2

    scores = MODULE.score_feature_sets(
        feature_sets,
        manifest,
        metadata,
        alphas=(1.0, 10.0),
    )

    assert len(scores) == len(feature_sets) * 2
    assert set(scores["alpha"]) == {1.0, 10.0}
    assert np.isfinite(scores[["train_qlike", "val_qlike", "val_rmse"]]).all().all()
