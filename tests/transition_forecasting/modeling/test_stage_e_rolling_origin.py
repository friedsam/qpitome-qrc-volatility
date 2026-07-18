from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path("scripts/transition_forecasting/modeling/run_stage_e_rolling_origin.py")
SPEC = importlib.util.spec_from_file_location("stage_e_rolling_origin", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    sequences = []
    rng = np.random.default_rng(5)
    sample = 0
    for episode in range(24):
        date = pd.Timestamp("2005-01-01") + pd.Timedelta(days=180 * episode)
        for local in range(2):
            sample += 1
            level = -4.2 + 0.01 * sample
            sequence = level + np.linspace(-0.04, 0.0, 40) + rng.normal(0.0, 0.003, 40)
            rows.append({
                "sample_id": f"S{sample}",
                "episode_id": f"E{episode:02d}",
                "event_onset": date,
                "market_group": f"M{episode % 4}",
                "label": local % 2,
                "lead": (1, 5)[local],
                "level": level,
                "mean5": float(sequence[-5:].mean()),
                "mean20": float(sequence[-20:].mean()),
                **{f"target_x_h{h + 1}": level + 0.01 * (h + 1) for h in range(10)},
            })
            sequences.append(sequence[:, None])
    return pd.DataFrame(rows), np.asarray(sequences)


def test_rolling_origin_folds_are_forward_only_and_keep_test_untouched() -> None:
    manifest, _ = _data()
    assignments, summaries = MODULE.rolling_origin_folds(
        manifest,
        date_column="event_onset",
        group_column="episode_id",
        n_folds=3,
        test_fraction=0.17,
        embargo_days=10,
    )

    assert len(summaries) == 3
    assert set(assignments["fold"]) == {1, 2, 3}
    for fold in (1, 2, 3):
        frame = assignments[assignments["fold"].eq(fold)]
        dates = pd.to_datetime(frame["event_onset"])
        assert dates[frame["fold_split"].eq("train")].max() < dates[frame["fold_split"].eq("val")].min()
        assert frame.groupby("episode_id")["fold_split"].nunique().max() == 1
        assert frame["fold_split"].eq("test").sum() > 0


def test_rolling_origin_fold_evaluation_scores_validation_only(monkeypatch) -> None:
    manifest, sequences = _data()
    assignments, _ = MODULE.rolling_origin_folds(
        manifest,
        date_column="event_onset",
        group_column="episode_id",
        n_folds=3,
        test_fraction=0.17,
        embargo_days=10,
    )
    fold_manifest = assignments[assignments["fold"].eq(1)].reset_index(drop=True)
    monkeypatch.setitem(MODULE.CONFIG, "n", 8)
    monkeypatch.setitem(MODULE.CONFIG, "sr", 0.7)
    monkeypatch.setitem(MODULE.CONFIG, "inp", 0.2)
    monkeypatch.setitem(MODULE.CONFIG, "leak", 0.3)

    results = MODULE.evaluate_fold(
        fold_manifest,
        sequences,
        fold=1,
        seeds=(1,),
        alphas=(1.0,),
        pca_components=3,
    )

    assert set(results["model"]) == {"har", "sequence_ridge", "pca10_esn", "shuffled_esn"}
    assert "test_qlike" not in results.columns
    assert np.isfinite(results[["val_qlike", "val_rmse"]]).all().all()
