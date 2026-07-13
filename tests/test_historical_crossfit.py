from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5.protocol import D1
from qpitome_qrc.evaluation.historical_crossfit import historical_crossfit_d1_logits


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_frame() -> pd.DataFrame:
    rows = []
    for i in range(12):
        cluster_start = pd.Timestamp("2000-01-01") + pd.DateOffset(years=i // 3)
        landmark_date = cluster_start - pd.Timedelta(days=30)
        row = {
            "cluster_start": cluster_start,
            "landmark_date": landmark_date,
            "y_recovery": i % 2,
        }
        for j, column in enumerate(D1):
            row[column] = float(i + j) / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_historical_crossfit_skips_groups_without_prior_history() -> None:
    frame = make_frame()

    positions, logits = historical_crossfit_d1_logits(frame, min_train=3)

    assert np.array_equal(positions, np.arange(3, len(frame)))
    assert logits.shape == (len(frame) - 3,)
    assert np.isfinite(logits).all()


def test_crossfit_shard_reexports_package_helper() -> None:
    module = load_script(
        "day5_residualized_crossfit_shard",
        "scripts/modeling/run_day5_residualized_rydberg_crossfit_shard.py",
    )

    assert module.historical_crossfit_d1_logits is historical_crossfit_d1_logits
