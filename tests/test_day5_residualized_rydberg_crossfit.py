"""Tests for historical cross-fitted D1 offsets."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5.protocol import D1
from qpitome_qrc.evaluation.binary import fit_train_test_probability_arrays

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_residualized_rydberg_crossfit_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_residualized_crossfit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_historical_crossfit_uses_only_prior_rows() -> None:
    rows = []
    for cluster in range(8):
        cluster_start = pd.Timestamp("2000-01-01") + pd.Timedelta(days=cluster)
        for j in range(6):
            x = cluster + j / 10
            rows.append({
                "cluster_start": cluster_start,
                "landmark_date": cluster_start,
                "y_recovery": (cluster + j) % 2,
                D1[0]: x,
                D1[1]: x + 1,
                D1[2]: x + 2,
                D1[3]: x + 3,
            })
    frame = pd.DataFrame(rows)
    positions, logits = module.historical_crossfit_d1_logits(frame)
    assert len(positions) == len(logits)
    assert len(positions) > 0
    assert np.all(np.isfinite(logits))
    assert positions.min() >= 12
    assert len(positions) >= module.MIN_CORRECTION_TRAIN


def test_crossfit_shard_uses_grouped_probability_helper() -> None:
    assert module.fit_train_test_probability_arrays is fit_train_test_probability_arrays
