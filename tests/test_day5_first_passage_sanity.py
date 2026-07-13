"""Tests for the minimal Day-5 first-passage sanity assay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5.protocol import D1, eligible_rows, load_frame
from qpitome_qrc.evaluation.binary import logistic_pipeline

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_first_passage_sanity.py"
SPEC = importlib.util.spec_from_file_location("day5_first_passage_sanity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_corridor_position_uses_distance_from_lower_barrier() -> None:
    frame = pd.DataFrame({
        "barrier_width": [1.0, 2.0],
        "distance_to_relapse_barrier": [0.25, 1.5],
    })
    result = module.corridor_position(frame)
    assert np.allclose(result, [0.25, 0.75])


def test_corridor_position_rejects_nonpositive_width() -> None:
    frame = pd.DataFrame({
        "barrier_width": [0.0],
        "distance_to_relapse_barrier": [0.0],
    })
    try:
        module.corridor_position(frame)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_metric_row_identical_predictions_are_finite() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    clusters = np.array(["a", "a", "b", "b"])
    result = module.metric_row("x", y, p, clusters)
    assert result["n"] == 4
    assert result["n_clusters"] == 2
    assert np.isfinite(result["logloss"])
    assert np.isfinite(result["brier"])
    assert np.isfinite(result["auc"])


def test_first_passage_runner_uses_package_helpers_directly() -> None:
    assert module.D1 == D1
    assert module.eligible_rows is eligible_rows
    assert module.load_frame is load_frame
    assert module.logistic_pipeline is logistic_pipeline
    assert not hasattr(module, "protected")
    assert not hasattr(module, "base")
