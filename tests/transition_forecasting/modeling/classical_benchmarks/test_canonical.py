from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from transition_forecasting.modeling.classical_benchmarks.canonical import MODEL_ORDER, run_canonical_comparison

_RESULTS_TEXT = os.environ.get("QPITOME_CLASSICAL_RESULTS_ROOT")
RESULTS_ROOT = Path(_RESULTS_TEXT) if _RESULTS_TEXT else None
_REQUIRED = (
    "linear/linear_current_data_001/predictions.csv.gz",
    "garch/garch_current_data_001/predictions.csv.gz",
    "esn/esn_current_data_001/predictions.csv.gz",
)
pytestmark = pytest.mark.skipif(
    RESULTS_ROOT is None or not all((RESULTS_ROOT / path).exists() for path in _REQUIRED),
    reason="set QPITOME_CLASSICAL_RESULTS_ROOT for the current-run integration test",
)


def test_canonical_current_runs(tmp_path: Path) -> None:
    assert RESULTS_ROOT is not None
    run_dir, summary = run_canonical_comparison(
        linear_run=RESULTS_ROOT / "linear/linear_current_data_001",
        garch_run=RESULTS_ROOT / "garch/garch_current_data_001",
        esn_run=RESULTS_ROOT / "esn/esn_current_data_001",
        results_root=tmp_path / "canonical",
        run_id="test_run",
    )
    assert summary["test_evaluated"] is False
    assert summary["common_rows"] == 1760
    predictions = pd.read_csv(run_dir / "common_predictions.csv.gz")
    assert set(predictions["model"]) == set(MODEL_ORDER)
    assert predictions.groupby("model").size().nunique() == 1
    metrics = pd.read_csv(run_dir / "submission_metrics.csv")
    assert set(metrics["group"]) == {"Transition", "L1", "L5", "L10", "Controls", "Pooled"}
