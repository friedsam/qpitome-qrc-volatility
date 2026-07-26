from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from transition_forecasting.modeling.classical_benchmarks.linear import run_linear_benchmark

_DATASET_TEXT = os.environ.get("QPITOME_CLASSICAL_DATASET_ROOT")
DATASET = Path(_DATASET_TEXT) if _DATASET_TEXT else None
pytestmark = pytest.mark.skipif(
    DATASET is None or not DATASET.exists(),
    reason="set QPITOME_CLASSICAL_DATASET_ROOT for the current-data integration test",
)


def test_linear_benchmark_writes_submission_artifacts(tmp_path: Path) -> None:
    assert DATASET is not None
    run_dir, summary = run_linear_benchmark(
        dataset_root=DATASET,
        results_root=tmp_path / "linear",
        run_id="test_run",
        selection_folds=(4,),
        confirmation_folds=(5,),
        sequence_alphas=(100.0, 1000.0),
    )
    assert summary["test_evaluated"] is False
    required = {
        "params.json", "config.json", "dataset_manifest.json", "predictions.csv.gz",
        "submission_metrics.csv", "metrics_by_fold.csv", "metrics_by_horizon.csv",
        "sequence_ridge_tuning_by_fold.csv", "sequence_ridge_tuning_summary.csv",
        "runtime.json", "summary.json",
    }
    assert required.issubset(path.name for path in run_dir.iterdir())
    predictions = pd.read_csv(run_dir / "predictions.csv.gz")
    assert set(predictions["fold_split"]) == {"val"}
    assert set(predictions["fold"]) == {4, 5}
    assert set(predictions["model"]) == {"persistence", "har", "sequence_ridge"}
    metrics = pd.read_csv(run_dir / "submission_metrics.csv")
    assert set(metrics["group"]) == {"Transition", "L1", "L5", "L10", "Controls", "Pooled"}
    assert not metrics["group"].str.contains("Control.*L|L.*Control", regex=True).any()
