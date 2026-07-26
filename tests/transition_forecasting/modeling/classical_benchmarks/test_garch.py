from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from transition_forecasting.modeling.classical_benchmarks.garch import run_garch_benchmark

_DATASET_TEXT = os.environ.get("QPITOME_CLASSICAL_DATASET_ROOT")
DATASET = Path(_DATASET_TEXT) if _DATASET_TEXT else None
pytestmark = pytest.mark.skipif(
    DATASET is None or not DATASET.exists(),
    reason="set QPITOME_CLASSICAL_DATASET_ROOT for the current-data integration test",
)


def test_garch_benchmark_writes_results(tmp_path: Path) -> None:
    assert DATASET is not None
    run_dir, summary = run_garch_benchmark(
        dataset_root=DATASET,
        results_root=tmp_path / "garch",
        run_id="test_run",
        selection_folds=(4,),
        confirmation_folds=(5,),
        history=500,
        minimum_history=100,
        backend="scipy",
        max_workers=2,
        limit_rows=12,
    )
    assert summary["test_evaluated"] is False
    predictions = pd.read_csv(run_dir / "predictions.csv.gz")
    assert set(predictions["fold_split"]) == {"val"}
    assert not predictions["sample_id"].duplicated().any()
    metrics = pd.read_csv(run_dir / "submission_metrics.csv")
    assert set(metrics["group"]).issubset({"Transition", "L1", "L5", "L10", "Controls", "Pooled"})
