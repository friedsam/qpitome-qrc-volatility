from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.evaluation.scoring import (
    binary_summary,
    cluster_weighted_summary,
    proper_score_deltas,
)


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_binary_summary_and_cluster_weighted_summary() -> None:
    frame = pd.DataFrame({
        "y": [0, 1, 1],
        "candidate": [0.20, 0.80, 0.60],
        "cluster_id": ["a", "a", "b"],
    })

    summary = binary_summary(frame, "candidate")
    cluster_summary = cluster_weighted_summary(frame, "candidate")

    y = frame["y"].to_numpy(int)
    p = frame["candidate"].to_numpy(float)
    row_logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    row_brier = (p - y) ** 2

    assert summary["model"] == "candidate"
    assert summary["n"] == 3
    np.testing.assert_allclose(summary["recovery_rate"], 2.0 / 3.0)
    np.testing.assert_allclose(summary["logloss"], row_logloss.mean())
    np.testing.assert_allclose(summary["brier"], row_brier.mean())

    assert cluster_summary["model"] == "candidate"
    assert cluster_summary["n_clusters"] == 2
    np.testing.assert_allclose(
        cluster_summary["cluster_mean_logloss"],
        np.mean([row_logloss[:2].mean(), row_logloss[2]]),
    )
    np.testing.assert_allclose(
        cluster_summary["cluster_mean_brier"],
        np.mean([row_brier[:2].mean(), row_brier[2]]),
    )
    np.testing.assert_allclose(cluster_summary["median_cluster_size"], 1.5)


def test_fixed_rydberg_baseline_reexports_scoring_helpers() -> None:
    module = load_script(
        "fixed_rydberg_metric_helpers",
        "scripts/modeling/day5_branching/static_rydberg/run_cross_market_day5_fixed_rydberg_feature.py",
    )

    assert module.score is binary_summary
    assert module.binary_summary is binary_summary
    assert module.cluster_weighted_summary is cluster_weighted_summary


def test_spatial_merge_uses_all_shared_scoring_helpers() -> None:
    module = load_script(
        "spatial_rydberg_merge_scoring",
        "scripts/modeling/day5_branching/spatial_rydberg/merge_day5_spatial_rydberg_assay.py",
    )
    frame = pd.DataFrame({
        "y": [0, 1, 1, 0],
        "D1": [0.2, 0.6, 0.7, 0.4],
        "candidate": [0.1, 0.8, 0.6, 0.3],
        "cluster_id": ["a", "a", "b", "b"],
    })

    assert module.score(frame, "candidate") == binary_summary(
        frame,
        "candidate",
        clip=1e-6,
    )
    assert module.paired_delta(frame, "candidate") == proper_score_deltas(
        frame,
        "candidate",
        baseline="D1",
        clip=1e-6,
    )
    assert module.cluster_metrics(frame, "candidate") == cluster_weighted_summary(
        frame,
        "candidate",
        clip=1e-6,
    )
    assert module.binary_summary is binary_summary
    assert module.proper_score_deltas is proper_score_deltas
    assert module.cluster_weighted_summary is cluster_weighted_summary
