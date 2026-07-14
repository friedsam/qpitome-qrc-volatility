from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.evaluation.scoring import binary_summary, proper_score_deltas


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proper_score_deltas_matches_manual_row_and_cluster_means() -> None:
    frame = pd.DataFrame({
        "y": [0, 1, 1],
        "D1": [0.20, 0.60, 0.70],
        "candidate": [0.10, 0.80, 0.60],
        "cluster_id": ["a", "a", "b"],
    })

    result = proper_score_deltas(frame, "candidate")

    y = frame["y"].to_numpy(int)
    p0 = frame["D1"].to_numpy(float)
    p1 = frame["candidate"].to_numpy(float)
    dll = (
        -(y * np.log(p1) + (1 - y) * np.log(1 - p1))
        + (y * np.log(p0) + (1 - y) * np.log(1 - p0))
    )
    dbr = (p1 - y) ** 2 - (p0 - y) ** 2

    assert result["model"] == "candidate"
    assert result["n"] == 3
    assert result["n_clusters"] == 2
    np.testing.assert_allclose(result["delta_logloss"], dll.mean())
    np.testing.assert_allclose(result["delta_brier"], dbr.mean())
    np.testing.assert_allclose(
        result["cluster_mean_delta_logloss"],
        np.mean([dll[:2].mean(), dll[2]]),
    )
    np.testing.assert_allclose(
        result["cluster_mean_delta_brier"],
        np.mean([dbr[:2].mean(), dbr[2]]),
    )


def test_input_audit_reexports_proper_score_helper() -> None:
    module = load_script(
        "day5_input_audit_scoring",
        "scripts/modeling/day5_branching/falsification/run_day5_input_audit.py",
    )

    assert module.proper_score_deltas is proper_score_deltas


def test_residualized_merge_uses_shared_scoring_with_historical_schema() -> None:
    module = load_script(
        "day5_residualized_merge_scoring",
        "scripts/modeling/day5_branching/residual_confirmation/merge_day5_residualized_rydberg.py",
    )
    frame = pd.DataFrame({
        "y": [0, 1, 1, 0],
        "D1": [0.2, 0.6, 0.7, 0.4],
        "candidate": [0.1, 0.8, 0.6, 0.3],
        "cluster_id": ["a", "a", "b", "b"],
    })

    expected_summary = binary_summary(frame, "candidate", clip=1e-6)
    actual_summary = module.metrics(frame, "candidate")
    assert actual_summary == {
        key: expected_summary[key]
        for key in ("model", "n", "auc", "pr_auc", "logloss", "brier")
    }
    assert "recovery_rate" not in actual_summary

    expected_delta = proper_score_deltas(
        frame,
        "candidate",
        baseline="D1",
        clip=1e-6,
    )
    assert module.deltas(frame, "candidate") == expected_delta
    assert module.binary_summary is binary_summary
    assert module.proper_score_deltas is proper_score_deltas


def test_crossfit_merge_uses_shared_scoring_with_historical_schemas() -> None:
    module = load_script(
        "day5_crossfit_merge_scoring",
        "scripts/modeling/day5_branching/residual_confirmation/merge_day5_residualized_rydberg_crossfit.py",
    )
    frame = pd.DataFrame({
        "y": [0, 1, 1, 0],
        "D1": [0.2, 0.6, 0.7, 0.4],
        "crossfit_resid_occupations": [0.1, 0.8, 0.6, 0.3],
        "cluster_id": ["a", "a", "b", "b"],
    })

    expected_summary = binary_summary(frame, "D1", clip=1e-6)
    actual_summary = module.score(frame, "D1")
    assert actual_summary == {
        key: expected_summary[key]
        for key in ("model", "n", "auc", "pr_auc", "logloss", "brier")
    }
    assert "recovery_rate" not in actual_summary

    expected_delta = proper_score_deltas(
        frame,
        "crossfit_resid_occupations",
        baseline="D1",
        clip=1e-6,
    )
    actual_delta = module.paired(frame)
    assert actual_delta == {
        key: expected_delta[key]
        for key in (
            "n",
            "n_clusters",
            "delta_logloss",
            "delta_brier",
            "cluster_mean_delta_logloss",
            "cluster_mean_delta_brier",
        )
    }
    assert "model" not in actual_delta
    assert module.binary_summary is binary_summary
    assert module.proper_score_deltas is proper_score_deltas
