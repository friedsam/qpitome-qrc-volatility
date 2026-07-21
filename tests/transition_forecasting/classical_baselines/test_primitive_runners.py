from __future__ import annotations

from argparse import Namespace

import pandas as pd

from transition_forecasting.classical_baselines import field_confirmation
from transition_forecasting.classical_baselines import frozen_fold_baselines


def test_field_confirmation_writes_separate_outputs(tmp_path, monkeypatch):
    metrics = pd.DataFrame(
        [
            {
                "dataset": "broad_chronological_field_reproduction",
                "model": "har",
                "fold": "field",
                "group_type": "pooled",
                "group_value": "all",
                "horizon": "path",
                "n": 1,
                "rmse": 0.1,
                "qlike": 0.2,
            }
        ]
    )
    predictions = pd.DataFrame([{"model": "har", "actual_h1": 0.0, "predicted_h1": 0.0}])
    monkeypatch.setattr(
        field_confirmation,
        "run_field_reproduction",
        lambda panel, alpha, stride: (
            metrics,
            predictions,
            {"train": 1, "validation": 1, "reserved_test": 1, "test_predictions_written": 0},
        ),
    )
    monkeypatch.setattr(
        field_confirmation,
        "compare_with_reference",
        lambda observed, reference: {"all_checks_passed": True, "checks": []},
    )

    out_dir = tmp_path / "field_run"
    summary = field_confirmation.run(
        Namespace(
            panel=tmp_path / "panel.csv",
            out_dir=out_dir,
            reference=tmp_path / "reference.json",
            alpha=100.0,
            stride=1,
        )
    )

    assert summary["experiment"] == "primitive_field_confirmation"
    assert summary["test_evaluated"] is False
    assert summary["reproduction_passed"] is True
    assert (out_dir / "metrics.csv").is_file()
    assert (out_dir / "predictions.csv").is_file()
    assert (out_dir / "field_experiment_comparison.json").is_file()
    assert (out_dir / "summary.json").is_file()


def test_frozen_fold_directional_failure_is_reported_not_fatal(tmp_path, monkeypatch):
    metrics = pd.DataFrame(
        [
            {
                "dataset": "frozen_transition_folds",
                "model": "har",
                "fold": 1,
                "group_type": "pooled",
                "group_value": "all",
                "horizon": "path",
                "n": 1,
                "rmse": 0.1,
                "qlike": 0.2,
            },
            {
                "dataset": "frozen_transition_folds",
                "model": "har",
                "fold": 1,
                "group_type": "label",
                "group_value": "transition",
                "horizon": 1,
                "n": 1,
                "rmse": 0.1,
                "qlike": 0.2,
            },
        ]
    )
    predictions = pd.DataFrame([{"model": "har", "actual_h1": 0.0, "predicted_h1": 0.0}])
    monkeypatch.setattr(
        frozen_fold_baselines,
        "run_frozen_fold_baselines",
        lambda fold_dir, alpha, shuffled_seed: (
            metrics,
            predictions,
            {"fold_counts": [], "aggregate_path_metrics": []},
        ),
    )
    monkeypatch.setattr(
        frozen_fold_baselines,
        "frozen_sanity",
        lambda observed: {
            "all_checks_passed": False,
            "checks": [{"check": "ordered_beats_shuffled", "passed": False}],
            "interpretation": "scientific result",
        },
    )

    out_dir = tmp_path / "fold_run"
    args = Namespace(
        fold_dir=tmp_path / "folds",
        out_dir=out_dir,
        alpha=100.0,
        shuffled_seed=7,
    )
    summary = frozen_fold_baselines.run(args)

    assert summary["experiment"] == "primitive_frozen_fold_baselines"
    assert summary["test_evaluated"] is False
    assert summary["directional_checks"]["all_checks_passed"] is False
    assert (out_dir / "metrics_by_fold.csv").is_file()
    assert (out_dir / "metrics_by_horizon.csv").is_file()
    assert (out_dir / "metrics_by_group.csv").is_file()
    assert (out_dir / "predictions.csv").is_file()
    assert (out_dir / "directional_checks.json").is_file()
    assert (out_dir / "summary.json").is_file()
