from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from transition_forecasting.modeling.classical_benchmarks.common import REQUIRED_GROUPS
from transition_forecasting.modeling.classical_benchmarks.validation import (
    MODELS,
    validate_classical_run,
)


def test_submission_model_set_is_only_agreed_classical_models():
    assert MODELS == (
        "persistence",
        "har",
        "sequence_ridge",
        "garch_1_1_t",
        "esn_direct_tuned",
        "esn_shuffled_tuned",
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_valid_result_family(root: Path, run_id: str) -> dict[str, Path]:
    runs = {
        name: root / name / "run" / run_id
        for name in ("linear", "garch", "esn", "canonical")
    }
    for path in runs.values():
        path.mkdir(parents=True)

    common_required = {
        "linear": [
            "predictions.csv.gz",
            "submission_metrics.csv",
            "dataset_manifest.json",
        ],
        "garch": [
            "predictions.csv.gz",
            "submission_metrics.csv",
            "fit_diagnostics.csv.gz",
            "dataset_manifest.json",
        ],
        "esn": [
            "predictions.csv.gz",
            "submission_metrics.csv",
            "dataset_manifest.json",
        ],
    }
    for topic, names in common_required.items():
        for name in names:
            (runs[topic] / name).write_text("fixture\n", encoding="utf-8")

    _write_json(
        runs["linear"] / "config.json",
        {"sequence_alpha": 3000.0, "har_alpha": 100.0},
    )
    _write_json(
        runs["garch"] / "config.json",
        {
            "garch_config": {
                "p": 1,
                "q": 1,
                "mean": "Zero",
                "distribution": "StudentsT",
                "return_scale": 100.0,
                "backend": "arch",
            },
            "history": 2500,
            "minimum_history": 250,
        },
    )
    _write_json(
        runs["esn"] / "config.json",
        {"benchmark": "direct_esn_frozen"},
    )
    _write_json(
        runs["esn"] / "selected_spec.json",
        {
            "selected_config": {
                "config_id": "short_leak095",
                "n": 300,
                "connectivity": 0.02,
                "sr": 0.55,
                "inp": 0.2,
                "leak": 0.95,
            },
            "selected_alpha": 120000.0,
            "final_seeds": [1, 2, 3, 4, 5],
        },
    )
    for topic in ("linear", "garch", "esn", "canonical"):
        _write_json(
            runs[topic] / "summary.json",
            {"run_id": run_id, "test_evaluated": False},
        )

    rows = []
    for fold in (4, 5, 6, 7, 8):
        for model in MODELS:
            row = {
                "model": model,
                "fold": fold,
                "sample_id": f"sample-{fold}",
                "fold_split": "val",
            }
            for horizon in range(1, 11):
                row[f"actual_h{horizon}"] = float(horizon)
                row[f"predicted_h{horizon}"] = float(horizon)
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        runs["canonical"] / "common_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    metric_rows = [
        {"model": model, "group": group}
        for model in MODELS
        for group in REQUIRED_GROUPS
    ]
    pd.DataFrame(metric_rows).to_csv(
        runs["canonical"] / "submission_metrics.csv",
        index=False,
    )
    for name in (
        "submission_table_confirmation.csv",
        "coverage.csv",
        "paired_deltas_vs_sequence_ridge.csv",
    ):
        (runs["canonical"] / name).write_text("fixture\n", encoding="utf-8")
    return runs


def test_validator_accepts_frozen_family_and_rejects_parameter_drift(
    tmp_path: Path,
) -> None:
    run_id = "fixture-run"
    root = tmp_path / "classical_baselines"
    runs = _build_valid_result_family(root, run_id)
    report_path = tmp_path / "classical_audit.json"
    report = validate_classical_run(
        classical_root=root,
        run_id=run_id,
        report_path=report_path,
    )
    assert report["passed"] is True
    assert report["resolved_parameters"]["garch"]["backend"] == "arch"

    selected = json.loads(
        (runs["esn"] / "selected_spec.json").read_text(encoding="utf-8")
    )
    selected["final_seeds"] = [1, 2, 3]
    _write_json(runs["esn"] / "selected_spec.json", selected)
    with pytest.raises(RuntimeError, match="seed set mismatch"):
        validate_classical_run(
            classical_root=root,
            run_id=run_id,
            report_path=report_path,
        )
