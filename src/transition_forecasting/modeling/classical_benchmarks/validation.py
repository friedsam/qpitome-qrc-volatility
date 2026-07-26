from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.classical_benchmarks.common import REQUIRED_GROUPS

MODELS = (
    "persistence",
    "har",
    "sequence_ridge",
    "garch_1_1_t",
    "esn_direct_tuned",
    "esn_shuffled_tuned",
)


def validate_classical_run(
    *,
    classical_root: Path,
    run_id: str,
    report_path: Path,
) -> dict[str, object]:
    root = Path(classical_root)
    runs = {
        name: root / name / "run" / run_id
        for name in ("linear", "garch", "esn", "canonical")
    }
    required = {
        "linear": [
            "predictions.csv.gz", "submission_metrics.csv", "config.json",
            "dataset_manifest.json", "summary.json",
        ],
        "garch": [
            "predictions.csv.gz", "submission_metrics.csv", "fit_diagnostics.csv.gz",
            "config.json", "dataset_manifest.json", "summary.json",
        ],
        "esn": [
            "predictions.csv.gz", "submission_metrics.csv", "selected_spec.json",
            "config.json", "dataset_manifest.json", "summary.json",
        ],
        "canonical": [
            "common_predictions.csv.gz", "submission_metrics.csv",
            "submission_table_confirmation.csv", "coverage.csv",
            "paired_deltas_vs_sequence_ridge.csv", "summary.json",
        ],
    }
    failures: list[str] = []
    inventory: dict[str, dict[str, bool]] = {}
    for topic, path in runs.items():
        inventory[topic] = {
            name: (path / name).is_file()
            for name in required[topic]
        }
        failures.extend(
            f"missing {topic}/{name}"
            for name, exists in inventory[topic].items()
            if not exists
        )
    if not failures:
        predictions = pd.read_csv(runs["canonical"] / "common_predictions.csv.gz")
        metrics = pd.read_csv(runs["canonical"] / "submission_metrics.csv")
        if set(predictions["model"]) != set(MODELS):
            failures.append("canonical model set mismatch")
        if set(predictions["fold_split"]) != {"val"}:
            failures.append("non-validation prediction rows present")
        if set(predictions["fold"].astype(int)) != {4, 5, 6, 7, 8}:
            failures.append("expected folds 4-8 only")
        if set(metrics["group"]) != set(REQUIRED_GROUPS):
            failures.append("reporting group mismatch")
        if metrics["group"].astype(str).str.contains(
            r"Control.*L|L.*Control",
            regex=True,
        ).any():
            failures.append("control-by-lead metric row present")
        actual_columns = [f"actual_h{h}" for h in range(1, 11)]
        if not np.isfinite(predictions[actual_columns].to_numpy(dtype=float)).all():
            failures.append("nonfinite actual targets")
        summaries = [
            json.loads((runs[name] / "summary.json").read_text(encoding="utf-8"))
            for name in ("linear", "garch", "esn", "canonical")
        ]
        if any(summary.get("test_evaluated") is not False for summary in summaries):
            failures.append("test_evaluated is not false")
        esn_spec = json.loads(
            (runs["esn"] / "selected_spec.json").read_text(encoding="utf-8")
        )
        if float(esn_spec["selected_alpha"]) != 120000.0:
            failures.append("frozen ESN alpha mismatch")
        garch_config = json.loads(
            (runs["garch"] / "config.json").read_text(encoding="utf-8")
        )
        if garch_config["garch_config"]["backend"] != "arch":
            failures.append("submission GARCH backend is not arch")
    report = {
        "schema_version": 1,
        "passed": not failures,
        "run_id": run_id,
        "classical_root": str(root),
        "inventory": inventory,
        "failures": failures,
        "models": list(MODELS),
        "reporting_groups": list(REQUIRED_GROUPS),
        "test_evaluated": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("classical validation failed: " + "; ".join(failures))
    return report
