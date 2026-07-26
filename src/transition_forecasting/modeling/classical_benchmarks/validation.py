from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.classical_benchmarks.common import REQUIRED_GROUPS
from transition_forecasting.modeling.classical_benchmarks.spec import load_frozen_spec

MODELS = (
    "persistence",
    "har",
    "sequence_ridge",
    "garch_1_1_t",
    "esn_direct_tuned",
    "esn_shuffled_tuned",
)


def _same_number(actual: object, expected: object) -> bool:
    try:
        return bool(np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12))
    except (TypeError, ValueError):
        return False


def validate_classical_run(
    *,
    classical_root: Path,
    run_id: str,
    report_path: Path,
) -> dict[str, object]:
    root = Path(classical_root)
    frozen = load_frozen_spec()
    runs = {
        name: root / name / "run" / run_id
        for name in ("linear", "garch", "esn", "canonical")
    }
    required = {
        "linear": [
            "predictions.csv.gz",
            "submission_metrics.csv",
            "config.json",
            "dataset_manifest.json",
            "summary.json",
        ],
        "garch": [
            "predictions.csv.gz",
            "submission_metrics.csv",
            "fit_diagnostics.csv.gz",
            "config.json",
            "dataset_manifest.json",
            "summary.json",
        ],
        "esn": [
            "predictions.csv.gz",
            "submission_metrics.csv",
            "selected_spec.json",
            "config.json",
            "dataset_manifest.json",
            "summary.json",
        ],
        "canonical": [
            "common_predictions.csv.gz",
            "submission_metrics.csv",
            "submission_table_confirmation.csv",
            "coverage.csv",
            "paired_deltas_vs_sequence_ridge.csv",
            "summary.json",
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
    if any(root.rglob("*.py")):
        failures.append("source file found beneath classical result root")

    resolved_parameters: dict[str, object] = {}
    if not failures:
        predictions = pd.read_csv(runs["canonical"] / "common_predictions.csv.gz")
        metrics = pd.read_csv(runs["canonical"] / "submission_metrics.csv")
        if set(predictions["model"]) != set(MODELS):
            failures.append("canonical model set mismatch")
        if set(metrics["model"]) != set(MODELS):
            failures.append("canonical metric model set mismatch")
        if set(predictions["fold_split"]) != {"val"}:
            failures.append("non-validation prediction rows present")
        expected_folds = set(frozen["selection_folds"] + frozen["confirmation_folds"])
        if set(predictions["fold"].astype(int)) != expected_folds:
            failures.append("prediction folds differ from frozen folds")
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

        summaries = {
            name: json.loads(
                (runs[name] / "summary.json").read_text(encoding="utf-8")
            )
            for name in ("linear", "garch", "esn", "canonical")
        }
        if any(summary.get("test_evaluated") is not False for summary in summaries.values()):
            failures.append("test_evaluated is not false")
        for name, summary in summaries.items():
            if summary.get("run_id") != run_id:
                failures.append(f"{name} summary run_id mismatch")

        linear_config = json.loads(
            (runs["linear"] / "config.json").read_text(encoding="utf-8")
        )
        if not _same_number(
            linear_config.get("sequence_alpha"),
            frozen["linear"]["sequence_ridge_alpha"],
        ):
            failures.append("frozen sequence-ridge alpha mismatch")
        if not _same_number(
            linear_config.get("har_alpha"),
            frozen["linear"]["har_alpha"],
        ):
            failures.append("frozen HAR alpha mismatch")

        garch_config = json.loads(
            (runs["garch"] / "config.json").read_text(encoding="utf-8")
        )
        actual_garch = garch_config.get("garch_config", {})
        expected_garch = frozen["garch"]
        for key in ("p", "q", "mean", "distribution", "backend"):
            if actual_garch.get(key) != expected_garch[key]:
                failures.append(f"frozen GARCH {key} mismatch")
        for key in ("history", "minimum_history"):
            if int(garch_config.get(key, -1)) != int(expected_garch[key]):
                failures.append(f"frozen GARCH {key} mismatch")

        selected = json.loads(
            (runs["esn"] / "selected_spec.json").read_text(encoding="utf-8")
        )
        expected_esn = frozen["esn"]
        actual_esn = selected.get("selected_config", {})
        expected_config = {
            "config_id": expected_esn["config_id"],
            "n": expected_esn["n"],
            "connectivity": expected_esn["connectivity"],
            "sr": expected_esn["spectral_radius"],
            "inp": expected_esn["input_scale"],
            "leak": expected_esn["leak"],
        }
        for key, expected in expected_config.items():
            actual = actual_esn.get(key)
            if isinstance(expected, (int, float)):
                matched = _same_number(actual, expected)
            else:
                matched = actual == expected
            if not matched:
                failures.append(f"frozen ESN {key} mismatch")
        if not _same_number(selected.get("selected_alpha"), expected_esn["alpha"]):
            failures.append("frozen ESN alpha mismatch")
        if [int(seed) for seed in selected.get("final_seeds", [])] != [
            int(seed) for seed in expected_esn["seeds"]
        ]:
            failures.append("frozen ESN seed set mismatch")

        resolved_parameters = {
            "linear": {
                "har_alpha": linear_config.get("har_alpha"),
                "sequence_ridge_alpha": linear_config.get("sequence_alpha"),
            },
            "garch": {
                **actual_garch,
                "history": garch_config.get("history"),
                "minimum_history": garch_config.get("minimum_history"),
            },
            "esn": {
                **actual_esn,
                "alpha": selected.get("selected_alpha"),
                "seeds": selected.get("final_seeds"),
            },
        }

    report = {
        "schema_version": 2,
        "passed": not failures,
        "run_id": run_id,
        "classical_root": str(root),
        "inventory": inventory,
        "failures": failures,
        "models": list(MODELS),
        "reporting_groups": list(REQUIRED_GROUPS),
        "resolved_parameters": resolved_parameters,
        "frozen_specification": frozen,
        "test_evaluated": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError("classical validation failed: " + "; ".join(failures))
    return report
