from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.classical_benchmarks.common import (
    REQUIRED_GROUPS,
    grouped_fold_metrics,
    grouped_horizon_metrics,
    grouped_path_metrics,
)

MODEL_ORDER = (
    "persistence",
    "har",
    "sequence_ridge",
    "garch_1_1_t",
    "esn_direct",
    "esn_shuffled",
)
KEY_COLUMNS = ["fold", "sample_id"]
ACTUAL_COLUMNS = [f"actual_h{h}" for h in range(1, 11)]
PREDICTED_COLUMNS = [f"predicted_h{h}" for h in range(1, 11)]


def _load_predictions(run_dir: Path, models: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(run_dir / "predictions.csv.gz")
    frame = frame[frame["model"].isin(models)].copy()
    finite = np.all(np.isfinite(frame[PREDICTED_COLUMNS].to_numpy(dtype=float)), axis=1)
    return frame.loc[finite].reset_index(drop=True)


def _available_keys(frame: pd.DataFrame, model: str) -> set[tuple[int, str]]:
    selected = frame[frame["model"].eq(model)]
    if selected.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"duplicate prediction keys for {model}")
    return set(zip(selected["fold"].astype(int), selected["sample_id"].astype(str)))


def _restrict_common(frame: pd.DataFrame, common_keys: set[tuple[int, str]]) -> pd.DataFrame:
    keys = list(zip(frame["fold"].astype(int), frame["sample_id"].astype(str)))
    return frame.loc[[key in common_keys for key in keys]].reset_index(drop=True)


def _validate_actuals(predictions: pd.DataFrame) -> None:
    reference = None
    for model in MODEL_ORDER:
        current = predictions[predictions["model"].eq(model)].sort_values(KEY_COLUMNS).reset_index(drop=True)
        if reference is None:
            reference = current[KEY_COLUMNS + ACTUAL_COLUMNS]
            continue
        if not current[KEY_COLUMNS].equals(reference[KEY_COLUMNS]):
            raise ValueError(f"common key alignment differs for {model}")
        if not np.allclose(current[ACTUAL_COLUMNS].to_numpy(float), reference[ACTUAL_COLUMNS].to_numpy(float)):
            raise ValueError(f"actual targets differ for {model}")


def _stage_metrics(predictions: pd.DataFrame, selection_folds: tuple[int, ...], confirmation_folds: tuple[int, ...]):
    stage_frames = {
        "selection": predictions[predictions["fold"].isin(selection_folds)].reset_index(drop=True),
        "confirmation": predictions[predictions["fold"].isin(confirmation_folds)].reset_index(drop=True),
        "development_all": predictions.reset_index(drop=True),
    }
    path = pd.concat([grouped_path_metrics(frame, stage=stage) for stage, frame in stage_frames.items()], ignore_index=True)
    horizon = pd.concat([grouped_horizon_metrics(frame, stage=stage) for stage, frame in stage_frames.items()], ignore_index=True)
    fold = grouped_fold_metrics(predictions)
    return path, horizon, fold


def _paired_deltas(metrics: pd.DataFrame, reference_model: str = "sequence_ridge") -> pd.DataFrame:
    reference = metrics[metrics["model"].eq(reference_model)][
        ["stage", "group", "rmse", "qlike", "mz_alpha", "mz_beta", "mz_r2"]
    ].rename(columns={
        "rmse": "reference_rmse",
        "qlike": "reference_qlike",
        "mz_alpha": "reference_mz_alpha",
        "mz_beta": "reference_mz_beta",
        "mz_r2": "reference_mz_r2",
    })
    compared = metrics.merge(reference, on=["stage", "group"], how="left", validate="many_to_one")
    compared = compared[~compared["model"].eq(reference_model)].copy()
    compared["reference_model"] = reference_model
    compared["delta_rmse"] = compared["rmse"] - compared["reference_rmse"]
    compared["delta_qlike"] = compared["qlike"] - compared["reference_qlike"]
    compared["delta_mz_r2"] = compared["mz_r2"] - compared["reference_mz_r2"]
    return compared


def run_canonical_comparison(
    *,
    linear_run: Path,
    garch_run: Path,
    esn_run: Path,
    results_root: Path,
    run_id: str | None = None,
    selection_folds: tuple[int, ...] = (4, 5, 6),
    confirmation_folds: tuple[int, ...] = (7, 8),
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    params = {
        "linear_run": linear_run,
        "garch_run": garch_run,
        "esn_run": esn_run,
        "selection_folds": selection_folds,
        "confirmation_folds": confirmation_folds,
        "models": MODEL_ORDER,
        "common_row_policy": "intersection across every reported model",
        "test_evaluated": False,
    }
    run_dir = begin_run(results_root, params, run_id=run_id)
    linear = _load_predictions(linear_run, ("persistence", "har", "sequence_ridge"))
    garch = _load_predictions(garch_run, ("garch_1_1_t",))
    esn = _load_predictions(esn_run, ("esn_direct", "esn_shuffled"))
    all_predictions = pd.concat([linear, garch, esn], ignore_index=True, sort=False)
    keys_by_model = {model: _available_keys(all_predictions, model) for model in MODEL_ORDER}
    common_keys = set.intersection(*(keys_by_model[model] for model in MODEL_ORDER))
    if not common_keys:
        raise ValueError("no common prediction rows across classical models")
    predictions = _restrict_common(all_predictions, common_keys)
    _validate_actuals(predictions)
    model_order = pd.Categorical(predictions["model"], categories=MODEL_ORDER, ordered=True)
    predictions = predictions.assign(_model_order=model_order).sort_values(["fold", "sample_id", "_model_order"]).drop(columns="_model_order")
    metrics, metrics_by_horizon, metrics_by_fold = _stage_metrics(predictions, selection_folds, confirmation_folds)
    metrics["model"] = pd.Categorical(metrics["model"], categories=MODEL_ORDER, ordered=True)
    metrics["group"] = pd.Categorical(metrics["group"], categories=REQUIRED_GROUPS, ordered=True)
    metrics = metrics.sort_values(["stage", "group", "model"]).reset_index(drop=True)
    metrics["model"] = metrics["model"].astype(str)
    metrics["group"] = metrics["group"].astype(str)
    paired = _paired_deltas(metrics)
    expected_keys = set.union(*(keys_by_model[model] for model in MODEL_ORDER))
    coverage_rows = []
    for model in MODEL_ORDER:
        available = keys_by_model[model]
        coverage_rows.append({
            "model": model,
            "available_rows": len(available),
            "common_rows": len(common_keys),
            "union_rows": len(expected_keys),
            "available_fraction_of_union": len(available) / len(expected_keys),
            "common_fraction_of_available": len(common_keys) / len(available),
        })
    coverage = pd.DataFrame(coverage_rows)
    predictions.to_csv(run_dir / "common_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "submission_metrics.csv", index=False)
    metrics_by_fold.to_csv(run_dir / "metrics_by_fold.csv", index=False)
    metrics_by_horizon.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    paired.to_csv(run_dir / "paired_deltas_vs_sequence_ridge.csv", index=False)
    coverage.to_csv(run_dir / "coverage.csv", index=False)
    for stage in ("selection", "confirmation", "development_all"):
        metrics[metrics["stage"].eq(stage)].to_csv(run_dir / f"submission_table_{stage}.csv", index=False)
    confirmation_pooled = metrics[
        metrics["stage"].eq("confirmation") & metrics["group"].eq("Pooled")
    ].sort_values(["qlike", "rmse"])
    confirmation_transition = metrics[
        metrics["stage"].eq("confirmation") & metrics["group"].eq("Transition")
    ].sort_values(["qlike", "rmse"])
    summary = {
        "status": "complete",
        "benchmark": "canonical_classical_comparison",
        "run_id": run_dir.name,
        "models": list(MODEL_ORDER),
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "test_evaluated": False,
        "common_rows": len(common_keys),
        "common_prediction_rows": int(len(predictions)),
        "confirmation_pooled_ranking": confirmation_pooled[
            ["model", "qlike", "rmse", "mz_alpha", "mz_beta", "mz_r2"]
        ].to_dict("records"),
        "confirmation_transition_ranking": confirmation_transition[
            ["model", "qlike", "rmse", "mz_alpha", "mz_beta", "mz_r2"]
        ].to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "runtime.json").write_text(json.dumps({"wall_seconds": time.perf_counter() - started}, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary
