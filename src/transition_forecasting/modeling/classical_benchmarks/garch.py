from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.garch import GARCHConfig, config_to_dict, fit_garch_variance_path, variance_path_to_log_volatility_path
from experiments.runs import begin_run
from transition_forecasting.modeling.classical_benchmarks.common import (
    grouped_fold_metrics,
    grouped_horizon_metrics,
    grouped_path_metrics,
    load_rematched_dataset,
    sha256_file,
)
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS

DEFAULT_SELECTION_FOLDS = (4, 5, 6)
DEFAULT_CONFIRMATION_FOLDS = (7, 8)
_WORKER_CLOSES: dict[str, pd.Series] = {}


def _load_close_panel(path: Path) -> dict[str, pd.Series]:
    frame = pd.read_csv(path, usecols=["index", "date", "close"])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["index", "date", "close"])
    frame = frame[frame["close"] > 0]
    frame = frame.drop_duplicates(["index", "date"], keep="last").sort_values(["index", "date"])
    return {str(name): group.set_index("date")["close"].astype(float) for name, group in frame.groupby("index", sort=False)}


def _init_worker(ohlc_path: str) -> None:
    global _WORKER_CLOSES
    _WORKER_CLOSES = _load_close_panel(Path(ohlc_path))


def _return_history(close: pd.Series, origin_date: pd.Timestamp, history: int) -> np.ndarray:
    origin = pd.Timestamp(origin_date)
    if origin.tzinfo is not None:
        origin = origin.tz_localize(None)
    available = close.loc[:origin]
    returns = np.log(available).diff().dropna()
    if history > 0:
        returns = returns.iloc[-history:]
    return returns.to_numpy(dtype=float)


def _fit_one(task: tuple[int, str, str, int, int, dict[str, object]]) -> dict[str, object]:
    row_number, index_name, origin_text, history, minimum_history, config_payload = task
    close = _WORKER_CLOSES.get(index_name)
    if close is None:
        return {"row_number": row_number, "prediction": [np.nan] * 10, "history_rows": 0, "converged": False, "fit_note": "missing_market", "parameters": {}}
    returns = _return_history(close, pd.Timestamp(origin_text), history)
    if len(returns) < minimum_history:
        return {
            "row_number": row_number,
            "prediction": [np.nan] * 10,
            "history_rows": int(len(returns)),
            "converged": False,
            "fit_note": f"insufficient_history:{len(returns)}<{minimum_history}",
            "parameters": {},
        }
    config = GARCHConfig(**config_payload)
    forecast = fit_garch_variance_path(returns, horizon=10, config=config)
    prediction = variance_path_to_log_volatility_path(forecast.variance_path, return_scale=config.return_scale)
    return {
        "row_number": row_number,
        "prediction": prediction.tolist(),
        "history_rows": int(len(returns)),
        "converged": bool(forecast.converged and np.isfinite(prediction).all()),
        "convergence_flag": forecast.convergence_flag,
        "fit_note": forecast.note,
        "parameters": forecast.parameters,
    }


def run_garch_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    run_id: str | None = None,
    selection_folds: tuple[int, ...] = DEFAULT_SELECTION_FOLDS,
    confirmation_folds: tuple[int, ...] = DEFAULT_CONFIRMATION_FOLDS,
    history: int = 2500,
    minimum_history: int = 250,
    backend: str = "auto",
    max_workers: int | None = None,
    limit_rows: int | None = None,
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    dataset = load_rematched_dataset(dataset_root)
    ohlc_path = Path(dataset_root) / "cleaned_ohlc.csv.gz"
    all_folds = tuple(selection_folds) + tuple(confirmation_folds)
    source = dataset.manifest[
        dataset.manifest["fold"].isin(all_folds) & dataset.manifest["fold_split"].eq("val")
    ].copy().reset_index(drop=True)
    if limit_rows is not None:
        source = source.iloc[:limit_rows].copy().reset_index(drop=True)
    if source.empty:
        raise ValueError("no validation rows selected")
    config = GARCHConfig(backend=backend)
    params = {
        "benchmark": "garch_1_1_t",
        "dataset_root": dataset_root,
        "ohlc_path": ohlc_path,
        "selection_folds": selection_folds,
        "confirmation_folds": confirmation_folds,
        "history": history,
        "minimum_history": minimum_history,
        "garch_config": config_to_dict(config),
        "max_workers": max_workers,
        "limit_rows": limit_rows,
        "test_evaluated": False,
    }
    run_dir = begin_run(results_root, params, run_id=run_id)
    tasks = [
        (int(i), str(row["index"]), str(row["origin_date"]), history, minimum_history, config_to_dict(config))
        for i, row in source.iterrows()
    ]
    workers = max_workers or min(8, os.cpu_count() or 1)
    _init_worker(str(ohlc_path))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fitted = list(pool.map(_fit_one, tasks))
    fitted = sorted(fitted, key=lambda row: int(row["row_number"]))
    prediction = np.asarray([row["prediction"] for row in fitted], dtype=float)
    finite = np.all(np.isfinite(prediction), axis=1)
    metadata_columns = [
        "sample_id", "episode_id", "index", "market_group", "origin_date", "event_onset",
        "label", "lead", "control_stratum", "fold", "fold_split",
    ]
    output = source[metadata_columns].copy()
    output.insert(0, "model", "garch_1_1_t")
    actual = source[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    for horizon_number in range(1, 11):
        output[f"actual_h{horizon_number}"] = actual[:, horizon_number - 1]
        output[f"predicted_h{horizon_number}"] = prediction[:, horizon_number - 1]
    output["history_rows"] = [row["history_rows"] for row in fitted]
    output["converged"] = [row["converged"] for row in fitted]
    output["fit_note"] = [row["fit_note"] for row in fitted]
    output["parameters_json"] = [json.dumps(row.get("parameters", {}), sort_keys=True) for row in fitted]
    scored = output.loc[finite].reset_index(drop=True)
    if scored.empty:
        raise RuntimeError("GARCH produced no finite forecasts")
    stage_frames = {
        "selection": scored[scored["fold"].isin(selection_folds)].reset_index(drop=True),
        "confirmation": scored[scored["fold"].isin(confirmation_folds)].reset_index(drop=True),
        "development_all": scored.reset_index(drop=True),
    }
    submission_metrics = pd.concat([grouped_path_metrics(frame, stage=stage) for stage, frame in stage_frames.items()], ignore_index=True)
    metrics_by_horizon = pd.concat([grouped_horizon_metrics(frame, stage=stage) for stage, frame in stage_frames.items()], ignore_index=True)
    metrics_by_fold = grouped_fold_metrics(scored)
    output.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    submission_metrics.to_csv(run_dir / "submission_metrics.csv", index=False)
    metrics_by_fold.to_csv(run_dir / "metrics_by_fold.csv", index=False)
    metrics_by_horizon.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    diagnostics = output[["sample_id", "fold", "index", "origin_date", "history_rows", "converged", "fit_note", "parameters_json"]]
    diagnostics.to_csv(run_dir / "fit_diagnostics.csv.gz", index=False, compression="gzip")
    config_json = {**params, "reporting_groups": ["Transition", "L1", "L5", "L10", "Controls", "Pooled"]}
    (run_dir / "config.json").write_text(json.dumps(config_json, indent=2, default=str) + "\n", encoding="utf-8")
    dataset_manifest = {
        "manifest_path": str(dataset.manifest_path),
        "tensors_path": str(dataset.tensors_path),
        "ohlc_path": str(ohlc_path),
        "manifest_sha256": dataset.manifest_sha256,
        "tensors_sha256": dataset.tensors_sha256,
        "ohlc_sha256": sha256_file(ohlc_path),
        "source_rows": int(len(dataset.manifest)),
        "requested_prediction_rows": int(len(output)),
        "finite_prediction_rows": int(finite.sum()),
        "test_rows_used": 0,
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")
    runtime = {"wall_seconds": time.perf_counter() - started, "max_workers": workers}
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    confirmation = submission_metrics[
        submission_metrics["stage"].eq("confirmation") & submission_metrics["group"].eq("Pooled")
    ]
    summary = {
        "status": "complete",
        "benchmark": "garch_1_1_t",
        "run_id": run_dir.name,
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "test_evaluated": False,
        "requested_rows": int(len(output)),
        "finite_rows": int(finite.sum()),
        "coverage": float(finite.mean()),
        "confirmation_pooled": confirmation[["model", "qlike", "rmse", "mz_alpha", "mz_beta", "mz_r2"]].to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir, summary
