from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.classical_benchmarks.common import (
    grouped_fold_metrics,
    grouped_horizon_metrics,
    grouped_path_metrics,
    load_rematched_dataset,
)
from transition_forecasting.modeling.classical_benchmarks.esn import predict_fold


def run_frozen_esn_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    run_id: str,
    config: dict[str, object],
    alpha: float,
    seeds: tuple[int, ...],
    selection_folds: tuple[int, ...] = (4, 5, 6),
    confirmation_folds: tuple[int, ...] = (7, 8),
) -> tuple[Path, dict[str, object]]:
    started = time.perf_counter()
    dataset = load_rematched_dataset(dataset_root)
    all_folds = tuple(selection_folds) + tuple(confirmation_folds)
    params = {
        "benchmark": "direct_esn_frozen",
        "dataset_root": dataset_root,
        "selection_folds": selection_folds,
        "confirmation_folds": confirmation_folds,
        "config": config,
        "alpha": alpha,
        "seeds": seeds,
        "representation": "level_diff_time",
        "pooling": "final_mean_std",
        "washout": 10,
        "test_evaluated": False,
    }
    run_dir = begin_run(results_root, params, run_id=run_id)
    predictions = pd.concat(
        [
            predict_fold(
                dataset,
                fold=fold,
                config=config,
                alpha=alpha,
                seeds=seeds,
            )
            for fold in all_folds
        ],
        ignore_index=True,
    )
    if set(predictions["fold_split"]) != {"val"}:
        raise AssertionError("ESN predictions must contain validation rows only")
    stages = {
        "selection": predictions[predictions["fold"].isin(selection_folds)].reset_index(drop=True),
        "confirmation": predictions[predictions["fold"].isin(confirmation_folds)].reset_index(drop=True),
        "development_all": predictions.reset_index(drop=True),
    }
    metrics = pd.concat(
        [grouped_path_metrics(frame, stage=stage) for stage, frame in stages.items()],
        ignore_index=True,
    )
    horizon_metrics = pd.concat(
        [grouped_horizon_metrics(frame, stage=stage) for stage, frame in stages.items()],
        ignore_index=True,
    )
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(run_dir / "submission_metrics.csv", index=False)
    grouped_fold_metrics(predictions).to_csv(run_dir / "metrics_by_fold.csv", index=False)
    horizon_metrics.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    config_payload = {
        **params,
        "models": ["esn_direct_tuned", "esn_shuffled_tuned"],
        "reporting_groups": ["Transition", "L1", "L5", "L10", "Controls", "Pooled"],
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (run_dir / "selected_spec.json").write_text(
        json.dumps(
            {
                "selected_config": config,
                "selected_alpha": alpha,
                "final_seeds": list(seeds),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "manifest_path": str(dataset.manifest_path),
                "tensors_path": str(dataset.tensors_path),
                "manifest_sha256": dataset.manifest_sha256,
                "tensors_sha256": dataset.tensors_sha256,
                "source_rows": int(len(dataset.manifest)),
                "prediction_rows": int(len(predictions)),
                "test_rows_used": 0,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (run_dir / "runtime.json").write_text(
        json.dumps({"wall_seconds": time.perf_counter() - started}, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "complete",
        "benchmark": "direct_esn_frozen",
        "run_id": run_id,
        "selected_config": config,
        "selected_alpha": alpha,
        "seeds": list(seeds),
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "test_evaluated": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir, summary
