from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from transition_forecasting.modeling.classical_benchmarks.common import (
    grouped_fold_metrics,
    grouped_horizon_metrics,
    grouped_path_metrics,
    sha256_file,
)


def finalize_esn_run(
    *,
    run_dir: Path,
    dataset_root: Path,
    selection_folds: tuple[int, ...] = (4, 5, 6),
    confirmation_folds: tuple[int, ...] = (7, 8),
    seeds: tuple[int, ...] = (1, 2, 3),
) -> dict[str, object]:
    started = time.perf_counter()
    selection_root = run_dir / "selection"
    selected = json.loads((selection_root / "selected_spec.json").read_text(encoding="utf-8"))
    selected_config = selected["selected_config"]
    selected_alpha = float(selected["selected_alpha"])
    all_folds = tuple(selection_folds) + tuple(confirmation_folds)
    fold_paths = [run_dir / "fold_predictions" / f"fold_{fold}.csv.gz" for fold in all_folds]
    missing = [str(path) for path in fold_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing ESN fold predictions: {missing}")
    predictions = pd.concat([pd.read_csv(path) for path in fold_paths], ignore_index=True)
    if set(predictions["fold_split"]) != {"val"}:
        raise AssertionError("ESN predictions contain non-validation rows")
    stage_frames = {
        "selection": predictions[predictions["fold"].isin(selection_folds)].reset_index(drop=True),
        "confirmation": predictions[predictions["fold"].isin(confirmation_folds)].reset_index(drop=True),
        "development_all": predictions.reset_index(drop=True),
    }
    submission_metrics = pd.concat(
        [grouped_path_metrics(frame, stage=stage) for stage, frame in stage_frames.items()], ignore_index=True
    )
    metrics_by_horizon = pd.concat(
        [grouped_horizon_metrics(frame, stage=stage) for stage, frame in stage_frames.items()], ignore_index=True
    )
    metrics_by_fold = grouped_fold_metrics(predictions)
    predictions.to_csv(run_dir / "predictions.csv.gz", index=False, compression="gzip")
    submission_metrics.to_csv(run_dir / "submission_metrics.csv", index=False)
    metrics_by_fold.to_csv(run_dir / "metrics_by_fold.csv", index=False)
    metrics_by_horizon.to_csv(run_dir / "metrics_by_horizon.csv", index=False)
    for name in ("esn_tuning_by_fold.csv", "esn_seed_metrics.csv", "esn_tuning_summary.csv"):
        (run_dir / name).write_bytes((selection_root / name).read_bytes())
    config = {
        "benchmark": "direct_esn",
        "dataset_root": str(dataset_root),
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "selected_config": selected_config,
        "selected_alpha": selected_alpha,
        "seeds": list(seeds),
        "representation": "level_diff_time",
        "pooling": "final_mean_std",
        "washout": 10,
        "target": "direct future log-volatility path",
        "test_evaluated": False,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest_path = dataset_root / "purged_walk_forward_folds" / "rematched_rolling_manifest.csv"
    tensors_path = dataset_root / "purged_walk_forward_folds" / "rematched_rolling_tensors.npz"
    source_rows = int(sum(1 for _ in manifest_path.open("rb")) - 1)
    dataset_manifest = {
        "manifest_path": str(manifest_path),
        "tensors_path": str(tensors_path),
        "manifest_sha256": sha256_file(manifest_path),
        "tensors_sha256": sha256_file(tensors_path),
        "source_rows": source_rows,
        "prediction_rows": int(len(predictions)),
        "test_rows_used": 0,
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8")
    selection_runtime = json.loads((selection_root / "runtime.json").read_text())["wall_seconds"]
    fold_runtime = 0.0
    for path in fold_paths:
        runtime_path = path.with_suffix(path.suffix + ".runtime.json")
        fold_runtime += float(json.loads(runtime_path.read_text())["wall_seconds"])
    finalization_seconds = time.perf_counter() - started
    runtime = {
        "selection_wall_seconds": selection_runtime,
        "sum_fold_wall_seconds": fold_runtime,
        "finalization_wall_seconds": finalization_seconds,
        "total_sequential_wall_seconds": selection_runtime + fold_runtime + finalization_seconds,
    }
    (run_dir / "runtime.json").write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    confirmation = submission_metrics[
        submission_metrics["stage"].eq("confirmation") & submission_metrics["group"].eq("Pooled")
    ].sort_values(["qlike", "rmse"])
    summary = {
        "status": "complete",
        "benchmark": "direct_esn",
        "run_id": run_dir.name,
        "selected_config": selected_config,
        "selected_alpha": selected_alpha,
        "selection_folds": list(selection_folds),
        "confirmation_folds": list(confirmation_folds),
        "seeds": list(seeds),
        "test_evaluated": False,
        "confirmation_pooled_ranking": confirmation[
            ["model", "qlike", "rmse", "mz_alpha", "mz_beta", "mz_r2"]
        ].to_dict("records"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--selection-folds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--confirmation-folds", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    summary = finalize_esn_run(
        run_dir=args.run_dir,
        dataset_root=args.dataset_root,
        selection_folds=tuple(args.selection_folds),
        confirmation_folds=tuple(args.confirmation_folds),
        seeds=tuple(args.seeds),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
