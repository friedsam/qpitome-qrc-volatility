#!/usr/bin/env python3
"""Canonical purged walk-forward evaluation of the historical Phase 2 HAR-like ridge.

This reproduces the exact Phase 2 formulation under the common Phase 3 fold
protocol:

- inputs: rv_5d, rv_10d, rv_20d, rv_60d, vix_close
- train-fitted StandardScaler
- Ridge(alpha=1.0)
- raw future_rv_20d target, not log target
- predictions clipped to >= 1e-8

Only the evaluation protocol changes; the model formulation does not.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run

MASTER_PATH = Path(__file__).with_name("run_master_comparison.py")
TARGET = "future_rv_20d"
SPLITS = ("train", "val", "test")
HAR_FEATURES = ["rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close"]


def load_master():
    spec = importlib.util.spec_from_file_location("canonical_master_helpers", MASTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MASTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/spy_vix_volatility/spy_vix_volatility.csv"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/canonical/run_canonical_har"),
    )
    p.add_argument("--run-id", default=None)
    p.add_argument("--tag", default="har_ridge")
    p.add_argument("--only-folds", nargs="*", type=int, default=None)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--lookback", type=int, default=40, help="Common comparison date alignment only")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    args.out_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    master = load_master()

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    pred_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    if all(p.exists() for p in (per_fold_path, pred_path, aggregate_path, manifest_path)) and not args.force:
        print(f"HAR segment already complete: {args.out_dir}")
        return

    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    required = {"date", TARGET, *HAR_FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing HAR columns: {missing}")

    folds = master.make_folds(
        len(df), n_folds=args.n_folds, min_train=args.min_train,
        val_size=args.val_size, purge=args.purge,
    )
    if args.only_folds:
        keep = set(args.only_folds)
        folds = [f for f in folds if f["fold"] in keep]

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []

    for fold in folds:
        fold_id = fold["fold"]
        print(f"\n=== Canonical HAR fold {fold_id} ===")
        aligned = master.aligned_frames(df, fold, args.lookback)
        y = {s: aligned[s][TARGET].to_numpy(float) for s in SPLITS}
        dates = {s: aligned[s]["date"].to_numpy() for s in SPLITS}
        q90_target, lab90 = master.labels_for(y["train"], y, 0.90)
        q95_target, lab95 = master.labels_for(y["train"], y, 0.95)

        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        start = time.perf_counter()
        model.fit(aligned["train"][HAR_FEATURES].to_numpy(float), y["train"])
        raw_predictions = {
            s: np.maximum(model.predict(aligned[s][HAR_FEATURES].to_numpy(float)), 1e-8)
            for s in SPLITS
        }
        fit_seconds = time.perf_counter() - start
        scores = {s: np.log(raw_predictions[s]) for s in SPLITS}

        row, preds = master.evaluate_model(
            model_name="har_ridge",
            protocol="classical",
            fold_id=fold_id,
            y=y,
            dates=dates,
            scores=scores,
            q90_threshold=q90_target,
            lab90=lab90,
            q95_threshold=q95_target,
            lab95=lab95,
            metadata={
                "quantum_tasks_per_date": 0,
                "shots": np.nan,
                "shot_seed": np.nan,
                "fit_seconds": fit_seconds,
                "feature_seconds": 0.0,
                "selected_config": json.dumps({
                    "features": HAR_FEATURES,
                    "scaler": "StandardScaler",
                    "estimator": "Ridge",
                    "alpha": 1.0,
                    "target_transform": "none",
                    "prediction_floor": 1e-8,
                }, sort_keys=True),
                "selection_metric": "frozen_phase2_har_ridge_alpha_1",
            },
        )
        metric_rows.append(row)
        prediction_rows.extend(preds)

        atomic_csv(pd.DataFrame(metric_rows), per_fold_path)
        atomic_csv(pd.DataFrame(prediction_rows), pred_path)

    per_fold = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate = master.aggregate_metrics(per_fold)
    atomic_csv(per_fold, per_fold_path)
    atomic_csv(predictions, pred_path)
    atomic_csv(aggregate, aggregate_path)
    manifest_path.write_text(json.dumps({
        "model": "har_ridge",
        "historical_source": "archive/phase2/scripts/run_phase2_classical_baselines.py",
        "historical_model_name": "har_ridge_alpha_1",
        "features": HAR_FEATURES,
        "target_transform": "none",
        "prediction_floor": 1e-8,
        "folds": [f["fold"] for f in folds],
        "checkpoint_policy": "atomic CSV write after every completed fold",
    }, indent=2))

    print("\n=== Canonical HAR aggregate ===")
    print(aggregate.to_string(index=False))
    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {pred_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
