#!/usr/bin/env python3
"""Canonical purged walk-forward evaluation of the Phase 2 final TFIM QRC.

This ports the frozen ``linear_clip_top240_alpha1000`` model:

- train-only StandardScaler and PCA-6 inputs;
- 40-day split-local windows with leak 0.3;
- six-qubit full-topology exact-state TFIM reservoir;
- ten recent anchors, three Trotter steps and three virtual nodes per anchor;
- Z, X, and nearest-neighbor ZZ trajectory observables;
- fixed disorder strength 0.20;
- train-percentile clipping and top-240 train-correlation feature selection;
- StandardScaler plus Ridge(alpha=1000) on the log volatility target.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from data.features import (
    FEATURE_COLUMNS,
    drop_nonfinite_model_rows,
    make_sequence_arrays,
    scale_splits_train_only,
)
from evaluation.walkforward import make_purged_walkforward_folds, slice_fold_frames
from experiments.runs import begin_run
from reservoirs.tfim import (
    TFIMQRCConfig,
    build_qrc_feature_matrix,
    config_to_dict,
    fit_canonical_readout,
    leaky_integrate_windows,
    predict_canonical_readout,
)

SCRIPT_DIR = Path(__file__).resolve().parent
MASTER_PATH = SCRIPT_DIR.parents[1] / "canonical/run_master_comparison.py"
TARGET = "future_rv_20d"
SPLITS = ("train", "val", "test")


def load_master():
    spec = importlib.util.spec_from_file_location("canonical_master_helpers", MASTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MASTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/spy_vix_volatility/spy_vix_volatility.csv"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/quantum/tfim/run_tfim_walkforward"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default="tfim_phase2_final")
    parser.add_argument("--n-folds", type=int, default=7)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--only-folds", nargs="*", type=int, default=None)
    parser.add_argument("--verbose-features", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    args.out_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    master = load_master()
    config = TFIMQRCConfig()

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    required = {"date", TARGET, *FEATURE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    folds = make_purged_walkforward_folds(
        len(frame),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        requested = set(args.only_folds)
        available = {int(fold["fold"]) for fold in folds}
        folds = [fold for fold in folds if int(fold["fold"]) in requested]
        if not folds:
            raise ValueError(
                f"Requested folds {sorted(requested)} do not match available fold IDs "
                f"{sorted(available)}"
            )

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    fold_manifests: list[dict] = []

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    predictions_path = args.out_dir / f"predictions_{args.tag}.csv"
    diagnostics_path = args.out_dir / f"feature_diagnostics_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    for fold in folds:
        fold_id = int(fold["fold"])
        print(f"\n=== TFIM fold {fold_id} ===")
        split_frames = slice_fold_frames(frame, fold)
        cleaned = {
            split: drop_nonfinite_model_rows(
                split_frame,
                feature_columns=list(FEATURE_COLUMNS),
                target_columns=[TARGET],
            )
            for split, split_frame in split_frames.items()
        }
        scaled, _ = scale_splits_train_only(
            cleaned,
            feature_columns=list(FEATURE_COLUMNS),
            scaler_name="standard",
        )

        pca = PCA(n_components=config.pca_components, random_state=config.seed)
        pca.fit(scaled["train"][FEATURE_COLUMNS].to_numpy(dtype=float))
        pca_columns = [f"pca{index + 1}" for index in range(config.pca_components)]

        windows: dict[str, np.ndarray] = {}
        targets: dict[str, np.ndarray] = {}
        dates: dict[str, np.ndarray] = {}
        for split in SPLITS:
            transformed = pca.transform(
                scaled[split][FEATURE_COLUMNS].to_numpy(dtype=float)
            )
            sequence_frame = pd.DataFrame(transformed, columns=pca_columns)
            sequence_frame["date"] = scaled[split]["date"].to_numpy()
            sequence_frame[TARGET] = scaled[split][TARGET].to_numpy(dtype=float)
            X, y, output_dates = make_sequence_arrays(
                sequence_frame,
                feature_columns=pca_columns,
                target_column=TARGET,
                lookback=config.lookback_days,
            )
            windows[split] = leaky_integrate_windows(X, leak=config.input_leak)
            targets[split] = y
            dates[split] = output_dates.to_numpy()

        start = time.perf_counter()
        reservoir_features = {
            split: build_qrc_feature_matrix(
                windows[split],
                config,
                verbose=args.verbose_features,
            )
            for split in SPLITS
        }
        feature_seconds = time.perf_counter() - start

        fit_start = time.perf_counter()
        readout, feature_scaler, selected, lower, upper = fit_canonical_readout(
            reservoir_features["train"],
            targets["train"],
            config,
        )
        forecasts = {
            split: predict_canonical_readout(
                reservoir_features[split],
                readout=readout,
                scaler=feature_scaler,
                selected=selected,
                lower=lower,
                upper=upper,
                config=config,
            )
            for split in SPLITS
        }
        fit_seconds = time.perf_counter() - fit_start
        scores = {
            split: np.log(np.clip(forecasts[split], 1e-12, None))
            for split in SPLITS
        }

        q90_target, labels90 = master.labels_for(targets["train"], targets, 0.90)
        q95_target, labels95 = master.labels_for(targets["train"], targets, 0.95)
        row, predictions = master.evaluate_model(
            model_name="tfim_phase2_final",
            protocol="exact_state",
            fold_id=fold_id,
            y=targets,
            dates=dates,
            scores=scores,
            q90_threshold=q90_target,
            lab90=labels90,
            q95_threshold=q95_target,
            lab95=labels95,
            metadata={
                "quantum_tasks_per_date": 0,
                "shots": np.nan,
                "shot_seed": np.nan,
                "fit_seconds": fit_seconds,
                "feature_seconds": feature_seconds,
                "selected_config": json.dumps(config_to_dict(config), sort_keys=True),
                "selection_metric": "frozen_phase2_linear_clip_top240_alpha1000",
                "n_reservoir_features": int(reservoir_features["train"].shape[1]),
                "n_selected_features": int(len(selected)),
            },
        )
        metric_rows.append(row)
        prediction_rows.extend(predictions)

        for split in SPLITS:
            feature_matrix = reservoir_features[split]
            diagnostic_rows.append(
                {
                    "fold": fold_id,
                    "split": split,
                    "n_samples": int(feature_matrix.shape[0]),
                    "n_features": int(feature_matrix.shape[1]),
                    "near_constant_features": int(
                        np.sum(feature_matrix.std(axis=0) < 1e-8)
                    ),
                    "feature_std_median": float(
                        np.median(feature_matrix.std(axis=0))
                    ),
                    "pca_explained_variance_sum": float(
                        pca.explained_variance_ratio_.sum()
                    ),
                }
            )

        fold_manifests.append(
            {
                "fold": fold_id,
                "row_ranges": {
                    split: list(fold[split])
                    for split in ("train", "val", "purge", "test")
                },
                "sequence_counts": {
                    split: int(len(windows[split])) for split in SPLITS
                },
                "reservoir_feature_count": int(
                    reservoir_features["train"].shape[1]
                ),
                "selected_feature_count": int(len(selected)),
                "feature_seconds": feature_seconds,
                "fit_seconds": fit_seconds,
            }
        )

        atomic_csv(pd.DataFrame(metric_rows), per_fold_path)
        atomic_csv(pd.DataFrame(prediction_rows), predictions_path)
        atomic_csv(pd.DataFrame(diagnostic_rows), diagnostics_path)

    per_fold = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    aggregate = master.aggregate_metrics(per_fold)
    atomic_csv(per_fold, per_fold_path)
    atomic_csv(predictions, predictions_path)
    atomic_csv(diagnostics, diagnostics_path)
    atomic_csv(aggregate, aggregate_path)

    manifest_path.write_text(
        json.dumps(
            {
                "model": "tfim_phase2_final",
                "historical_source": (
                    "archive/phase2/scripts/export_final_qrc_predictions.py and "
                    "phase2_qrc_final_encoding_readout_probe.ipynb"
                ),
                "historical_run_name": "linear_clip_top240_alpha1000",
                "protocol": "purged_walkforward_exact_state",
                "data": str(args.data),
                "target": TARGET,
                "configuration": config_to_dict(config),
                "folds": fold_manifests,
                "classification_layer": (
                    "q90/q95 future-volatility event labels with validation-selected "
                    "F1 thresholds and training-quantile fallback"
                ),
                "checkpoint_policy": (
                    "atomic CSV write after every completed fold within an immutable run"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== TFIM aggregate ===")
    print(aggregate.to_string(index=False))
    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {diagnostics_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
