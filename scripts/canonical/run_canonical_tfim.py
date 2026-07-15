#!/usr/bin/env python3
"""Canonical purged walk-forward evaluation of the frozen Phase 2 final TFIM-QRC.

``level`` reproduces the historical ``linear_clip_top240_alpha1000`` model.
``innovation`` preserves the frozen reservoir and readout geometry while fitting
Ridge directly to ``log(future_rv_20d / rv_20d)``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data.features import FEATURE_COLUMNS, make_sequence_arrays
from data.targets import (
    RV_INNOVATION_TARGET,
    RV_LEVEL_TARGET,
    RV_REFERENCE_COLUMN,
    add_rv_innovation_target,
    reconstruct_future_rv,
)
from evaluation.metrics import evaluate_volatility_forecast
from evaluation.transition import evaluate_transition_forecast
from experiments.runs import begin_run
from reservoirs.tfim import (
    TFIMQRCConfig,
    build_qrc_feature_matrix,
    leaky_integrate_windows,
    safe_feature_target_correlations,
)

MASTER_PATH = Path(__file__).with_name("run_master_comparison.py")
SPLITS = ("train", "val", "test")
LEAK = 0.3
TOP_K = 240
READOUT_ALPHA = 1000.0
DEFAULT_LEVEL_OUT = Path("results/canonical/run_canonical_tfim")
DEFAULT_LEVEL_TAG = "tfim_phase2_final"


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
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_LEVEL_OUT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default=DEFAULT_LEVEL_TAG)
    parser.add_argument("--task", choices=("level", "innovation"), default="level")
    parser.add_argument("--only-folds", nargs="*", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=7)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--verbose-features", action="store_true")
    return parser.parse_args()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def exact_phase2_readout(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    *,
    task: str,
) -> tuple[dict[str, np.ndarray], dict]:
    lower = np.percentile(features["train"], 1.0, axis=0)
    upper = np.percentile(features["train"], 99.0, axis=0)
    clipped = {split: np.clip(features[split], lower, upper) for split in SPLITS}

    correlations = safe_feature_target_correlations(clipped["train"], targets["train"])
    count = min(TOP_K, clipped["train"].shape[1])
    selected_indices = np.argsort(np.abs(correlations))[-count:]
    selected = {split: clipped[split][:, selected_indices] for split in SPLITS}

    scaler = StandardScaler()
    transformed = {
        "train": scaler.fit_transform(selected["train"]),
        "val": scaler.transform(selected["val"]),
        "test": scaler.transform(selected["test"]),
    }
    model = Ridge(alpha=READOUT_ALPHA)
    train_target = (
        np.log(np.maximum(targets["train"], 1e-8))
        if task == "level"
        else targets["train"]
    )
    model.fit(transformed["train"], train_target)
    scores = {split: model.predict(transformed[split]) for split in SPLITS}
    metadata = {
        "n_raw_features": int(features["train"].shape[1]),
        "n_selected_features": int(count),
        "selected_feature_indices": selected_indices.tolist(),
        "clip_percentiles": [1.0, 99.0],
        "feature_selection": "train_abs_feature_target_correlation",
        "ridge_alpha": READOUT_ALPHA,
        "target_transform": "log" if task == "level" else "none",
    }
    return scores, metadata


def evaluate_innovation_model(
    *,
    master,
    fold_id: int,
    targets: dict[str, np.ndarray],
    dates: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
    future_levels: dict[str, np.ndarray],
    q90_threshold: float,
    labels90: dict[str, np.ndarray],
    q95_threshold: float,
    labels95: dict[str, np.ndarray],
    metadata: dict,
) -> tuple[dict, list[dict]]:
    row = {
        "fold": fold_id,
        "model": "tfim_phase2_final",
        "protocol": "exact",
        "task": "innovation",
        **metadata,
    }
    for split in ("val", "test"):
        transition = evaluate_transition_forecast(targets[split], scores[split])
        row.update({f"{split}_{key}": value for key, value in transition.items()})
        reconstructed = reconstruct_future_rv(references[split], scores[split])
        level_metrics = asdict(evaluate_volatility_forecast(future_levels[split], reconstructed))
        row.update({f"{split}_reconstructed_{key}": value for key, value in level_metrics.items()})

    row["q90_target_threshold"] = q90_threshold
    row["q95_target_threshold"] = q95_threshold
    for quantile, labels in ((90, labels90), (95, labels95)):
        threshold, source = master.select_f1_threshold(
            labels["train"], labels["val"], scores["train"], scores["val"], quantile / 100.0
        )
        row[f"q{quantile}_score_threshold"] = threshold
        row[f"q{quantile}_threshold_source"] = source
        for split in ("val", "test"):
            values = master.classification_metrics(labels[split], scores[split], threshold)
            for key, value in values.items():
                row[f"q{quantile}_{split}_{key}"] = value

    prediction_rows: list[dict] = []
    for split in SPLITS:
        reconstructed = reconstruct_future_rv(references[split], scores[split])
        for index in range(len(targets[split])):
            prediction_rows.append(
                {
                    "fold": fold_id,
                    "model": "tfim_phase2_final",
                    "protocol": "exact",
                    "task": "innovation",
                    "split": split,
                    "date": dates[split][index],
                    "innovation_true": float(targets[split][index]),
                    "innovation_pred": float(scores[split][index]),
                    "reference_rv": float(references[split][index]),
                    "future_rv_true": float(future_levels[split][index]),
                    "predicted_future_rv": float(reconstructed[index]),
                    "q90_label": int(labels90[split][index]),
                    "q95_label": int(labels95[split][index]),
                }
            )
    return row, prediction_rows


def aggregate_task_metrics(per_fold: pd.DataFrame, task: str, master) -> pd.DataFrame:
    if task == "level":
        return master.aggregate_metrics(per_fold)
    numeric = [
        column
        for column in per_fold.columns
        if column.startswith("test_")
        or column.startswith("q90_test_")
        or column.startswith("q95_test_")
    ]
    rows = []
    for (model, protocol), group in per_fold.groupby(["model", "protocol"], dropna=False):
        base = {
            "model": model,
            "protocol": protocol,
            "task": task,
            "n_folds_regression": int(group["test_innovation_rmse"].notna().sum()),
            "n_folds_q90_valid": int(group["q90_test_ap"].notna().sum()),
            "n_folds_q95_valid": int(group["q95_test_ap"].notna().sum()),
        }
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            base[f"{column}_median"] = float(values.median()) if values.notna().any() else np.nan
            base[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            base[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.out_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    master = load_master()

    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    prediction_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    if args.task == "innovation":
        frame = add_rv_innovation_target(frame)
        target = RV_INNOVATION_TARGET
    else:
        target = RV_LEVEL_TARGET

    required = ["date", target, *FEATURE_COLUMNS]
    if args.task == "innovation":
        required += [RV_REFERENCE_COLUMN, RV_LEVEL_TARGET]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    folds = master.make_folds(
        len(frame),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        requested = set(args.only_folds)
        folds = [fold for fold in folds if fold["fold"] in requested]
        if not folds:
            raise ValueError(f"No available folds matched {sorted(requested)}")

    config = TFIMQRCConfig(
        qubits=6,
        pca_components=6,
        lookback_days=args.lookback,
        anchor_count=10,
        anchor_policy="recent",
        observable_mode="zxzz",
        trotter_steps_per_anchor=3,
        virtual_nodes_per_anchor=3,
        topology="full",
        coupling_scale=0.7,
        transverse_field=0.5,
        evolution_time=0.5,
        angle_max=np.pi / 2,
        ridge_alpha=READOUT_ALPHA,
        target_transform="log" if args.task == "level" else "none",
        seed=42,
        collect_anchor_features=True,
        use_disorder=True,
        disorder_strength=0.20,
    )

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        print(f"\n=== Canonical exact Phase 2 TFIM fold {fold_id} ({args.task}) ===")
        frames = {
            split: frame.iloc[fold[split][0] : fold[split][1]].copy().reset_index(drop=True)
            for split in SPLITS
        }

        scaler = StandardScaler()
        pca = PCA(n_components=config.pca_components, random_state=42)
        pca.fit(scaler.fit_transform(frames["train"][FEATURE_COLUMNS]))
        pca_columns = [f"pca{index + 1}" for index in range(config.pca_components)]

        sequences = {}
        for split in SPLITS:
            transformed = pca.transform(scaler.transform(frames[split][FEATURE_COLUMNS]))
            sequence_frame = pd.DataFrame(transformed, columns=pca_columns)
            sequence_frame[target] = frames[split][target].to_numpy()
            sequence_frame["date"] = frames[split]["date"].to_numpy()
            sequences[split] = make_sequence_arrays(
                sequence_frame,
                feature_columns=pca_columns,
                target_column=target,
                lookback=config.lookback_days,
            )

        windows = {
            split: leaky_integrate_windows(sequences[split][0], leak=LEAK)
            for split in SPLITS
        }
        targets = {split: np.asarray(sequences[split][1], dtype=float) for split in SPLITS}
        dates = {
            split: np.asarray(
                sequences[split][2].to_numpy()
                if hasattr(sequences[split][2], "to_numpy")
                else sequences[split][2]
            )
            for split in SPLITS
        }
        references = (
            {
                split: frames[split]
                .iloc[config.lookback_days - 1 :][RV_REFERENCE_COLUMN]
                .to_numpy(dtype=float)
                for split in SPLITS
            }
            if args.task == "innovation"
            else {}
        )
        future_levels = (
            {
                split: frames[split]
                .iloc[config.lookback_days - 1 :][RV_LEVEL_TARGET]
                .to_numpy(dtype=float)
                for split in SPLITS
            }
            if args.task == "innovation"
            else {}
        )

        q90_target, labels90 = master.labels_for(targets["train"], targets, 0.90)
        q95_target, labels95 = master.labels_for(targets["train"], targets, 0.95)

        start = time.perf_counter()
        features = {
            split: build_qrc_feature_matrix(
                windows[split], config, verbose=args.verbose_features and split == "train"
            )
            for split in SPLITS
        }
        feature_seconds = time.perf_counter() - start

        start = time.perf_counter()
        scores, readout_metadata = exact_phase2_readout(features, targets, task=args.task)
        fit_seconds = time.perf_counter() - start
        metadata = {
            "quantum_tasks_per_date": 1,
            "shots": np.nan,
            "shot_seed": np.nan,
            "fit_seconds": fit_seconds,
            "feature_seconds": feature_seconds,
            "selected_config": json.dumps(
                {
                    **config.__dict__,
                    "input_leak": LEAK,
                    **readout_metadata,
                },
                sort_keys=True,
                default=str,
            ),
            "selection_metric": "frozen_phase2_linear_clip_top240_alpha1000",
            "n_raw_features": readout_metadata["n_raw_features"],
            "n_selected_features": readout_metadata["n_selected_features"],
        }

        if args.task == "level":
            row, predictions = master.evaluate_model(
                model_name="tfim_phase2_final",
                protocol="exact",
                fold_id=fold_id,
                y=targets,
                dates=dates,
                scores=scores,
                q90_threshold=q90_target,
                lab90=labels90,
                q95_threshold=q95_target,
                lab95=labels95,
                metadata=metadata,
            )
            row["task"] = "level"
        else:
            row, predictions = evaluate_innovation_model(
                master=master,
                fold_id=fold_id,
                targets=targets,
                dates=dates,
                scores=scores,
                references=references,
                future_levels=future_levels,
                q90_threshold=q90_target,
                labels90=labels90,
                q95_threshold=q95_target,
                labels95=labels95,
                metadata=metadata,
            )

        metric_rows.append(row)
        prediction_rows.extend(predictions)
        atomic_csv(pd.DataFrame(metric_rows).sort_values("fold"), per_fold_path)
        atomic_csv(
            pd.DataFrame(prediction_rows).sort_values(["fold", "split", "date"]),
            prediction_path,
        )

    per_fold = pd.DataFrame(metric_rows).sort_values("fold").reset_index(drop=True)
    predictions = pd.DataFrame(prediction_rows).sort_values(["fold", "split", "date"]).reset_index(drop=True)
    aggregate = aggregate_task_metrics(per_fold, args.task, master)
    atomic_csv(per_fold, per_fold_path)
    atomic_csv(predictions, prediction_path)
    atomic_csv(aggregate, aggregate_path)
    manifest_path.write_text(
        json.dumps(
            {
                "model": "tfim_phase2_final",
                "historical_source": "scripts/canonical/run_canonical_tfim.py@phase3-refactor",
                "historical_notebook": "phase2-volatility-regression-qrc:notebooks/phase2_qrc_final_encoding_readout_probe.ipynb",
                "historical_run_name": "linear_clip_top240_alpha1000",
                "protocol": "exact",
                "task": args.task,
                "target": target,
                "folds": [int(fold["fold"]) for fold in folds],
                "configuration": config.__dict__,
                "input_leak": LEAK,
                "readout": {
                    "train_clip_percentiles": [1.0, 99.0],
                    "feature_selection": "top 240 by absolute train feature-target correlation",
                    "scaler": "StandardScaler fit on selected train features",
                    "ridge_alpha": READOUT_ALPHA,
                    "target_transform": "log" if args.task == "level" else "none",
                },
                "checkpoint_policy": "atomic CSV write after every completed fold within an immutable run directory",
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {prediction_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
