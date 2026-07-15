#!/usr/bin/env python3
"""Run the historical canonical Rydberg volatility models.

This runner preserves the late Phase 3 two-channel Rydberg implementation and
its six canonical variants. It is intentionally opt-in and does not redesign
the underdeveloped level/rate input encoding.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from reservoirs.rydberg import (
    RydbergQRCConfig,
    build_rydberg_feature_matrix,
    ensure_market_scalar,
    make_level_rate_sequence_splits,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
MASTER_PATH = SCRIPT_DIR / "run_master_comparison.py"
TARGET = "future_rv_20d"
SPLITS = ("train", "val", "test")
RYDBERG_MODELS = (
    "rydberg_temporal",
    "rydberg_memoryless",
    "rydberg_shuffled",
    "rydberg_multi_lb",
    "rydberg_multi_lb_memoryless",
    "rydberg_multi_lb_shuffled",
)
PROTOCOLS = ("exact", "exact_train_noisy_test", "noisy_train_noisy_test")


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
        default=Path("results/canonical/run_canonical_rydberg"),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tag", default="rydberg_historical")
    parser.add_argument("--models", nargs="*", choices=RYDBERG_MODELS, default=list(RYDBERG_MODELS))
    parser.add_argument("--protocols", nargs="*", choices=PROTOCOLS, default=["exact"])
    parser.add_argument("--only-folds", nargs="*", type=int, default=None)
    parser.add_argument("--n-folds", type=int, default=7)
    parser.add_argument("--min-train", type=int, default=2500)
    parser.add_argument("--val-size", type=int, default=504)
    parser.add_argument("--purge", type=int, default=60)
    parser.add_argument("--lookback", type=int, default=40)
    parser.add_argument("--lookback-fast", type=int, default=10)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--total-time-us", type=float, default=0.55)
    parser.add_argument("--level-col", default="vix_rv_spread")
    parser.add_argument("--rate-col", default="rv_accel_log_5_20")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--n-slow", type=int, default=4)
    parser.add_argument("--n-fast", type=int, default=4)
    parser.add_argument("--spacing-slow-um", type=float, default=9.0)
    parser.add_argument("--spacing-fast-um", type=float, default=15.0)
    parser.add_argument("--row-gap-um", type=float, default=14.0)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--shot-seeds", default="7")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("scratch/canonical_cache"),
    )
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def cache_key(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_or_build_feature_block(
    *,
    cache_dir: Path,
    cache_payload: dict,
    arrays: dict[str, np.ndarray],
    config: RydbergQRCConfig,
    force: bool,
) -> tuple[dict[str, np.ndarray], float, str]:
    key = cache_key(cache_payload)
    path = cache_dir / f"rydberg_{key}.npz"
    if path.exists() and not force:
        loaded = np.load(path)
        return {split: loaded[split] for split in SPLITS}, 0.0, str(path)

    start = time.perf_counter()
    block: dict[str, np.ndarray] = {}
    for split_index, split in enumerate(SPLITS):
        split_rng = None
        if config.shots is not None:
            split_rng = np.random.default_rng(config.shot_seed + 100_003 * split_index)
        block[split] = build_rydberg_feature_matrix(
            arrays[split],
            config,
            rng=split_rng,
        )
    elapsed = time.perf_counter() - start
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **block)
    return block, elapsed, str(path)


def fit_log_ridge(
    features: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, np.ndarray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(features["train"], np.log(np.clip(targets["train"], 1e-12, None)))
    return {split: model.predict(features[split]) for split in SPLITS}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.out_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    master = load_master()

    frame = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    frame = ensure_market_scalar(ensure_market_scalar(frame, args.level_col), args.rate_col)
    required = {
        "date",
        TARGET,
        args.level_col,
        args.rate_col,
        "rv_5d",
        "rv_20d",
        "vix_close",
    }
    missing = sorted(required - set(frame.columns))
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
        folds = [fold for fold in folds if int(fold["fold"]) in requested]
        if not folds:
            raise ValueError(f"No available folds matched {sorted(requested)}")

    base_config = RydbergQRCConfig(
        n_atoms_slow=args.n_slow,
        n_atoms_fast=args.n_fast,
        spacing_slow_um=args.spacing_slow_um,
        spacing_fast_um=args.spacing_fast_um,
        row_gap_um=args.row_gap_um,
        lookback_days=args.lookback,
        anchor_count=args.anchors,
        anchor_policy="even",
        reverse_anchors=True,
        total_time_us=args.total_time_us,
        shots=None,
        ridge_alpha=args.ridge_alpha,
    )
    fast_config = replace(base_config, lookback_days=args.lookback_fast)
    block_configs = {
        "temporal": base_config,
        "fast": fast_config,
        "memoryless": replace(base_config, memory_mode="memoryless"),
        "fast_memoryless": replace(fast_config, memory_mode="memoryless"),
        "shuffled": replace(base_config, shuffle_anchors=True),
        "fast_shuffled": replace(fast_config, shuffle_anchors=True),
    }
    model_blocks = {
        "rydberg_temporal": ("temporal",),
        "rydberg_memoryless": ("memoryless",),
        "rydberg_shuffled": ("shuffled",),
        "rydberg_multi_lb": ("temporal", "fast"),
        "rydberg_multi_lb_memoryless": ("memoryless", "fast_memoryless"),
        "rydberg_multi_lb_shuffled": ("shuffled", "fast_shuffled"),
    }
    shot_seeds = [int(value) for value in args.shot_seeds.split(",") if value.strip()]

    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    cache_manifest: list[dict] = []

    for fold in folds:
        fold_id = int(fold["fold"])
        print(f"\n=== Historical canonical Rydberg fold {fold_id} ===")
        frames = {
            split: frame.iloc[fold[split][0] : fold[split][1]].copy().reset_index(drop=True)
            for split in SPLITS
        }
        sequence = make_level_rate_sequence_splits(
            frames,
            level_col=args.level_col,
            rate_col=args.rate_col,
            target_column=TARGET,
            lookback_days=args.lookback,
        )
        sequence_fast = make_level_rate_sequence_splits(
            frames,
            level_col=args.level_col,
            rate_col=args.rate_col,
            target_column=TARGET,
            lookback_days=args.lookback_fast,
        )
        sequence_fast = {
            split: (
                X[len(X) - len(sequence[split][0]) :],
                y[len(y) - len(sequence[split][1]) :],
                dates.iloc[len(dates) - len(sequence[split][2]) :].reset_index(drop=True),
            )
            for split, (X, y, dates) in sequence_fast.items()
        }
        if args.train_stride > 1:
            stride = args.train_stride
            sequence = {
                split: (
                    (X[::stride], y[::stride], dates.iloc[::stride].reset_index(drop=True))
                    if split == "train"
                    else (X, y, dates)
                )
                for split, (X, y, dates) in sequence.items()
            }
            sequence_fast = {
                split: (
                    (X[::stride], y[::stride], dates.iloc[::stride].reset_index(drop=True))
                    if split == "train"
                    else (X, y, dates)
                )
                for split, (X, y, dates) in sequence_fast.items()
            }

        targets = {split: sequence[split][1] for split in SPLITS}
        dates = {split: np.asarray(sequence[split][2]) for split in SPLITS}
        q90_threshold, labels90 = master.labels_for(targets["train"], targets, 0.90)
        q95_threshold, labels95 = master.labels_for(targets["train"], targets, 0.95)

        requested_blocks = sorted(
            {block for model in args.models for block in model_blocks[model]}
        )
        sources = {
            block: sequence_fast if block.startswith("fast") else sequence
            for block in requested_blocks
        }
        exact_blocks: dict[str, dict[str, np.ndarray]] = {}
        exact_seconds: dict[str, float] = {}
        exact_paths: dict[str, str] = {}
        for block in requested_blocks:
            payload = {
                "historical_source": "scripts/canonical/run_master_comparison.py@phase3-refactor",
                "data": str(args.data),
                "fold": fold_id,
                "block": block,
                "config": asdict(block_configs[block]),
                "train_stride": args.train_stride,
            }
            exact_blocks[block], exact_seconds[block], exact_paths[block] = load_or_build_feature_block(
                cache_dir=args.cache_dir,
                cache_payload=payload,
                arrays={split: sources[block][split][0] for split in SPLITS},
                config=block_configs[block],
                force=args.force_recompute,
            )
            cache_manifest.append(
                {
                    "fold": fold_id,
                    "block": block,
                    "protocol": "exact",
                    "path": exact_paths[block],
                }
            )

        def combine_features(
            model: str,
            blocks: dict[str, dict[str, np.ndarray]],
        ) -> dict[str, np.ndarray]:
            names = model_blocks[model]
            return {
                split: np.column_stack([blocks[name][split] for name in names])
                for split in SPLITS
            }

        for model_name in args.models:
            exact_features = combine_features(model_name, exact_blocks)
            if "exact" in args.protocols:
                start = time.perf_counter()
                scores = fit_log_ridge(exact_features, targets, args.ridge_alpha)
                fit_seconds = time.perf_counter() - start
                row, predictions = master.evaluate_model(
                    model_name=model_name,
                    protocol="exact",
                    fold_id=fold_id,
                    y=targets,
                    dates=dates,
                    scores=scores,
                    q90_threshold=q90_threshold,
                    lab90=labels90,
                    q95_threshold=q95_threshold,
                    lab95=labels95,
                    metadata={
                        "quantum_tasks_per_date": len(model_blocks[model_name]),
                        "shots": np.nan,
                        "shot_seed": np.nan,
                        "fit_seconds": fit_seconds,
                        "feature_seconds": sum(
                            exact_seconds[name] for name in model_blocks[model_name]
                        ),
                        "cache_paths": json.dumps(
                            [exact_paths[name] for name in model_blocks[model_name]]
                        ),
                        "input_encoding_status": "historical_two_channel_unworked_up",
                    },
                )
                metric_rows.append(row)
                prediction_rows.extend(predictions)

            noisy_protocols = [protocol for protocol in args.protocols if protocol != "exact"]
            for shot_seed in shot_seeds if noisy_protocols else []:
                noisy_blocks: dict[str, dict[str, np.ndarray]] = {}
                noisy_seconds: dict[str, float] = {}
                for block in model_blocks[model_name]:
                    noisy_config = replace(
                        block_configs[block],
                        shots=args.shots,
                        shot_seed=shot_seed,
                    )
                    payload = {
                        "historical_source": "scripts/canonical/run_master_comparison.py@phase3-refactor",
                        "data": str(args.data),
                        "fold": fold_id,
                        "block": block,
                        "config": asdict(noisy_config),
                        "train_stride": args.train_stride,
                        "split_rng_policy": "independent_deterministic_streams",
                    }
                    noisy_blocks[block], noisy_seconds[block], cache_path = load_or_build_feature_block(
                        cache_dir=args.cache_dir,
                        cache_payload=payload,
                        arrays={split: sources[block][split][0] for split in SPLITS},
                        config=noisy_config,
                        force=args.force_recompute,
                    )
                    cache_manifest.append(
                        {
                            "fold": fold_id,
                            "block": block,
                            "protocol": "finite_shot",
                            "shot_seed": shot_seed,
                            "path": cache_path,
                        }
                    )
                noisy_features = combine_features(model_name, noisy_blocks)
                for protocol in noisy_protocols:
                    if protocol == "exact_train_noisy_test":
                        fit_features = {
                            "train": exact_features["train"],
                            "val": noisy_features["val"],
                            "test": noisy_features["test"],
                        }
                    elif protocol == "noisy_train_noisy_test":
                        fit_features = noisy_features
                    else:
                        raise ValueError(protocol)
                    start = time.perf_counter()
                    scores = fit_log_ridge(fit_features, targets, args.ridge_alpha)
                    fit_seconds = time.perf_counter() - start
                    row, predictions = master.evaluate_model(
                        model_name=model_name,
                        protocol=protocol,
                        fold_id=fold_id,
                        y=targets,
                        dates=dates,
                        scores=scores,
                        q90_threshold=q90_threshold,
                        lab90=labels90,
                        q95_threshold=q95_threshold,
                        lab95=labels95,
                        metadata={
                            "quantum_tasks_per_date": len(model_blocks[model_name]),
                            "shots": args.shots,
                            "shot_seed": shot_seed,
                            "fit_seconds": fit_seconds,
                            "feature_seconds": sum(noisy_seconds.values()),
                            "input_encoding_status": "historical_two_channel_unworked_up",
                        },
                    )
                    metric_rows.append(row)
                    prediction_rows.extend(predictions)

        per_fold = pd.DataFrame(metric_rows)
        predictions = pd.DataFrame(prediction_rows)
        atomic_csv(per_fold, args.out_dir / f"per_fold_metrics_{args.tag}.csv")
        atomic_csv(predictions, args.out_dir / f"predictions_{args.tag}.csv")

    per_fold = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate = master.aggregate_metrics(per_fold)
    per_fold_path = args.out_dir / f"per_fold_metrics_{args.tag}.csv"
    predictions_path = args.out_dir / f"predictions_{args.tag}.csv"
    aggregate_path = args.out_dir / f"aggregate_metrics_{args.tag}.csv"
    manifest_path = args.out_dir / f"run_manifest_{args.tag}.json"
    atomic_csv(per_fold, per_fold_path)
    atomic_csv(predictions, predictions_path)
    atomic_csv(aggregate, aggregate_path)
    manifest_path.write_text(
        json.dumps(
            {
                "historical_source": "scripts/canonical/run_master_comparison.py@phase3-refactor",
                "reservoir_source": "src/qpitome_qrc/qrc/rydberg_reservoir.py@phase3-refactor",
                "models": args.models,
                "protocols": args.protocols,
                "folds": [int(fold["fold"]) for fold in folds],
                "frozen_configuration": {
                    "total_time_us": args.total_time_us,
                    "anchors": args.anchors,
                    "anchor_policy": "even",
                    "reverse_anchors": True,
                    "geometry": "dual_chain",
                    "encoding": "plateau",
                    "level_col": args.level_col,
                    "rate_col": args.rate_col,
                },
                "scientific_status": (
                    "Historical underdeveloped two-channel Rydberg volatility prototype; "
                    "preserved for reproducibility, not presented as a successful mature model."
                ),
                "cache_policy": "feature blocks stored under scratch/canonical_cache",
                "finite_shot_rng_policy": "independent deterministic train/val/test streams",
                "cache_entries": cache_manifest,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(aggregate.to_string(index=False))
    print(f"\nWrote {per_fold_path}")
    print(f"Wrote {predictions_path}")
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
