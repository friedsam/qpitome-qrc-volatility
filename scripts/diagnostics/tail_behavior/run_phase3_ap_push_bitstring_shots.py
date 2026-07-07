#!/usr/bin/env python3
"""Physically faithful finite-shot AP robustness for Phase 3 Rydberg features.

This runner replaces independent Gaussian feature perturbations with the
repository's existing finite-shot path:

    exact state probabilities -> multinomial bitstring counts
    -> occupations and pair correlators estimated from the same shots

That preserves the shared-shot covariance structure among observables.

It compares two deployment protocols:

1. noisy_train_noisy_test
   Finite-shot train/val/test features; fit ridge on noisy train features.

2. exact_train_noisy_test
   Fit scaler/ridge on exact simulator train features; apply the frozen
   readout to finite-shot val/test features. This is closer to a limited-budget
   simulator-to-hardware transfer workflow.

Exact references are emitted once per fold. Finite-shot results are repeated
across independent shot seeds.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)
from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    build_rydberg_feature_matrix,
    make_level_rate_sequence_splits,
)
from qpitome_qrc.qrc.tfim_reservoir import select_anchor_indices

TARGET = "future_rv_20d"
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_STRATEGIES = (
    "multi_lb_ridge",
    "multi_lb_memoryless_ridge",
    "multi_lb_shuffled_ridge",
    "combined_ridge",
    "combined_memoryless_ridge",
    "combined_shuffled_ridge",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("scratch/ap_push_bitstring"))
    p.add_argument("--tag", default="shots1000_true_bitstring")
    p.add_argument("--only-folds", nargs="*", type=int, default=[1, 2, 4, 5])
    p.add_argument("--strategies", nargs="*", default=list(DEFAULT_STRATEGIES))
    p.add_argument("--shot-seeds", default="7,17,27,37,47")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--stride", type=int, default=1, help="Stride TRAIN only; val/test remain full resolution")

    p.add_argument("--level-col", default="vix_rv_spread")
    p.add_argument("--rate-col", default="rv_accel_log_5_20")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--lookback-fast", type=int, default=10)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)

    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--total-time-us", type=float, default=0.55)
    p.add_argument("--n-slow", type=int, default=4)
    p.add_argument("--n-fast", type=int, default=4)
    p.add_argument("--spacing-slow-um", type=float, default=9.0)
    p.add_argument("--spacing-fast-um", type=float, default=15.0)
    p.add_argument("--row-gap-um", type=float, default=12.0)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    return p.parse_args()


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    """Backward-compatible wrapper around the shared Phase 3 protocol."""
    return make_purged_walkforward_folds(
        n,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )


def raw_features(X: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))
    return np.column_stack([anchors] + stats)


def labels_for(y: dict[str, np.ndarray], q: float) -> dict[str, np.ndarray]:
    threshold = float(np.quantile(y["train"], q))
    return {split: (values >= threshold).astype(int) for split, values in y.items()}


def ap_auc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan, np.nan
    return float(average_precision_score(labels, scores)), float(roc_auc_score(labels, scores))


def fit_ridge_scores(
    H_train: np.ndarray,
    H_val: np.ndarray,
    H_test: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(H_train, np.log(np.clip(y_train, 1e-12, None)))
    return {
        "train": model.predict(H_train),
        "val": model.predict(H_val),
        "test": model.predict(H_test),
    }


def required_blocks(strategies: list[str]) -> set[str]:
    out: set[str] = set()
    for strategy in strategies:
        if strategy in {"multi_lb_ridge", "combined_ridge"}:
            out.add("temporal")
        if strategy == "multi_lb_ridge":
            out.add("fast")
        if strategy in {"multi_lb_memoryless_ridge", "combined_memoryless_ridge"}:
            out.add("memoryless")
        if strategy == "multi_lb_memoryless_ridge":
            out.add("fast_memoryless")
        if strategy in {"multi_lb_shuffled_ridge", "combined_shuffled_ridge"}:
            out.add("shuffled")
        if strategy == "multi_lb_shuffled_ridge":
            out.add("fast_shuffled")
    unknown = set(strategies) - set(DEFAULT_STRATEGIES)
    if unknown:
        raise ValueError(f"Unknown strategies: {sorted(unknown)}")
    return out


def strategy_features(
    strategy: str,
    blocks: dict[str, dict[str, np.ndarray]],
    raw: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if strategy == "multi_lb_ridge":
        return {k: np.column_stack([blocks["temporal"][k], blocks["fast"][k]]) for k in SPLIT_NAMES}
    if strategy == "multi_lb_memoryless_ridge":
        return {k: np.column_stack([blocks["memoryless"][k], blocks["fast_memoryless"][k]]) for k in SPLIT_NAMES}
    if strategy == "multi_lb_shuffled_ridge":
        return {k: np.column_stack([blocks["shuffled"][k], blocks["fast_shuffled"][k]]) for k in SPLIT_NAMES}
    if strategy == "combined_ridge":
        return {k: np.column_stack([raw[k], blocks["temporal"][k]]) for k in SPLIT_NAMES}
    if strategy == "combined_memoryless_ridge":
        return {k: np.column_stack([raw[k], blocks["memoryless"][k]]) for k in SPLIT_NAMES}
    if strategy == "combined_shuffled_ridge":
        return {k: np.column_stack([raw[k], blocks["shuffled"][k]]) for k in SPLIT_NAMES}
    raise ValueError(strategy)


def derive_seed(base_seed: int, block_index: int, split_index: int) -> int:
    # Separate RNG streams for separate reservoir blocks and split evaluations.
    return int(base_seed + 100_000 * block_index + 1_000 * split_index)


def build_blocks(
    seq: dict,
    seq_fast: dict,
    configs: dict[str, RydbergQRCConfig],
    names: set[str],
    *,
    shots: int | None,
    base_seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    split_index = {name: i for i, name in enumerate(SPLIT_NAMES)}

    for block_index, name in enumerate(sorted(names)):
        cfg = configs[name]
        source = seq_fast if name.startswith("fast") else seq
        out[name] = {}
        for split in SPLIT_NAMES:
            use_cfg = replace(
                cfg,
                shots=shots,
                shot_seed=derive_seed(base_seed, block_index, split_index[split]),
            )
            out[name][split] = build_rydberg_feature_matrix(source[split][0], use_cfg)
    return out


def add_result_rows(
    rows: list[dict],
    *,
    fold_id: int,
    strategy: str,
    protocol: str,
    shot_seed: int | None,
    shots: int | None,
    scores: dict[str, np.ndarray],
    lab90: dict[str, np.ndarray],
    lab95: dict[str, np.ndarray],
) -> None:
    q90_val_ap, q90_val_auc = ap_auc(lab90["val"], scores["val"])
    q90_test_ap, q90_test_auc = ap_auc(lab90["test"], scores["test"])
    q95_val_ap, q95_val_auc = ap_auc(lab95["val"], scores["val"])
    q95_test_ap, q95_test_auc = ap_auc(lab95["test"], scores["test"])
    rows.append(
        {
            "fold": fold_id,
            "strategy": strategy,
            "protocol": protocol,
            "noise_model": "exact" if shots is None else "multinomial_bitstring",
            "shots": shots,
            "shot_seed": shot_seed,
            "q90_val_ap": q90_val_ap,
            "q90_val_auc": q90_val_auc,
            "q90_test_ap": q90_test_ap,
            "q90_test_auc": q90_test_auc,
            "q95_val_ap": q95_val_ap,
            "q95_val_auc": q95_val_auc,
            "q95_test_ap": q95_test_ap,
            "q95_test_auc": q95_test_auc,
            "q90_val_n_pos": int(lab90["val"].sum()),
            "q90_test_n_pos": int(lab90["test"].sum()),
            "q95_val_n_pos": int(lab95["val"].sum()),
            "q95_test_n_pos": int(lab95["test"].sum()),
        }
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shot_seeds = [int(x) for x in args.shot_seeds.split(",") if x.strip()]
    strategies = list(args.strategies)
    needed = required_blocks(strategies)

    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    folds = make_folds(
        len(df),
        n_folds=args.n_folds,
        min_train=args.min_train,
        val_size=args.val_size,
        purge=args.purge,
    )
    if args.only_folds:
        keep = set(args.only_folds)
        folds = [f for f in folds if f["fold"] in keep]

    base = RydbergQRCConfig(
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
    )
    fast = replace(base, lookback_days=args.lookback_fast)
    configs = {
        "temporal": base,
        "fast": fast,
        "memoryless": replace(base, memory_mode="memoryless"),
        "fast_memoryless": replace(fast, memory_mode="memoryless"),
        "shuffled": replace(base, shuffle_anchors=True),
        "fast_shuffled": replace(fast, shuffle_anchors=True),
    }

    anchor_idx = select_anchor_indices(args.lookback, args.anchors, "even")
    rows: list[dict] = []

    for f in folds:
        fold_id = f["fold"]
        frames = slice_fold_frames(df, f)
        seq = make_level_rate_sequence_splits(
            frames,
            level_col=args.level_col,
            rate_col=args.rate_col,
            target_column=TARGET,
            lookback_days=args.lookback,
        )
        seq_fast = make_level_rate_sequence_splits(
            frames,
            level_col=args.level_col,
            rate_col=args.rate_col,
            target_column=TARGET,
            lookback_days=args.lookback_fast,
        )
        seq_fast = {
            k: (X[len(X) - len(seq[k][0]) :], y[len(y) - len(seq[k][1]) :], d[len(d) - len(seq[k][2]) :])
            for k, (X, y, d) in seq_fast.items()
        }

        if args.stride > 1:
            s = args.stride
            seq = {k: ((X[::s], y[::s], d[::s]) if k == "train" else (X, y, d)) for k, (X, y, d) in seq.items()}
            seq_fast = {k: ((X[::s], y[::s], d[::s]) if k == "train" else (X, y, d)) for k, (X, y, d) in seq_fast.items()}

        y = {k: seq[k][1] for k in SPLIT_NAMES}
        lab90 = labels_for(y, 0.90)
        lab95 = labels_for(y, 0.95)
        if lab95["test"].sum() == 0:
            print(f"fold {fold_id}: degenerate q95 test labels; skipping")
            continue

        print(
            f"=== fold {fold_id} "
            f"(train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])}, "
            f"val/test q95 pos {int(lab95['val'].sum())}/{int(lab95['test'].sum())}) ==="
        )

        raw = {k: raw_features(seq[k][0], anchor_idx) for k in SPLIT_NAMES}

        print("  building exact reservoir blocks")
        exact_blocks = build_blocks(seq, seq_fast, configs, needed, shots=None, base_seed=0)
        exact_features = {strategy: strategy_features(strategy, exact_blocks, raw) for strategy in strategies}

        for strategy, H in exact_features.items():
            scores = fit_ridge_scores(H["train"], H["val"], H["test"], y["train"], args.ridge_alpha)
            add_result_rows(
                rows,
                fold_id=fold_id,
                strategy=strategy,
                protocol="exact",
                shot_seed=None,
                shots=None,
                scores=scores,
                lab90=lab90,
                lab95=lab95,
            )

        for seed in shot_seeds:
            print(f"  true bitstring shots={args.shots}, seed={seed}")
            noisy_blocks = build_blocks(seq, seq_fast, configs, needed, shots=args.shots, base_seed=seed)
            noisy_features = {strategy: strategy_features(strategy, noisy_blocks, raw) for strategy in strategies}

            for strategy in strategies:
                H_exact = exact_features[strategy]
                H_noisy = noisy_features[strategy]

                scores_transfer = fit_ridge_scores(
                    H_exact["train"],
                    H_noisy["val"],
                    H_noisy["test"],
                    y["train"],
                    args.ridge_alpha,
                )
                add_result_rows(
                    rows,
                    fold_id=fold_id,
                    strategy=strategy,
                    protocol="exact_train_noisy_test",
                    shot_seed=seed,
                    shots=args.shots,
                    scores=scores_transfer,
                    lab90=lab90,
                    lab95=lab95,
                )

                scores_noise_aware = fit_ridge_scores(
                    H_noisy["train"],
                    H_noisy["val"],
                    H_noisy["test"],
                    y["train"],
                    args.ridge_alpha,
                )
                add_result_rows(
                    rows,
                    fold_id=fold_id,
                    strategy=strategy,
                    protocol="noisy_train_noisy_test",
                    shot_seed=seed,
                    shots=args.shots,
                    scores=scores_noise_aware,
                    lab90=lab90,
                    lab95=lab95,
                )

    result = pd.DataFrame(rows)
    out_csv = args.out_dir / f"ap_push_bitstring_{args.tag}.csv"
    result.to_csv(out_csv, index=False)

    noisy = result[result["protocol"] != "exact"].copy()
    per_seed_fold_median = (
        noisy.groupby(["strategy", "protocol", "shot_seed"])[["q90_test_ap", "q95_test_ap", "q95_test_auc"]]
        .median(numeric_only=True)
        .reset_index()
    )
    aggregate = (
        per_seed_fold_median.groupby(["strategy", "protocol"])[["q90_test_ap", "q95_test_ap", "q95_test_auc"]]
        .agg(["median", "mean", "std", "min", "max"])
    )

    exact_summary = (
        result[result["protocol"] == "exact"]
        .groupby("strategy")[["q90_test_ap", "q95_test_ap", "q95_test_auc"]]
        .median(numeric_only=True)
    )

    summary = {
        "tag": args.tag,
        "shots": args.shots,
        "shot_seeds": shot_seeds,
        "folds": [int(f["fold"]) for f in folds],
        "strategies": strategies,
        "noise_model": "true multinomial bitstring sampling through build_rydberg_feature_matrix",
        "protocols": ["exact", "exact_train_noisy_test", "noisy_train_noisy_test"],
        "hardware_cost_note": "multi_lb strategies require two reservoir tasks per market date; combined strategies require one reservoir task plus classical raw features",
    }
    out_json = args.out_dir / f"ap_push_bitstring_{args.tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print("\n=== Exact reference: median across folds ===")
    print(exact_summary.sort_values("q95_test_ap", ascending=False).to_string())
    print("\n=== True bitstring finite-shot: seed-distribution summary ===")
    print(aggregate.to_string())
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
