#!/usr/bin/env python3
"""q95 AP push: readout strategies + feature combinations on cached reservoir features.

Motivation (see docs/phase3 review brief): raise q95 warning AP from ~0.21
toward >= 0.25-0.30 with Aquila-compatible changes. Physics is frozen at the
current best candidate; this script attacks the READOUT and FEATURE-SET side,
where the q95 head is data-starved (~5% positives) and shot noise erodes AP.

Strategies (evaluated per fold on cached features; SELECT ON VAL AP ONLY):
  logistic            current reference: balanced logistic on q95 labels
  logistic_cv         same, inverse-reg strength selected on val AP
  ridge_rank          rank by ridge regression of log target (all samples)
  q90_head            q90-trained logistic scored against q95 labels (2x pos)
  blend               mean of rank-normalized logistic + ridge_rank scores
  combined            raw + temporal features, logistic
  combined_ridge      raw + temporal features, ridge_rank
  combined_memoryless raw + memoryless features (mechanism control)
  multi_lb            temporal(lb=40) + temporal(lb=10) concatenated (HAR-style)
  multi_lb_combined   raw + multi-lookback temporal

Shot-noise robustness (--shots-eval): re-evaluates the top strategies with
Gaussian-approximated 1000-shot noise (var p(1-p)/S per occupation/correlator
feature, clipped to [0,1]), with optional noise-augmented training replicas.
Approximation note: ignores cross-feature sampling covariance (same shots).

Outputs to scratch/ (git-ignored). Non-final until confirmed at stride 1.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.qrc.rydberg_reservoir import (
    RydbergQRCConfig,
    build_rydberg_feature_matrix,
    make_level_rate_sequence_splits,
)
from qpitome_qrc.qrc.tfim_reservoir import select_anchor_indices

TARGET = "future_rv_20d"
SPLIT_NAMES = ("train", "val", "test")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
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
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--shots-eval", type=int, default=1000)
    p.add_argument("--noise-replicas", type=int, default=3)
    p.add_argument("--only-folds", type=int, nargs="*", default=None)
    p.add_argument("--out-dir", type=Path, default=Path("scratch/ap_push"))
    p.add_argument("--tag", default="v1")
    return p.parse_args()


def make_folds(n, *, n_folds, min_train, val_size, purge):
    first_test_start = min_train + val_size + purge
    test_size = (n - first_test_start) // n_folds
    folds = []
    for i in range(n_folds):
        test_start = first_test_start + i * test_size
        test_end = n if i == n_folds - 1 else test_start + test_size
        val_end = test_start - purge
        val_start = val_end - val_size
        folds.append(dict(fold=i + 1, train=(0, val_start), val=(val_start, val_end), test=(test_start, test_end)))
    return folds


def raw_features(X: np.ndarray, anchor_idx: np.ndarray) -> np.ndarray:
    anchors = X[:, anchor_idx, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(np.column_stack([w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]))
    return np.column_stack([anchors] + stats)


def labels_for(y: dict[str, np.ndarray], q: float) -> dict[str, np.ndarray]:
    thr = float(np.quantile(y["train"], q))
    return {k: (v >= thr).astype(int) for k, v in y.items()}


def fit_logistic(H_tr, lab_tr, C=1.0):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", C=C, random_state=42),
    )
    clf.fit(H_tr, lab_tr)
    return clf


def rank_normalize(s: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(s))
    return order / max(len(s) - 1, 1)


def ap_auc(lab, score):
    if lab.sum() == 0 or lab.sum() == len(lab):
        return np.nan, np.nan
    return float(average_precision_score(lab, score)), float(roc_auc_score(lab, score))


def shot_noise(H: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Gaussian approximation of multinomial shot noise on probability features."""
    p = np.clip(H, 0.0, 1.0)
    noisy = p + rng.normal(0.0, np.sqrt(p * (1 - p) / shots))
    return np.clip(noisy, 0.0, 1.0)


def evaluate_scores(scores: dict[str, np.ndarray], lab: dict[str, np.ndarray]) -> dict:
    out = {}
    for split in ("val", "test"):
        ap, auc = ap_auc(lab[split], scores[split])
        out[f"{split}_ap"], out[f"{split}_auc"] = ap, auc
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data).sort_values("date").reset_index(drop=True)
    folds = make_folds(len(df), n_folds=args.n_folds, min_train=args.min_train, val_size=args.val_size, purge=args.purge)

    base = RydbergQRCConfig(
        n_atoms_slow=args.n_slow, n_atoms_fast=args.n_fast,
        spacing_slow_um=args.spacing_slow_um, spacing_fast_um=args.spacing_fast_um,
        row_gap_um=args.row_gap_um,
        lookback_days=args.lookback, anchor_count=args.anchors, anchor_policy="even",
        reverse_anchors=True, total_time_us=args.total_time_us,
    )
    fast_cfg = replace(base, lookback_days=args.lookback_fast)
    memless = replace(base, memory_mode="memoryless")
    fast_memless = replace(fast_cfg, memory_mode="memoryless")
    shuffled = replace(base, shuffle_anchors=True)
    fast_shuffled = replace(fast_cfg, shuffle_anchors=True)
    anchor_idx = select_anchor_indices(args.lookback, args.anchors, "even")

    rows = []
    rng = np.random.default_rng(202606)

    for f in folds:
        fold_id = f["fold"]
        if args.only_folds and fold_id not in args.only_folds:
            continue
        frames = {k: df.iloc[f[k][0]:f[k][1]].reset_index(drop=True) for k in SPLIT_NAMES}
        seq = make_level_rate_sequence_splits(frames, level_col=args.level_col, rate_col=args.rate_col,
                                              target_column=TARGET, lookback_days=args.lookback)
        seq_fast = make_level_rate_sequence_splits(frames, level_col=args.level_col, rate_col=args.rate_col,
                                                   target_column=TARGET, lookback_days=args.lookback_fast)
        # Align fast-lookback samples to slow-lookback dates (fast keeps more
        # head samples; drop them so both end-align 1:1).
        seq_fast = {
            k: (X[len(X) - len(seq[k][0]):], y[len(y) - len(seq[k][1]):], d[len(d) - len(seq[k][2]):])
            for k, (X, y, d) in seq_fast.items()
        }
        if args.stride > 1:
            s = args.stride
            # Stride the TRAIN split only (compute lives there); keep val/test
            # at full resolution so q95 label counts match the walk-forward.
            seq = {k: ((X[::s], y[::s], d[::s]) if k == "train" else (X, y, d)) for k, (X, y, d) in seq.items()}
            seq_fast = {k: ((X[::s], y[::s], d[::s]) if k == "train" else (X, y, d)) for k, (X, y, d) in seq_fast.items()}
        y = {k: v[1] for k, v in seq.items()}
        lab90, lab95 = labels_for(y, 0.90), labels_for(y, 0.95)
        if lab95["test"].sum() == 0:
            print(f"fold {fold_id}: degenerate q95 TEST labels, skipping")
            continue
        # NOTE: q95 VAL labels can be degenerate (calm val windows vs train
        # quantile) — folds 1 and 4. Selection therefore uses q90 val AP,
        # which is always populated and highly correlated with the q95 task.
        print(f"=== fold {fold_id} (train {len(y['train'])}, val {len(y['val'])}, test {len(y['test'])}, "
              f"val/test q95 pos {lab95['val'].sum()}/{lab95['test'].sum()}) ===")

        # --- cached feature builds (expensive part, once per fold) ---
        H_temporal = {k: build_rydberg_feature_matrix(seq[k][0], base) for k in SPLIT_NAMES}
        H_fast = {k: build_rydberg_feature_matrix(seq_fast[k][0], fast_cfg) for k in SPLIT_NAMES}
        H_memless = {k: build_rydberg_feature_matrix(seq[k][0], memless) for k in SPLIT_NAMES}
        H_fast_memless = {k: build_rydberg_feature_matrix(seq_fast[k][0], fast_memless) for k in SPLIT_NAMES}
        H_shuffled = {k: build_rydberg_feature_matrix(seq[k][0], shuffled) for k in SPLIT_NAMES}
        H_fast_shuffled = {k: build_rydberg_feature_matrix(seq_fast[k][0], fast_shuffled) for k in SPLIT_NAMES}

        H_raw = {k: raw_features(seq[k][0], anchor_idx) for k in SPLIT_NAMES}

        H_comb = {k: np.column_stack([H_raw[k], H_temporal[k]]) for k in SPLIT_NAMES}
        H_comb_ml = {k: np.column_stack([H_raw[k], H_memless[k]]) for k in SPLIT_NAMES}
        H_comb_shuf = {k: np.column_stack([H_raw[k], H_shuffled[k]]) for k in SPLIT_NAMES}

        H_multi = {k: np.column_stack([H_temporal[k], H_fast[k]]) for k in SPLIT_NAMES}
        H_multi_memless = {k: np.column_stack([H_memless[k], H_fast_memless[k]]) for k in SPLIT_NAMES}
        H_multi_shuffled = {k: np.column_stack([H_shuffled[k], H_fast_shuffled[k]]) for k in SPLIT_NAMES}
        H_multi_comb = {k: np.column_stack([H_raw[k], H_multi[k]]) for k in SPLIT_NAMES}

        y_log = {k: np.log(np.clip(v, 1e-12, None)) for k, v in y.items()}

        def ridge_scores(H):
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            model.fit(H["train"], y_log["train"])
            return {k: model.predict(H[k]) for k in SPLIT_NAMES}

        def logi_scores(H, lab, C=1.0):
            clf = fit_logistic(H["train"], lab["train"], C=C)
            return {k: clf.predict_proba(H[k])[:, 1] for k in SPLIT_NAMES}

        strategies: dict[str, dict[str, np.ndarray]] = {}
        strategies["logistic"] = logi_scores(H_temporal, lab95)
        # val-selected C
        best_C, best_ap = 1.0, -np.inf
        best_sc = strategies["logistic"]
        for C in (0.03, 0.1, 0.3, 1.0, 3.0):
            sc = logi_scores(H_temporal, lab95, C=C)
            # q95 val can be degenerate (folds 1/4): select C on q90 val AP then.
            ap, _ = ap_auc(lab95["val"], sc["val"])
            if np.isnan(ap):
                ap, _ = ap_auc(lab90["val"], sc["val"])
            if np.isfinite(ap) and ap > best_ap:
                best_ap, best_C, best_sc = ap, C, sc
        strategies["logistic_cv"] = best_sc
        strategies["ridge_rank"] = ridge_scores(H_temporal)
        strategies["q90_head"] = logi_scores(H_temporal, lab90)
        lg, rr = strategies["logistic"], strategies["ridge_rank"]
        strategies["blend"] = {k: 0.5 * rank_normalize(lg[k]) + 0.5 * rank_normalize(rr[k]) for k in SPLIT_NAMES}
        strategies["combined"] = logi_scores(H_comb, lab95)
        strategies["combined_ridge"] = ridge_scores(H_comb)
        strategies["combined_memoryless"] = logi_scores(H_comb_ml, lab95)
        strategies["raw_ridge"] = ridge_scores(H_raw)

        strategies["memoryless_ridge"] = ridge_scores(H_memless)
        strategies["shuffled_ridge"] = ridge_scores(H_shuffled)

        strategies["combined_memoryless_ridge"] = ridge_scores(H_comb_ml)
        strategies["combined_shuffled_ridge"] = ridge_scores(H_comb_shuf)

        strategies["multi_lb_ridge"] = ridge_scores(H_multi)
        strategies["multi_lb_memoryless_ridge"] = ridge_scores(H_multi_memless)
        strategies["multi_lb_shuffled_ridge"] = ridge_scores(H_multi_shuffled)

        strategies["multi_lb"] = logi_scores(H_multi, lab95)
        strategies["multi_lb_combined"] = logi_scores(H_multi_comb, lab95)
        cg, cr = strategies["combined"], strategies["combined_ridge"]
        strategies["combined_blend"] = {k: 0.5 * rank_normalize(cg[k]) + 0.5 * rank_normalize(cr[k]) for k in SPLIT_NAMES}

        for name, sc in strategies.items():
            res = evaluate_scores(sc, lab95)
            q90res = evaluate_scores(sc, lab90)
            rows.append({"fold": fold_id, "strategy": name, "noise": "exact",
                         **res, "q90_val_ap": q90res["val_ap"], "q90_test_ap": q90res["test_ap"]})

        # --- finite-shot robustness for the reservoir-feature strategies ---
        if args.shots_eval:
            def noisy(H):
                return {k: shot_noise(H[k], args.shots_eval, rng) for k in SPLIT_NAMES}

            def ridge_noisy(H, Hn):
                model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                model.fit(Hn["train"], y_log["train"])
                return {k: model.predict(Hn[k]) for k in SPLIT_NAMES}

            for name, H in (
                ("combined_ridge", H_comb),
                ("combined_memoryless_ridge", H_comb_ml),
                ("combined_shuffled_ridge", H_comb_shuf),
                ("multi_lb_ridge", H_multi),
                ("multi_lb_memoryless_ridge", H_multi_memless),
                ("multi_lb_shuffled_ridge", H_multi_shuffled),
            ):
                # raw block is classical -> noise applies only to reservoir block
                n_raw = H_raw["train"].shape[1] if name.startswith("combined_") else 0
                Hn = {}
                for k in SPLIT_NAMES:
                    Hk = H[k].copy()
                    Hk[:, n_raw:] = shot_noise(Hk[:, n_raw:], args.shots_eval, rng)
                    Hn[k] = Hk
                sc = ridge_noisy(H, Hn)
                rows.append({"fold": fold_id, "strategy": name, "noise": f"shots{args.shots_eval}",
                             **evaluate_scores(sc, lab95),
                             "q90_val_ap": evaluate_scores(sc, lab90)["val_ap"],
                             "q90_test_ap": evaluate_scores(sc, lab90)["test_ap"]})

            for name, H in (("logistic", H_temporal), ("combined", H_comb), ("multi_lb", H_multi)):
                Hn = noisy(H)
                sc = logi_scores(Hn, lab95)
                rows.append({"fold": fold_id, "strategy": name, "noise": f"shots{args.shots_eval}",
                             **evaluate_scores(sc, lab95)})
                # noise-aware training: augment train with noisy replicas
                H_aug_tr = np.vstack([Hn["train"]] + [shot_noise(H["train"], args.shots_eval, rng)
                                                      for _ in range(args.noise_replicas)])
                lab_aug = np.concatenate([lab95["train"]] * (1 + args.noise_replicas))
                clf = fit_logistic(H_aug_tr, lab_aug)
                sc = {k: clf.predict_proba(Hn[k])[:, 1] for k in SPLIT_NAMES}
                rows.append({"fold": fold_id, "strategy": name, "noise": f"shots{args.shots_eval}+aug",
                             **evaluate_scores(sc, lab95)})

    out = pd.DataFrame(rows)
    csv = args.out_dir / f"ap_push_{args.tag}.csv"
    out.to_csv(csv, index=False)

    med = out.groupby(["strategy", "noise"])[["q90_val_ap", "val_ap", "test_ap", "q90_test_ap", "test_auc"]].median().round(4)
    med = med.sort_values("q90_val_ap", ascending=False)
    print("\n=== Medians (RANKED BY q90 VAL AP; val_ap/test_ap are q95) ===")
    print(med.to_string())
    (args.out_dir / f"ap_push_{args.tag}_summary.json").write_text(
        json.dumps({"tag": args.tag, "median": med.reset_index().to_dict(orient="records")}, indent=2))
    print(f"\nWrote {csv}")


if __name__ == "__main__":
    main()
