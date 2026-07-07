#!/usr/bin/env python3
"""True temporal Rydberg reservoir experiment.

Runs six variants on identical level/rate inputs so any difference is
attributable to the reservoir transformation, not the inputs:

  raw_baseline               anchor values + window summaries, no quantum dynamics
  rydberg_temporal           one state evolved through the whole drive waveform
  rydberg_memoryless         fresh state per anchor (honest response-grid analog)
  rydberg_shuffled           temporal, but anchor order permuted (order control)
  rydberg_constant_omega     temporal, but rate channel is disabled
  rydberg_omega_zero         temporal with transverse drive disabled

Readout heads per variant:
  regression  ridge on log future_rv_20d  -> RMSE / QLIKE / MZ-R2
  q90 / q95   logistic warning heads      -> AUC / AP / F1 / P / R
  (labels from TRAIN-quantiles of future_rv_20d, as in earlier probes)

Outputs go to scratch/ by default (git-ignored); promote intentionally.

Interpretation guide:
  temporal > memoryless       -> temporal quantum memory adds value
  temporal ~ shuffled         -> reservoir is not using temporal order
  temporal <= raw             -> reservoir transformation not (yet) useful
  temporal <= constant_omega  -> rate-to-Omega encoding is not helping
  temporal <= omega_zero      -> transverse-drive dynamics are not helping

Note: the raw baseline here is deliberately minimal. Repo ESN / classical
baselines remain the external comparison; do not claim advantage from this
script alone, and evaluate under purged walk-forward CV before any claims.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.qrc.rydberg_reservoir import (
    C6_RAD_UM6_PER_US,
    RydbergQRCConfig,
    fit_rydberg_qrc_regressor,
    make_level_rate_sequence_splits,
    summarize_rydberg_result,
    validate_aquila_feasibility,
)
from qpitome_qrc.qrc.tfim_reservoir import select_anchor_indices

TARGET = "future_rv_20d"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--level-col", default="vix_rv_spread")
    p.add_argument("--rate-col", default="rv_accel_log_5_20")
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--anchors", type=int, default=6)
    p.add_argument("--anchor-policy", choices=("even", "recent"), default="even")
    p.add_argument("--reverse-anchors", action="store_true", help="Inject selected anchors newest-to-oldest")
    p.add_argument("--shuffle-seed", type=int, default=1234, help="Seed for the shuffled-anchor control")
    p.add_argument("--total-time-us", type=float, default=1.8)
    p.add_argument("--n-slow", type=int, default=4)
    p.add_argument("--n-fast", type=int, default=4)
    p.add_argument("--spacing-slow-um", type=float, default=9.0)
    p.add_argument("--spacing-fast-um", type=float, default=15.0)
    p.add_argument("--delta-center", type=float, default=6.0)
    p.add_argument("--delta-span", type=float, default=4.0)
    p.add_argument("--omega-base", type=float, default=6.0)
    p.add_argument("--omega-mod-frac", type=float, default=0.5)
    p.add_argument("--shots", type=int, default=None, help="None = exact expectations")
    p.add_argument("--stride", type=int, default=1, help="Subsample windows for smoke runs")
    p.add_argument("--out-dir", type=Path, default=Path("scratch/rydberg_temporal_reservoir"))
    p.add_argument("--tag", default="v1")
    p.add_argument("--verbose", action="store_true", help="Print per-segment reservoir progress")
    return p.parse_args()


def raw_anchor_indices(args: argparse.Namespace) -> np.ndarray:
    """Anchor indices for the raw baseline under the same exposed order policy."""
    anchor_idx = select_anchor_indices(args.lookback, args.anchors, args.anchor_policy)
    if args.reverse_anchors:
        anchor_idx = anchor_idx[::-1]
    return np.asarray(anchor_idx, dtype=int)


def raw_baseline_features(X: np.ndarray, anchor_indices: np.ndarray) -> np.ndarray:
    """Anchor values of both channels + per-channel window summaries."""
    anchors = X[:, anchor_indices, :].reshape(len(X), -1)
    stats = []
    for ch in range(X.shape[2]):
        w = X[:, :, ch]
        stats.append(
            np.column_stack(
                [w[:, -1], w.mean(1), w.std(1), w.min(1), w.max(1)]
            )
        )
    return np.column_stack([anchors] + stats)


def phase_budget(config: RydbergQRCConfig) -> dict[str, float | None]:
    """Dimensionless per-segment phase diagnostics for interpretable tuning."""
    k = int(config.anchor_count)
    t_seg = float(config.total_time_us / k)
    delta_abs_max = float(abs(config.delta_center_rad_us) + abs(config.delta_span_rad_us))

    if config.omega_mode == "constant":
        omega_min = omega_max = float(config.omega_base_rad_us)
    else:
        omega_min = float(max(config.omega_base_rad_us * (1.0 - abs(config.omega_mod_frac)), 0.0))
        omega_max = float(config.omega_base_rad_us * (1.0 + abs(config.omega_mod_frac)))

    if config.geometry == "dual_chain":
        v_slow = float(C6_RAD_UM6_PER_US / config.spacing_slow_um**6)
        v_fast = float(C6_RAD_UM6_PER_US / config.spacing_fast_um**6)
    else:
        v_slow = float(C6_RAD_UM6_PER_US / config.chain_spacing_um**6)
        v_fast = None

    out: dict[str, float | None] = {
        "t_segment_us": t_seg,
        "omega_min_rad_us": omega_min,
        "omega_max_rad_us": omega_max,
        "omega_min_t_segment": omega_min * t_seg,
        "omega_max_t_segment": omega_max * t_seg,
        "delta_abs_max_rad_us": delta_abs_max,
        "delta_abs_max_t_segment": delta_abs_max * t_seg,
        "v_slow_nn_rad_us": v_slow,
        "v_slow_nn_t_segment": v_slow * t_seg,
        "v_fast_nn_rad_us": v_fast,
        "v_fast_nn_t_segment": None if v_fast is None else v_fast * t_seg,
    }
    return out


def warning_head(
    H: dict[str, np.ndarray],
    y: dict[str, np.ndarray],
    quantile: float,
    seed: int = 42,
) -> dict:
    """Train-quantile labels + logistic readout, mirroring earlier probes."""
    threshold = float(np.quantile(y["train"], quantile))
    labels = {k: (v >= threshold).astype(int) for k, v in y.items()}
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
    )
    clf.fit(H["train"], labels["train"])

    out: dict = {"threshold": threshold, "train_positive_rate": float(labels["train"].mean())}
    for split in ("val", "test"):
        yl = labels[split]
        if yl.sum() == 0 or yl.sum() == len(yl):
            out[split] = {"note": "degenerate labels", "n_pos": int(yl.sum())}
            continue
        score = clf.predict_proba(H[split])[:, 1]
        pred = (score >= 0.5).astype(int)
        out[split] = {
            "n": int(len(yl)),
            "n_pos": int(yl.sum()),
            "auc": float(roc_auc_score(yl, score)),
            "ap": float(average_precision_score(yl, score)),
            "f1": float(f1_score(yl, pred, zero_division=0)),
            "precision": float(precision_score(yl, pred, zero_division=0)),
            "recall": float(recall_score(yl, pred, zero_division=0)),
        }
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    splits = chronological_tabular_split(df)
    seq = make_level_rate_sequence_splits(
        splits,
        level_col=args.level_col,
        rate_col=args.rate_col,
        target_column=TARGET,
        lookback_days=args.lookback,
    )
    if args.stride > 1:
        seq = {
            k: (X[:: args.stride], y[:: args.stride], d[:: args.stride].reset_index(drop=True))
            for k, (X, y, d) in seq.items()
        }
    y = {k: v[1] for k, v in seq.items()}
    print({k: len(v) for k, v in y.items()})

    base_config = RydbergQRCConfig(
        n_atoms_slow=args.n_slow,
        n_atoms_fast=args.n_fast,
        spacing_slow_um=args.spacing_slow_um,
        spacing_fast_um=args.spacing_fast_um,
        lookback_days=args.lookback,
        anchor_count=args.anchors,
        anchor_policy=args.anchor_policy,
        reverse_anchors=args.reverse_anchors,
        shuffle_seed=args.shuffle_seed,
        total_time_us=args.total_time_us,
        delta_center_rad_us=args.delta_center,
        delta_span_rad_us=args.delta_span,
        omega_base_rad_us=args.omega_base,
        omega_mod_frac=args.omega_mod_frac,
        shots=args.shots,
    )
    feasibility = validate_aquila_feasibility(base_config)
    print("Aquila feasibility:", feasibility["feasible"], feasibility["checks"])

    variants: dict[str, RydbergQRCConfig | None] = {
        "raw_baseline": None,
        "rydberg_temporal": base_config,
        "rydberg_memoryless": replace(base_config, memory_mode="memoryless"),
        "rydberg_shuffled": replace(base_config, shuffle_anchors=True),
        "rydberg_constant_omega": replace(base_config, omega_mode="constant"),
        "rydberg_omega_zero": replace(
            base_config,
            omega_base_rad_us=0.0,
            omega_mod_frac=0.0,
        ),
    }

    anchor_idx = raw_anchor_indices(args)
    summary: dict = {
        "tag": args.tag,
        "target": TARGET,
        "level_col": args.level_col,
        "rate_col": args.rate_col,
        "stride": args.stride,
        "anchor_policy": args.anchor_policy,
        "reverse_anchors": args.reverse_anchors,
        "shuffle_seed": args.shuffle_seed,
        "feasibility": feasibility,
        "base_phase_budget": phase_budget(base_config),
        "variants": {},
    }
    rows = []

    for name, config in variants.items():
        print(f"\n=== {name} ===")
        if config is None:
            H = {k: raw_baseline_features(seq[k][0], anchor_idx) for k in seq}
            reg_row = {"model": name, "n_reservoir_features": H["train"].shape[1]}
            variant_phase_budget = None
            # Ridge regression head on the same features for comparability.
            from qpitome_qrc.qrc.tfim_reservoir import fit_qrc_readout, predict_qrc_readout
            from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast

            readout, scaler = fit_qrc_readout(H["train"], y["train"], config=base_config)
            for split in ("train", "val", "test"):
                pred = predict_qrc_readout(readout, scaler, H[split], config=base_config)
                m = evaluate_volatility_forecast(y[split], pred)
                reg_row[f"{split}_rmse"] = m.rmse
                reg_row[f"{split}_qlike"] = m.qlike
                reg_row[f"{split}_mz_r2"] = m.mz_r2
        else:
            result = fit_rydberg_qrc_regressor(seq, config=config, target=TARGET, verbose=args.verbose)
            reg_row = summarize_rydberg_result(result)
            reg_row["model"] = name
            variant_phase_budget = phase_budget(config)
            for key, value in variant_phase_budget.items():
                reg_row[f"phase_{key}"] = value
            H = {
                "train": result.train_features,
                "val": result.val_features,
                "test": result.test_features,
            }

        variant_summary = {
            "regression": {
                k: reg_row[k]
                for k in reg_row
                if any(k.startswith(s) for s in ("train_", "val_", "test_"))
            },
            "n_features": int(reg_row["n_reservoir_features"]),
            "phase_budget": variant_phase_budget,
            "q90_warning": warning_head(H, y, 0.90),
            "q95_crisis": warning_head(H, y, 0.95),
        }
        summary["variants"][name] = variant_summary
        rows.append(reg_row)

        for task in ("q90_warning", "q95_crisis"):
            t = variant_summary[task].get("test", {})
            if "auc" in t:
                print(
                    f"{task} test: AUC={t['auc']:.3f} AP={t['ap']:.3f} "
                    f"F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f}"
                )

    out_json = args.out_dir / f"rydberg_temporal_reservoir_{args.tag}.json"
    out_csv = args.out_dir / f"rydberg_temporal_reservoir_{args.tag}.csv"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_json}\nWrote {out_csv}")


if __name__ == "__main__":
    main()
