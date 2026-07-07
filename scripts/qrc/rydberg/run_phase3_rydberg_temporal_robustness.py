#!/usr/bin/env python3
"""
Robustness sweep for the temporal Rydberg response probe.

This script deliberately avoids best-run storytelling. It runs a small grid over
lookback, market scalar, and detuning window, then reports whether the temporal
Rydberg feature map consistently improves test metrics over the raw temporal
baseline.

It calls scripts/qrc/rydberg/run_phase3_rydberg_temporal_probe.py for each configuration and
aggregates the resulting metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SweepConfig:
    data: Path
    response_grid: Path
    lookbacks: tuple[int, ...]
    scalars: tuple[str, ...]
    windows: tuple[tuple[float, float], ...]
    target_col: str
    out_dir: Path
    runner: Path


def parse_csv_ints(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_csv_strings(text: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def parse_windows(text: str) -> tuple[tuple[float, float], ...]:
    windows = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Window must use low:high form, got {item}")
        lo_s, hi_s = item.split(":", 1)
        lo, hi = float(lo_s), float(hi_s)
        if lo >= hi:
            raise ValueError(f"Invalid window {item}: low must be < high")
        windows.append((lo, hi))
    return tuple(windows)


def safe_name(s: str) -> str:
    return s.replace(".", "p").replace("-", "m").replace(":", "_").replace("/", "_")


def run_one(cfg: SweepConfig, scalar: str, lookback: int, lo: float, hi: float) -> pd.DataFrame:
    tag = f"scalar-{scalar}__lb-{lookback}__delta-{safe_name(str(lo))}-{safe_name(str(hi))}"
    out_prefix = cfg.out_dir / tag
    cmd = [
        sys.executable,
        str(cfg.runner),
        "--data", str(cfg.data),
        "--response-grid", str(cfg.response_grid),
        "--target-col", cfg.target_col,
        "--scalar-col", scalar,
        "--lookback", str(lookback),
        "--delta-min", str(lo),
        "--delta-max", str(hi),
        "--out-prefix", str(out_prefix),
    ]
    print("RUN", tag, flush=True)
    subprocess.run(cmd, check=True)
    metrics_path = out_prefix.with_name(out_prefix.name + "_metrics.csv")
    metrics = pd.read_csv(metrics_path)
    metrics.insert(0, "delta_max", hi)
    metrics.insert(0, "delta_min", lo)
    metrics.insert(0, "lookback", lookback)
    metrics.insert(0, "scalar", scalar)
    metrics.insert(0, "run_tag", tag)
    return metrics


def add_pairwise_deltas(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["run_tag", "scalar", "lookback", "delta_min", "delta_max", "split", "task"]
    for key_vals, group in all_metrics.groupby(keys, dropna=False):
        row_key = dict(zip(keys, key_vals, strict=True))
        raw = group[group["model"] == "raw_last_baseline"]
        combo = group[group["model"] == "raw_plus_rydberg_temporal"]
        ryd = group[group["model"] == "rydberg_temporal_features"]
        if raw.empty or combo.empty:
            continue
        raw = raw.iloc[0]
        combo = combo.iloc[0]
        out = dict(row_key)
        for metric in ["rmse", "corr", "precision", "recall", "f1", "average_precision", "roc_auc"]:
            if metric in group.columns:
                raw_val = raw.get(metric, np.nan)
                combo_val = combo.get(metric, np.nan)
                out[f"raw_{metric}"] = raw_val
                out[f"raw_plus_rydberg_{metric}"] = combo_val
                out[f"delta_{metric}"] = combo_val - raw_val if pd.notna(raw_val) and pd.notna(combo_val) else np.nan
        if not ryd.empty:
            ryd = ryd.iloc[0]
            for metric in ["rmse", "corr", "f1", "average_precision", "roc_auc"]:
                if metric in group.columns:
                    out[f"rydberg_only_{metric}"] = ryd.get(metric, np.nan)
        rows.append(out)
    return pd.DataFrame(rows)


def summarize_deltas(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, task), group in delta_df.groupby(["split", "task"], dropna=False):
        row = {"split": split, "task": task, "n_runs": int(len(group))}
        for metric in ["rmse", "corr", "f1", "average_precision", "roc_auc"]:
            col = f"delta_{metric}"
            if col not in group.columns:
                continue
            vals = group[col].dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_min"] = float(np.min(vals))
            row[f"{col}_max"] = float(np.max(vals))
            # For rmse, improvement means negative delta; for all others positive.
            if metric == "rmse":
                row[f"{col}_improvement_rate"] = float(np.mean(vals < 0))
            else:
                row[f"{col}_improvement_rate"] = float(np.mean(vals > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Rydberg temporal robustness sweep")
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument("--response-grid", type=Path, default=Path("artifacts/rydberg_tfim_response_zoom_v2.csv"))
    p.add_argument("--lookbacks", default="10,20,40")
    p.add_argument("--scalars", default="rv_accel_log_5_20,vix_rv_spread")
    p.add_argument("--windows", default="0.25:1.00,0.40:1.10,0.55:1.25")
    p.add_argument("--target-col", default="future_rv_20d")
    p.add_argument("--out-dir", type=Path, default=Path("results/qrc/rydberg/phase3_rydberg_temporal_robustness"))
    p.add_argument("--runner", type=Path, default=Path("scripts/qrc/rydberg/run_phase3_rydberg_temporal_probe.py"))
    args = p.parse_args()

    cfg = SweepConfig(
        data=args.data,
        response_grid=args.response_grid,
        lookbacks=parse_csv_ints(args.lookbacks),
        scalars=parse_csv_strings(args.scalars),
        windows=parse_windows(args.windows),
        target_col=args.target_col,
        out_dir=args.out_dir,
        runner=args.runner,
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    all_parts = []
    for scalar in cfg.scalars:
        for lookback in cfg.lookbacks:
            for lo, hi in cfg.windows:
                all_parts.append(run_one(cfg, scalar, lookback, lo, hi))

    all_metrics = pd.concat(all_parts, ignore_index=True)
    delta_df = add_pairwise_deltas(all_metrics)
    summary = summarize_deltas(delta_df)

    all_metrics_path = cfg.out_dir / "all_metrics.csv"
    deltas_path = cfg.out_dir / "raw_vs_rydberg_deltas.csv"
    summary_path = cfg.out_dir / "summary_by_split_task.csv"
    config_path = cfg.out_dir / "sweep_config.json"
    all_metrics.to_csv(all_metrics_path, index=False)
    delta_df.to_csv(deltas_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True, default=str), encoding="utf-8")

    print("Saved:")
    print(all_metrics_path)
    print(deltas_path)
    print(summary_path)
    print(config_path)

    print("\nTest q90/q95 summary:")
    mask = (summary["split"] == "test") & (summary["task"].isin(["q90_warning", "q95_crisis"]))
    print(summary.loc[mask].to_string(index=False))

    print("\nBest/worst test q95 AUC deltas:")
    q95 = delta_df[(delta_df["split"] == "test") & (delta_df["task"] == "q95_crisis")].copy()
    cols = ["run_tag", "scalar", "lookback", "delta_min", "delta_max", "raw_roc_auc", "raw_plus_rydberg_roc_auc", "delta_roc_auc", "raw_average_precision", "raw_plus_rydberg_average_precision", "delta_average_precision", "raw_f1", "raw_plus_rydberg_f1", "delta_f1"]
    cols = [c for c in cols if c in q95.columns]
    if not q95.empty:
        print(q95.sort_values("delta_roc_auc", ascending=False)[cols].head(8).to_string(index=False))
        print("\nWorst:")
        print(q95.sort_values("delta_roc_auc", ascending=True)[cols].head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
