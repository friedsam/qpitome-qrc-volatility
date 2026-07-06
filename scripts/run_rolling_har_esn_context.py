#!/usr/bin/env python3
"""Rolling paired HAR-vs-ESN performance aligned to temporal context.

Uses expanding training, fixed validation, purge, and annual test blocks.
HAR is exact canonical HAR. ESN is exact Phase 3 source logic with the same
four frozen configurations and q90 validation-AP selection.

Primary question: does paired ESN-vs-HAR advantage covary with pre-test market
memory/context metrics from the temporal-context audit?
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.evaluation.walkforward import make_purged_rolling_windows

ESN_SOURCE = Path(__file__).with_name("run_phase3_esn_ridge_walkforward.py")
TARGET = "future_rv_20d"
HAR_FEATURES = ["rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close"]


def load_esn_source():
    spec = importlib.util.spec_from_file_location("exact_esn_source", ESN_SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase3_spy_vix_volatility_extended.csv"))
    p.add_argument("--context", type=Path, default=Path("results/diagnostics/temporal_context/temporal_context_windows.csv"))
    p.add_argument("--outdir", type=Path, default=Path("results/diagnostics/rolling_har_esn_context"))
    p.add_argument("--min-train", type=int, default=2500)
    p.add_argument("--val-size", type=int, default=504)
    p.add_argument("--purge", type=int, default=60)
    p.add_argument("--test-size", type=int, default=252)
    p.add_argument("--step", type=int, default=252)
    p.add_argument("--lookback", type=int, default=40)
    p.add_argument("--pca-components", type=int, default=6)
    return p.parse_args()


def make_windows(n, min_train, val_size, purge, test_size, step):
    """Backward-compatible wrapper around the shared rolling protocol."""
    windows = make_purged_rolling_windows(
        n,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
        test_size=test_size,
        step=step,
    )
    return [
        {
            "window": w["window"],
            "train": w["train"],
            "val": w["val"],
            "test": w["test"],
        }
        for w in windows
    ]


def rmse(y, pred):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def qlike(y, pred):
    m = evaluate_volatility_forecast(np.asarray(y), np.asarray(pred))
    return float(m.qlike)


def main():
    a = parse_args(); a.outdir.mkdir(parents=True, exist_ok=True)
    esn = load_esn_source()
    df = pd.read_csv(a.data).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    context = pd.read_csv(a.context)
    context["window_end"] = pd.to_datetime(context["window_end"])

    required = {"date", TARGET, *HAR_FEATURES, *FEATURE_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing: raise ValueError(f"Missing columns: {missing}")

    windows = make_windows(len(df), a.min_train, a.val_size, a.purge, a.test_size, a.step)
    grid = esn.make_grid([42])
    rows = []

    for w in windows:
        split = {k: df.iloc[s:e].copy().reset_index(drop=True) for k, (s, e) in (("train", w["train"]), ("val", w["val"]), ("test", w["test"]))}

        # Exact HAR
        har = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        har.fit(split["train"][HAR_FEATURES], split["train"][TARGET])
        har_pred = np.maximum(har.predict(split["test"][HAR_FEATURES]), 1e-8)

        # Exact ESN preprocessing and model selection
        scaler = StandardScaler(); pca = PCA(n_components=a.pca_components, random_state=42)
        pca.fit(scaler.fit_transform(split["train"][FEATURE_COLUMNS]))
        pca_cols = [f"pca{i+1}" for i in range(a.pca_components)]
        seq = {}
        for name in ("train", "val", "test"):
            z = pca.transform(scaler.transform(split[name][FEATURE_COLUMNS]))
            frame = pd.DataFrame(z, columns=pca_cols); frame[TARGET] = split[name][TARGET].to_numpy()
            seq[name] = esn.make_sequence_arrays(frame, pca_cols, a.lookback)
        X = {s: seq[s][0] for s in seq}; y = {s: seq[s][1] for s in seq}
        _, lab90 = esn.label_blocks(y, 0.90)

        candidates = []
        for cfg in grid:
            W_in, W = esn.make_esn_weights(X["train"].shape[2], cfg["n"], cfg["sr"], cfg["inp"], cfg["seed"])
            H = {s: esn.esn_states(X[s], W_in, W, cfg["leak"]) for s in X}
            scores = esn.fit_ridge_scores(H, y, cfg["alpha"])
            val_ap = esn.binary_metrics(lab90["val"], scores["val"])["ap"]
            candidates.append((val_ap, cfg, scores))
        finite = [c for c in candidates if np.isfinite(c[0])]
        chosen = max(finite, key=lambda t: t[0]) if finite else candidates[0]
        _, cfg, scores = chosen
        esn_pred = np.exp(scores["test"])

        # Align HAR to ESN post-lookback test dates
        har_aligned = har_pred[a.lookback - 1:]
        test_dates = split["test"]["date"].iloc[a.lookback - 1:].reset_index(drop=True)
        ytest = y["test"]

        har_rmse = rmse(ytest, har_aligned); esn_rmse = rmse(ytest, esn_pred)
        har_qlike = qlike(ytest, har_aligned); esn_qlike = qlike(ytest, esn_pred)

        # Use latest context window ending no later than test start.
        test_start = test_dates.iloc[0]
        eligible = context[context["window_end"] <= test_start]
        ctx = eligible.iloc[-1] if len(eligible) else None

        row = {
            "window": w["window"],
            "test_start": str(test_dates.iloc[0].date()),
            "test_end": str(test_dates.iloc[-1].date()),
            "n_test": int(len(ytest)),
            "har_rmse": har_rmse,
            "esn_rmse": esn_rmse,
            "delta_rmse_esn_minus_har": esn_rmse - har_rmse,
            "har_qlike": har_qlike,
            "esn_qlike": esn_qlike,
            "delta_qlike_esn_minus_har": esn_qlike - har_qlike,
            "esn_config": cfg["config_id"],
        }
        if ctx is not None:
            for c in context.columns:
                if c not in {"window_start", "window_end", "mid_date"}:
                    row[f"context__{c}"] = ctx[c]
            row["context_window_end"] = str(ctx["window_end"].date())
        rows.append(row)
        print(f"window {w['window']:02d} {row['test_start']} -> {row['test_end']}  dRMSE={row['delta_rmse_esn_minus_har']:+.6f}  dQLIKE={row['delta_qlike_esn_minus_har']:+.6f}")

    perf = pd.DataFrame(rows)
    perf.to_csv(a.outdir / "rolling_har_esn_performance.csv", index=False)

    assoc_rows = []
    for target_col in ["delta_rmse_esn_minus_har", "delta_qlike_esn_minus_har"]:
        for c in [x for x in perf.columns if x.startswith("context__")]:
            x = pd.to_numeric(perf[c], errors="coerce")
            yv = pd.to_numeric(perf[target_col], errors="coerce")
            ok = x.notna() & yv.notna()
            if ok.sum() < 8 or x[ok].std() <= 1e-15:
                continue
            rho, p = spearmanr(x[ok], yv[ok])
            assoc_rows.append({"performance_delta": target_col, "context_metric": c, "spearman_rho": float(rho), "p_value": float(p), "n_windows": int(ok.sum())})
    assoc = pd.DataFrame(assoc_rows).sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    assoc.to_csv(a.outdir / "context_performance_associations.csv", index=False)

    summary = {
        "n_windows": int(len(perf)),
        "esn_beats_har_rmse_fraction": float((perf["delta_rmse_esn_minus_har"] < 0).mean()),
        "esn_beats_har_qlike_fraction": float((perf["delta_qlike_esn_minus_har"] < 0).mean()),
        "median_delta_rmse_esn_minus_har": float(perf["delta_rmse_esn_minus_har"].median()),
        "median_delta_qlike_esn_minus_har": float(perf["delta_qlike_esn_minus_har"].median()),
    }
    (a.outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Rolling HAR vs ESN summary ===")
    print(pd.Series(summary).to_string())
    print("\n=== Strongest context/performance associations ===")
    print(assoc.head(20).to_string(index=False))
    print(f"\nWrote outputs to {a.outdir}")


if __name__ == "__main__":
    main()
