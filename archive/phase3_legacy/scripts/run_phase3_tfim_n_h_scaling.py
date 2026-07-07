"""Bounded TFIM n-by-h scaling diagnostic for Phase 3.

Runs a small non-feedback TFIM statevector map on the real Phase 2 volatility
CSV, then evaluates ridge forecasts. Diagnostic only; not the final production
Phase 2 feedback reservoir.
"""
from __future__ import annotations

import argparse, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

TARGET = "future_rv_20d"
DATA_CANDIDATES = [
    "data/processed/phase2_spy_vix_volatility.csv",
    "data/processed/phase2_spy_vix_dataset.csv",
    "data/processed/phase2_spy_vix_features.csv",
    "data/processed/phase2_modeling_dataset.csv",
    "results/tables/phase2_modeling_dataset.csv",
]
HAR_COLS = ["rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close",
            "rv_slope_5_20", "spy_drawdown_20d", "vix_log_change"]

def root() -> Path:
    p = Path.cwd().resolve()
    for q in [p, *p.parents]:
        if q.name == "qpitome-qrc-volatility" and (q / "scripts").exists():
            return q
    return p

def chrono_split(n: int) -> np.ndarray:
    if n >= 7658:
        return np.array(["train"] * 5420 + ["val"] * 1219 + ["test"] * (n - 6639))[:n]
    a, b = int(.70*n), int(.85*n)
    return np.array(["train"]*a + ["val"]*(b-a) + ["test"]*(n-b))

def dataset_path(r: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        for c in ([p] if p.is_absolute() else [r/p, Path.cwd()/p]):
            if c.exists(): return c
        raise FileNotFoundError(explicit)
    for rel in DATA_CANDIDATES:
        p = r / rel
        if p.exists(): return p
    raise FileNotFoundError("no Phase 2 data CSV found")

def load_data(path: Path, max_train: int, max_val: int, max_test: int):
    df = pd.read_csv(path)
    date_col = next((c for c in ["date", "Date", "timestamp", "time"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)
    split = df["split"].astype(str).str.lower().to_numpy() if "split" in df.columns else chrono_split(len(df))
    blocked = {TARGET, "split"}
    if date_col: blocked.add(date_col)
    bad = ("future", "target", "actual", "pred", "threshold", "tp", "fp", "fn")
    feat_cols = [c for c in df.columns if c not in blocked and pd.api.types.is_numeric_dtype(df[c]) and not any(t in c.lower() for t in bad)]
    har_cols = [c for c in HAR_COLS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(har_cols) < 3:
        har_cols = [c for c in feat_cols if "rv" in c.lower() or "vix" in c.lower()][:8]
    y = df[TARGET].to_numpy(float); X = df[feat_cols].to_numpy(float); Xh = df[har_cols].to_numpy(float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Xh), axis=1)
    df, y, X, Xh, split = df.loc[ok].reset_index(drop=True), y[ok], X[ok], Xh[ok], split[ok]
    keep = np.zeros(len(y), bool)
    for sp, mx in [("train", max_train), ("val", max_val), ("test", max_test)]:
        idx = np.where(split == sp)[0]
        keep[idx[:min(mx, len(idx))]] = True
    dates = df[date_col].to_numpy() if date_col else np.arange(len(df))
    return X[keep], Xh[keep], y[keep], split[keep], dates[keep], feat_cols, har_cols

def pca_inputs(X, split, n):
    tr = split == "train"
    Xs = StandardScaler().fit(X[tr]).transform(X)
    U = PCA(n_components=n, random_state=42).fit(Xs[tr]).transform(Xs)
    U = StandardScaler().fit(U[tr]).transform(U)
    return np.clip(U, -3, 3)

def leaky(U, leak):
    V = np.zeros_like(U); V[0] = U[0]
    for t in range(1, len(U)): V[t] = leak*U[t] + (1-leak)*V[t-1]
    return V

def rx(a):
    c, s = math.cos(a/2), math.sin(a/2)
    return np.array([[c, -1j*s], [-1j*s, c]], complex)

def ry(a):
    c, s = math.cos(a/2), math.sin(a/2)
    return np.array([[c, -s], [s, c]], complex)

def oneq(psi, gate, q, n):
    T = psi.reshape([2]*n); T = np.moveaxis(T, q, 0)
    T = np.tensordot(gate, T, axes=([1], [0])); T = np.moveaxis(T, 0, q)
    return T.reshape(-1)

def zz(psi, i, j, a, n):
    out = psi.copy()
    for k in range(len(out)):
        zi = 1 if ((k >> i) & 1) == 0 else -1; zj = 1 if ((k >> j) & 1) == 0 else -1
        out[k] *= np.exp(-1j * a * zi * zj)
    return out

def features(psi, n):
    p = np.abs(psi)**2; zvals = np.empty((len(psi), n))
    for k in range(len(psi)):
        for q in range(n): zvals[k, q] = 1 if ((k >> q) & 1) == 0 else -1
    z = p @ zvals
    pairs = [p @ (zvals[:, i] * zvals[:, j]) for i in range(n) for j in range(i+1, n)]
    return np.r_[z, pairs]

def tfim_features(U, n, h, seed, input_scale, dt, layers):
    rng = np.random.default_rng(seed); J = np.triu(rng.normal(0, 1/np.sqrt(max(1, n)), size=(n, n)), 1)
    F = []
    for u in U:
        psi = np.zeros(2**n, complex); psi[0] = 1
        for q, v in enumerate(u): psi = oneq(psi, ry(input_scale*float(v)), q, n)
        for _ in range(layers):
            for i in range(n):
                for j in range(i+1, n): psi = zz(psi, i, j, dt*J[i, j], n)
            for q in range(n): psi = oneq(psi, rx(dt*h), q, n)
        F.append(features(psi, n))
    return np.vstack(F)

def erank(A):
    s = np.linalg.svd(A - A.mean(0, keepdims=True), full_matrices=False, compute_uv=False)
    p = s/(s.sum()+1e-12); return float(np.exp(-np.sum(p*np.log(p+1e-12))))

def qlike(y, yp):
    return float(np.mean(np.log(np.maximum(yp**2, 1e-12)) + np.maximum(y**2, 1e-12)/np.maximum(yp**2, 1e-12)))

def corr(y, yp): return float(np.corrcoef(y, yp)[0, 1]) if np.std(yp) > 1e-12 else float("nan")

def fit_pred(F, y, split, alpha):
    tr = split == "train"; Fs = StandardScaler().fit(F[tr]).transform(F)
    return np.exp(Ridge(alpha=alpha).fit(Fs[tr], np.log(np.maximum(y[tr], 1e-8))).predict(Fs))

def eval_rows(name, y, yp, split, thr, extra):
    rows = []
    for sp in ["train", "val", "test"]:
        m = split == sp
        row = dict(model=name, split=sp, n_rows=int(m.sum()), rmse=float(np.sqrt(mean_squared_error(y[m], yp[m]))), qlike=qlike(y[m], yp[m]), corr=corr(y[m], yp[m]), actual_std=float(y[m].std()), pred_std=float(yp[m].std()), pred_mean=float(yp[m].mean()), **extra)
        for label, t in thr.items():
            a, c = y[m] >= t, yp[m] >= t
            row[f"{label}_precision"] = float(precision_score(a, c, zero_division=0)); row[f"{label}_recall"] = float(recall_score(a, c, zero_division=0)); row[f"{label}_f1"] = float(f1_score(a, c, zero_division=0)); row[f"{label}_pred_rate"] = float(c.mean())
        if sp == "test":
            top = y[m] >= np.quantile(y[m], .80); row["top20_pred_actual_ratio"] = float(yp[m][top].mean()/y[m][top].mean())
        rows.append(row)
    return rows

def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--data-path", default=None)
    ap.add_argument("--n-qubits", nargs="+", type=int, default=[4,6,8,10,12]); ap.add_argument("--h-grid", nargs="+", type=float, default=[0.2,0.5,1.0]); ap.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    ap.add_argument("--max-train", type=int, default=1600); ap.add_argument("--max-val", type=int, default=500); ap.add_argument("--max-test", type=int, default=700)
    ap.add_argument("--leak", type=float, default=.3); ap.add_argument("--input-scale", type=float, default=math.pi/3); ap.add_argument("--dt", type=float, default=1.0); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--ridge-alpha", type=float, default=3000.0)
    args = ap.parse_args(argv)
    r = root(); out = r/"results"/"tables"; out.mkdir(parents=True, exist_ok=True)
    X, Xh, y, split, dates, feat_cols, har_cols = load_data(dataset_path(r, args.data_path), args.max_train, args.max_val, args.max_test)
    tr = split == "train"; thr = {"q80": float(np.quantile(y[tr], .8)), "q90": float(np.quantile(y[tr], .9)), "q95": float(np.quantile(y[tr], .95))}
    rows = []; preds = pd.DataFrame({"date": dates, "actual_future_rv_20d": y, "split": split})
    hp = fit_pred(Xh, y, split, args.ridge_alpha); preds["HAR_memory_ridge_pred"] = hp
    rows += eval_rows("HAR_memory_ridge", y, hp, split, thr, dict(n_qubits=0, h=np.nan, seed=-1, features=Xh.shape[1], effective_rank=erank(Xh[tr]), elapsed_feature_seconds=0.0))
    Ucache = {n: leaky(pca_inputs(X, split, n), args.leak) for n in args.n_qubits}
    for n in args.n_qubits:
        for h in args.h_grid:
            for seed in args.seeds:
                name = f"TFIM_{n}q_h{h:g}_seed{seed}"; print("Running", name, flush=True)
                t0 = time.time(); F = tfim_features(Ucache[n], n, h, seed, args.input_scale, args.dt, args.layers); elapsed = time.time()-t0
                yp = fit_pred(F, y, split, args.ridge_alpha); preds[name+"_pred"] = yp; e = erank(F[tr])
                print(name, F.shape, "erank", round(e,2), "sec", round(elapsed,1), flush=True)
                rows += eval_rows(name, y, yp, split, thr, dict(n_qubits=n, h=h, seed=seed, features=F.shape[1], effective_rank=e, elapsed_feature_seconds=elapsed))
    metrics = pd.DataFrame(rows); metrics.to_csv(out/"phase3_tfim_n_h_scaling_metrics.csv", index=False); metrics[metrics.split.eq("test")].to_csv(out/"phase3_tfim_n_h_scaling_test_summary.csv", index=False); preds.to_csv(out/"phase3_tfim_n_h_scaling_predictions.csv", index=False)
    test = metrics[metrics.split.eq("test") & (metrics.n_qubits > 0)]
    agg = test.groupby(["n_qubits", "h"], as_index=False).agg(qlike_mean=("qlike","mean"), corr_mean=("corr","mean"), pred_std_mean=("pred_std","mean"), q90_f1_mean=("q90_f1","mean"), q95_f1_mean=("q95_f1","mean"), effective_rank_mean=("effective_rank","mean"), elapsed_mean=("elapsed_feature_seconds","mean"))
    agg.to_csv(out/"phase3_tfim_n_h_scaling_test_agg.csv", index=False)
    print("Saved TFIM n-h scaling outputs"); print(metrics[metrics.split.eq("test")].sort_values("qlike").head(10).to_string(index=False))

if __name__ == "__main__": main()
