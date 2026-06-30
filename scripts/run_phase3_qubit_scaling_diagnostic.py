"""Phase 3 RF-QRC qubit scaling diagnostic.

Tiny scaling test to check whether the 6-qubit bottleneck is causing the
memory/nonlinearity tradeoff. This is intentionally bounded:

- n_qubits default: 6, 8, 10
- one ring topology
- one seed
- limited train/val/test rows by default
- selected Z + ring/local ZZ readout only
- compares QRC transition classifiers against a HAR classifier baseline

This is a diagnostic, not a final benchmark.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET_COL = "future_rv_20d"
DATA_CANDIDATES = [
    "data/processed/phase2_spy_vix_volatility.csv",
    "data/processed/phase2_spy_vix_dataset.csv",
    "data/processed/phase2_spy_vix_features.csv",
    "data/processed/phase2_modeling_dataset.csv",
    "data/phase2_spy_vix_dataset.csv",
    "data/phase2_spy_vix_features.csv",
    "results/tables/phase2_modeling_dataset.csv",
]

HAR_CANDIDATES = [
    "rv_5d", "rv_10d", "rv_20d", "rv_60d", "vix_close",
    "vix_rv_ratio", "rv_5_20_ratio", "rv_20_60_ratio",
    "rv_slope_5_20", "spy_drawdown_20d", "vix_log_change",
]


def project_root_from_cwd() -> Path:
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if p.name == "qpitome-qrc-volatility" and (p / "scripts").exists():
            return p
    if cwd.name == "notebooks" and (cwd.parent / "scripts").exists():
        return cwd.parent
    return cwd


def find_dataset(project_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        candidates = [p] if p.is_absolute() else [project_root / p, Path.cwd() / p]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(candidates[0])
    for rel in DATA_CANDIDATES:
        p = project_root / rel
        if p.exists():
            return p
    raise FileNotFoundError("No Phase 2 modeling dataset found.")


def chronological_split(n: int) -> np.ndarray:
    if n >= 7658:
        return np.array(["train"] * 5420 + ["val"] * 1219 + ["test"] * (n - 6639))[:n]
    n_train = int(0.70 * n)
    n_val = int(0.85 * n)
    return np.array(["train"] * n_train + ["val"] * (n_val - n_train) + ["test"] * (n - n_val))


def load_frame(path: Path, target_col: str):
    df = pd.read_csv(path)
    date_col = next((c for c in ["date", "Date", "timestamp", "time"] if c in df.columns), None)
    if date_col:
        df = df.sort_values(date_col).reset_index(drop=True)
    if target_col not in df.columns:
        raise KeyError(f"Missing target column {target_col!r} in {path}")
    split = df["split"].astype(str).str.lower().to_numpy() if "split" in df.columns else chronological_split(len(df))
    return df, split, date_col


def numeric_feature_cols(df: pd.DataFrame, target_col: str, date_col: str | None) -> list[str]:
    blocked = {target_col, "split", "run_name"}
    if date_col:
        blocked.add(date_col)
    leakage_tokens = ("pred", "actual", "future", "target", "threshold", "tp", "fp", "fn")
    return [
        c
        for c in df.columns
        if c not in blocked
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(tok in c.lower() for tok in leakage_tokens)
    ]


def build_har_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in HAR_CANDIDATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) < 3:
        # fallback to volatility/VIX-ish numeric columns only
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and ("rv" in c.lower() or "vix" in c.lower()) and "future" not in c.lower()]
    if not cols:
        raise ValueError("Could not identify HAR/memory baseline features.")
    X = df[cols].to_numpy(dtype=float)
    return X, cols


def finite_mask(arrays: list[np.ndarray]) -> np.ndarray:
    m = np.ones(len(arrays[0]), dtype=bool)
    for A in arrays:
        if A.ndim == 1:
            m &= np.isfinite(A)
        else:
            m &= np.all(np.isfinite(A), axis=1)
    return m


def subset_indices(split: np.ndarray, max_train: int, max_val: int, max_test: int) -> np.ndarray:
    keep = np.zeros(len(split), dtype=bool)
    for sp, mx in [("train", max_train), ("val", max_val), ("test", max_test)]:
        idx = np.where(split == sp)[0]
        if mx and mx > 0:
            idx = idx[: min(mx, len(idx))]
        keep[idx] = True
    return keep


def make_pca_inputs(X: np.ndarray, split: np.ndarray, n_qubits: int, mode: str) -> np.ndarray:
    train = split == "train"
    Xs = StandardScaler().fit(X[train]).transform(X)
    if mode == "pca":
        U = PCA(n_components=n_qubits, random_state=SEED).fit(Xs[train]).transform(Xs)
    elif mode == "level_rate":
        k = n_qubits // 2
        level = PCA(n_components=k, random_state=SEED).fit(Xs[train]).transform(Xs)
        rate = np.zeros_like(level)
        rate[1:] = level[1:] - level[:-1]
        U = np.hstack([level, rate])
        if U.shape[1] < n_qubits:
            pad = np.zeros((len(U), n_qubits - U.shape[1]))
            U = np.hstack([U, pad])
    else:
        raise ValueError(mode)
    U = StandardScaler().fit(U[train]).transform(U)
    return np.clip(U, -3.0, 3.0)


def leaky_filter(U: np.ndarray, leak: float) -> np.ndarray:
    out = np.zeros_like(U)
    out[0] = U[0]
    for t in range(1, len(U)):
        out[t] = leak * U[t] + (1.0 - leak) * out[t - 1]
    return out


def ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0], [0.0, np.exp(0.5j * theta)]], dtype=complex)


def apply_one(state: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    tensor = state.reshape([2] * n)
    tensor = np.moveaxis(tensor, q, 0)
    tensor = np.tensordot(gate, tensor, axes=([1], [0]))
    tensor = np.moveaxis(tensor, 0, q)
    return tensor.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    out = np.empty_like(state)
    for idx, amp in enumerate(state):
        dest = idx ^ (1 << target) if ((idx >> control) & 1) else idx
        out[dest] = amp
    return out


def apply_zz_phase(state: np.ndarray, i: int, j: int, theta: float, n: int) -> np.ndarray:
    out = state.copy()
    for idx in range(len(out)):
        zi = 1.0 if ((idx >> i) & 1) == 0 else -1.0
        zj = 1.0 if ((idx >> j) & 1) == 0 else -1.0
        out[idx] *= np.exp(-1j * theta * zi * zj)
    return out


def selected_z_zz_features(state: np.ndarray, n: int, zz_mode: str) -> np.ndarray:
    probs = np.abs(state) ** 2
    zvals = np.empty((len(state), n), dtype=float)
    for idx in range(len(state)):
        for q in range(n):
            zvals[idx, q] = 1.0 if ((idx >> q) & 1) == 0 else -1.0
    z = probs @ zvals
    pairs: list[tuple[int, int]] = []
    if zz_mode in {"ring", "ring_plus_next"}:
        pairs.extend([(i, (i + 1) % n) for i in range(n)])
    if zz_mode == "ring_plus_next":
        pairs.extend([(i, (i + 2) % n) for i in range(n)])
    if zz_mode == "all":
        pairs.extend([(i, j) for i in range(n) for j in range(i + 1, n)])
    # de-duplicate unordered pairs
    uniq = []
    seen = set()
    for i, j in pairs:
        a, b = sorted((i, j))
        if (a, b) not in seen:
            seen.add((a, b))
            uniq.append((a, b))
    zz = [probs @ (zvals[:, i] * zvals[:, j]) for i, j in uniq]
    return np.concatenate([z, np.asarray(zz, dtype=float)])


class RFQRCRingMap:
    def __init__(self, n_qubits: int, input_scale: float, random_scale: float, seed: int, zz_mode: str):
        self.n = n_qubits
        self.input_scale = input_scale
        self.zz_mode = zz_mode
        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.ry_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.zz_angles = np.triu(rng.normal(0.0, random_scale, size=(n_qubits, n_qubits)), 1)

    def one(self, u: np.ndarray) -> np.ndarray:
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        for factor in [1.0]:
            for q, val in enumerate(u):
                state = apply_one(state, ry(float(factor * self.input_scale * val)), q, self.n)
        for i in range(self.n):
            state = apply_cnot(state, i, (i + 1) % self.n, self.n)
        for q, val in enumerate(u):
            state = apply_one(state, ry(float(self.input_scale * val)), q, self.n)
        for q in range(self.n):
            state = apply_one(state, rz(float(self.rz_angles[q])), q, self.n)
            state = apply_one(state, ry(float(self.ry_angles[q])), q, self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                state = apply_zz_phase(state, i, j, float(self.zz_angles[i, j]), self.n)
        return selected_z_zz_features(state, self.n, self.zz_mode)

    def transform(self, U: np.ndarray) -> np.ndarray:
        return np.vstack([self.one(u) for u in U])


def effective_rank(A: np.ndarray) -> float:
    X = A - A.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def best_threshold_from_val(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if len(thr) == 0:
        return 0.5, 0.0
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(f1[idx])


def fit_classifier(F: np.ndarray, event: np.ndarray, split: np.ndarray, C: float):
    train = split == "train"
    val = split == "val"
    scaler = StandardScaler().fit(F[train])
    Fs = scaler.transform(F)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=SEED)
    clf.fit(Fs[train], event[train])
    prob = clf.predict_proba(Fs)[:, 1]
    thr, val_best = best_threshold_from_val(event[val], prob[val])
    return prob, thr, val_best


def classifier_metrics(name: str, event_name: str, prob: np.ndarray, event: np.ndarray, split: np.ndarray, threshold: float, extra: dict):
    rows = []
    for sp in ["train", "val", "test"]:
        m = split == sp
        called = prob[m] >= threshold
        row = {
            "model": name,
            "event": event_name,
            "split": sp,
            "n": int(m.sum()),
            "event_rate": float(event[m].mean()),
            "called_rate": float(called.mean()),
            "precision": float(precision_score(event[m], called, zero_division=0)),
            "recall": float(recall_score(event[m], called, zero_division=0)),
            "f1": float(f1_score(event[m], called, zero_division=0)),
            "average_precision": float(average_precision_score(event[m], prob[m])),
            "threshold": threshold,
            **extra,
        }
        try:
            row["roc_auc"] = float(roc_auc_score(event[m], prob[m]))
        except ValueError:
            row["roc_auc"] = float("nan")
        rows.append(row)
    return rows


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default=None)
    p.add_argument("--target-col", default=TARGET_COL)
    p.add_argument("--n-qubits", nargs="+", type=int, default=[6, 8, 10])
    p.add_argument("--input-mode", choices=["level_rate", "pca"], default="level_rate")
    p.add_argument("--zz-mode", choices=["ring", "ring_plus_next", "all"], default="ring")
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--input-scale", type=float, default=math.pi / 3)
    p.add_argument("--random-scale", type=float, default=0.35)
    p.add_argument("--logistic-C", type=float, default=1.0)
    p.add_argument("--max-train", type=int, default=1800)
    p.add_argument("--max-val", type=int, default=600)
    p.add_argument("--max-test", type=int, default=800)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root_from_cwd()
    outdir = root / "results" / "tables"
    outdir.mkdir(parents=True, exist_ok=True)
    data_path = find_dataset(root, args.data_path)
    print(f"Project root: {root}")
    print(f"Dataset: {data_path}")

    df, split, date_col = load_frame(data_path, args.target_col)
    y = df[args.target_col].to_numpy(dtype=float)
    X_all = df[numeric_feature_cols(df, args.target_col, date_col)].to_numpy(dtype=float)
    X_har, har_cols = build_har_features(df)
    m = finite_mask([y, X_all, X_har])
    df = df.loc[m].reset_index(drop=True)
    y = y[m]
    X_all = X_all[m]
    X_har = X_har[m]
    split = split[m]
    dates = df[date_col].to_numpy() if date_col else np.arange(len(df))

    keep = subset_indices(split, args.max_train, args.max_val, args.max_test)
    y = y[keep]
    X_all = X_all[keep]
    X_har = X_har[keep]
    split = split[keep]
    dates = dates[keep]

    train = split == "train"
    thresholds = {
        "q80": float(np.quantile(y[train], 0.80)),
        "q90": float(np.quantile(y[train], 0.90)),
        "q95": float(np.quantile(y[train], 0.95)),
    }
    print("Subset counts:", {sp: int((split == sp).sum()) for sp in ["train", "val", "test"]})
    print("HAR cols:", har_cols)

    all_rows = []
    pred_df = pd.DataFrame({"date": dates, "actual_future_rv_20d": y, "split": split})

    # HAR baseline classifier.
    for event_name, thr0 in thresholds.items():
        event = y >= thr0
        prob, threshold, val_best = fit_classifier(X_har, event, split, args.logistic_C)
        pred_df[f"har_{event_name}_prob"] = prob
        all_rows.extend(classifier_metrics(
            "HAR_memory_classifier", event_name, prob, event, split, threshold,
            {"n_qubits": 0, "features": X_har.shape[1], "effective_rank": effective_rank(X_har[train]), "val_best_f1": val_best},
        ))

    # QRC scaling classifiers and HAR+QRC concatenation.
    for nq in args.n_qubits:
        t0 = time.time()
        print(f"\nBuilding QRC features n_qubits={nq}")
        U = make_pca_inputs(X_all, split, nq, args.input_mode)
        U = leaky_filter(U, args.leak)
        fmap = RFQRCRingMap(nq, args.input_scale, args.random_scale, args.seed, args.zz_mode)
        Fq = fmap.transform(U)
        elapsed = time.time() - t0
        print(f"n={nq}: Fq shape={Fq.shape}, effective_rank={effective_rank(Fq[train]):.3f}, elapsed={elapsed:.1f}s")

        Fhq = np.hstack([StandardScaler().fit_transform(X_har), StandardScaler().fit_transform(Fq)])
        for model_name, F in [(f"QRC_ring_{nq}q", Fq), (f"HAR_plus_QRC_ring_{nq}q", Fhq)]:
            for event_name, thr0 in thresholds.items():
                event = y >= thr0
                prob, threshold, val_best = fit_classifier(F, event, split, args.logistic_C)
                pred_df[f"{model_name}_{event_name}_prob"] = prob
                all_rows.extend(classifier_metrics(
                    model_name, event_name, prob, event, split, threshold,
                    {
                        "n_qubits": nq,
                        "features": F.shape[1],
                        "effective_rank": effective_rank(F[train]),
                        "val_best_f1": val_best,
                        "elapsed_feature_seconds": elapsed,
                    },
                ))

    results = pd.DataFrame(all_rows)
    results_path = outdir / "phase3_qubit_scaling_transition_classifier_metrics.csv"
    pred_path = outdir / "phase3_qubit_scaling_transition_classifier_predictions.csv"
    test_path = outdir / "phase3_qubit_scaling_transition_classifier_test_summary.csv"
    results.to_csv(results_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    results.loc[results["split"].eq("test")].to_csv(test_path, index=False)

    print("\nSaved:")
    print(results_path)
    print(pred_path)
    print(test_path)
    print("\nTest summary sorted by q95 F1:")
    show = ["model", "event", "n_qubits", "features", "effective_rank", "precision", "recall", "f1", "average_precision", "roc_auc", "called_rate", "event_rate", "threshold"]
    print(results.loc[(results["split"].eq("test")) & (results["event"].eq("q95")), show].sort_values("f1", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
