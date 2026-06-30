"""Phase 3 one-QRC two-head RF-QRC readout experiment.

This keeps one RF-QRC ring feature map and trains two readout heads on the same
raw quantum features:

1. Ridge regression head for continuous future volatility.
2. Logistic crisis/warning classifier heads for q80/q90/q95 event labels.

It tests whether the RF-QRC crisis signal is lost because the regression loss is
misaligned with rare-event detection, not because a second reservoir is needed.
"""

from __future__ import annotations

import argparse
import math
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


def load_matrix(path: Path, target_col: str = TARGET_COL):
    df = pd.read_csv(path)
    date_col = next((c for c in ["date", "Date", "timestamp", "time"] if c in df.columns), None)
    if date_col:
        df = df.sort_values(date_col).reset_index(drop=True)
    if target_col not in df.columns:
        raise KeyError(f"Missing target column {target_col!r} in {path}")

    split = df["split"].astype(str).str.lower().to_numpy() if "split" in df.columns else chronological_split(len(df))
    blocked = {target_col, "split", "run_name"}
    if date_col:
        blocked.add(date_col)
    leakage_tokens = ("pred", "actual", "future", "target", "threshold", "tp", "fp", "fn")
    feature_cols = [
        c
        for c in df.columns
        if c not in blocked
        and pd.api.types.is_numeric_dtype(df[c])
        and not any(tok in c.lower() for tok in leakage_tokens)
    ]
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found after leakage filters.")

    finite = np.isfinite(df[target_col].to_numpy(dtype=float))
    for c in feature_cols:
        finite &= np.isfinite(df[c].to_numpy(dtype=float))

    work = df.loc[finite].reset_index(drop=True)
    X = work[feature_cols].to_numpy(dtype=float)
    y = work[target_col].to_numpy(dtype=float)
    split = split[finite]
    dates = work[date_col].to_numpy() if date_col else np.arange(len(work))
    return X, y, split, dates, feature_cols


def make_level_rate_inputs(X: np.ndarray, split: np.ndarray, n_qubits: int) -> np.ndarray:
    train = split == "train"
    k = n_qubits // 2
    Xs = StandardScaler().fit(X[train]).transform(X)
    level = PCA(n_components=k, random_state=SEED).fit(Xs[train]).transform(Xs)
    rate = np.zeros_like(level)
    rate[1:] = level[1:] - level[:-1]
    U = np.hstack([level, rate])
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


def z_zz_features(state: np.ndarray, n: int) -> np.ndarray:
    probs = np.abs(state) ** 2
    zvals = np.empty((len(state), n), dtype=float)
    for idx in range(len(state)):
        for q in range(n):
            zvals[idx, q] = 1.0 if ((idx >> q) & 1) == 0 else -1.0
    z = probs @ zvals
    zz = [probs @ (zvals[:, i] * zvals[:, j]) for i in range(n) for j in range(i + 1, n)]
    return np.concatenate([z, np.asarray(zz)])


class RFQRCRingMap:
    def __init__(self, n_qubits: int, input_scale: float, random_scale: float, seed: int):
        self.n = n_qubits
        self.input_scale = input_scale
        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.ry_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.zz_angles = np.triu(rng.normal(0.0, random_scale, size=(n_qubits, n_qubits)), 1)

    def _encode(self, state: np.ndarray, u: np.ndarray, factor: float = 1.0) -> np.ndarray:
        for q, val in enumerate(u):
            state = apply_one(state, ry(float(factor * self.input_scale * val)), q, self.n)
        return state

    def _ring_entangle(self, state: np.ndarray) -> np.ndarray:
        for i in range(self.n):
            state = apply_cnot(state, i, (i + 1) % self.n, self.n)
        return state

    def _random_layer(self, state: np.ndarray) -> np.ndarray:
        for q in range(self.n):
            state = apply_one(state, rz(float(self.rz_angles[q])), q, self.n)
            state = apply_one(state, ry(float(self.ry_angles[q])), q, self.n)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                state = apply_zz_phase(state, i, j, float(self.zz_angles[i, j]), self.n)
        return state

    def one(self, u: np.ndarray) -> np.ndarray:
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        state = self._encode(state, u, factor=1.0)
        state = self._ring_entangle(state)
        state = self._encode(state, u, factor=1.0)
        state = self._random_layer(state)
        return z_zz_features(state, self.n)

    def transform(self, U: np.ndarray) -> np.ndarray:
        return np.vstack([self.one(u) for u in U])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def repo_qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    target = np.maximum(np.asarray(y, dtype=float) ** 2, 1e-12)
    pred = np.maximum(np.asarray(yhat, dtype=float) ** 2, 1e-12)
    return float(np.mean(np.log(pred) + target / pred))


def effective_rank(A: np.ndarray) -> float:
    X = A - A.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    return float(np.exp(-np.sum(p * np.log(p + 1e-12))))


def regression_head(F: np.ndarray, y: np.ndarray, split: np.ndarray, alpha: float) -> np.ndarray:
    train = split == "train"
    scaler = StandardScaler().fit(F[train])
    Fs = scaler.transform(F)
    model = Ridge(alpha=alpha).fit(Fs[train], np.log(np.maximum(y[train], 1e-8)))
    return np.maximum(np.exp(model.predict(Fs)), 1e-8)


def evaluate_regression(y: np.ndarray, pred: np.ndarray, split: np.ndarray, thresholds: dict[str, float]) -> pd.DataFrame:
    rows = []
    for sp in ["train", "val", "test"]:
        m = split == sp
        row = {
            "head": "ridge_regression",
            "split": sp,
            "n": int(m.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y[m], pred[m]))),
            "qlike": repo_qlike(y[m], pred[m]),
            "corr": corr(y[m], pred[m]),
            "actual_std": float(y[m].std()),
            "pred_std": float(pred[m].std()),
            "pred_mean": float(pred[m].mean()),
        }
        for label, thr in thresholds.items():
            actual = y[m] >= thr
            called = pred[m] >= thr
            row[f"{label}_f1"] = float(f1_score(actual, called, zero_division=0))
            row[f"{label}_precision"] = float(precision_score(actual, called, zero_division=0))
            row[f"{label}_recall"] = float(recall_score(actual, called, zero_division=0))
            row[f"{label}_pred_rate"] = float(called.mean())
        if sp == "test":
            top = y[m] >= np.quantile(y[m], 0.80)
            row["top20_pred_actual_ratio"] = float(pred[m][top].mean() / y[m][top].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def best_threshold_from_val(y_val: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if len(thr) == 0:
        return 0.5, 0.0
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thr[idx]), float(f1[idx])


def classifier_head(
    F: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    threshold: float,
    event_name: str,
    C: float,
) -> tuple[pd.DataFrame, np.ndarray, float]:
    train = split == "train"
    val = split == "val"
    event = y >= threshold
    scaler = StandardScaler().fit(F[train])
    Fs = scaler.transform(F)
    clf = LogisticRegression(
        C=C,
        penalty="l2",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(Fs[train], event[train])
    prob = clf.predict_proba(Fs)[:, 1]
    decision_threshold, val_best_f1 = best_threshold_from_val(event[val], prob[val])

    rows = []
    for sp in ["train", "val", "test"]:
        m = split == sp
        called = prob[m] >= decision_threshold
        row = {
            "head": f"logistic_{event_name}",
            "event": event_name,
            "split": sp,
            "n": int(m.sum()),
            "event_rate": float(event[m].mean()),
            "prob_mean": float(prob[m].mean()),
            "prob_std": float(prob[m].std()),
            "decision_threshold": decision_threshold,
            "val_best_f1": val_best_f1,
            "called_rate": float(called.mean()),
            "precision": float(precision_score(event[m], called, zero_division=0)),
            "recall": float(recall_score(event[m], called, zero_division=0)),
            "f1": float(f1_score(event[m], called, zero_division=0)),
            "average_precision": float(average_precision_score(event[m], prob[m])),
        }
        try:
            row["roc_auc"] = float(roc_auc_score(event[m], prob[m]))
        except ValueError:
            row["roc_auc"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows), prob, decision_threshold


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default=None)
    p.add_argument("--target-col", default=TARGET_COL)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--input-scale", type=float, default=math.pi / 3)
    p.add_argument("--random-scale", type=float, default=0.35)
    p.add_argument("--ridge-alpha", type=float, default=3000.0)
    p.add_argument("--logistic-C", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root_from_cwd()
    results_dir = root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = find_dataset(root, args.data_path)
    print(f"Project root: {root}")
    print(f"Dataset: {data_path}")

    X, y, split, dates, feature_cols = load_matrix(data_path, args.target_col)
    U = make_level_rate_inputs(X, split, args.n_qubits)
    U = leaky_filter(U, args.leak)

    fmap = RFQRCRingMap(args.n_qubits, args.input_scale, args.random_scale, args.seed)
    F = fmap.transform(U)
    train = split == "train"
    thresholds = {
        "q80": float(np.quantile(y[train], 0.80)),
        "q90": float(np.quantile(y[train], 0.90)),
        "q95": float(np.quantile(y[train], 0.95)),
    }

    reg_pred = regression_head(F, y, split, args.ridge_alpha)
    reg_metrics = evaluate_regression(y, reg_pred, split, thresholds)
    reg_metrics.insert(0, "feature_map", "rf_qrc_ring_level_rate")
    reg_metrics.insert(1, "ridge_alpha", args.ridge_alpha)
    reg_metrics["effective_rank_train"] = effective_rank(F[train])

    clf_frames = []
    probs = {}
    thresholds_used = {}
    for event_name, thr in thresholds.items():
        clf_metrics, prob, decision_threshold = classifier_head(F, y, split, thr, event_name, args.logistic_C)
        clf_metrics.insert(0, "feature_map", "rf_qrc_ring_level_rate")
        clf_metrics.insert(1, "logistic_C", args.logistic_C)
        clf_metrics["effective_rank_train"] = effective_rank(F[train])
        clf_frames.append(clf_metrics)
        probs[event_name] = prob
        thresholds_used[event_name] = decision_threshold

    clf_metrics = pd.concat(clf_frames, ignore_index=True)
    preds = pd.DataFrame({
        "date": dates,
        "actual_future_rv_20d": y,
        "split": split,
        "rf_qrc_ring_ridge_pred": reg_pred,
        "rf_qrc_ring_q80_prob": probs["q80"],
        "rf_qrc_ring_q90_prob": probs["q90"],
        "rf_qrc_ring_q95_prob": probs["q95"],
    })
    for event_name, thr in thresholds.items():
        preds[f"actual_{event_name}_event"] = y >= thr
        preds[f"called_{event_name}_event"] = probs[event_name] >= thresholds_used[event_name]

    reg_path = results_dir / "phase3_one_qrc_two_head_regression_metrics.csv"
    clf_path = results_dir / "phase3_one_qrc_two_head_classifier_metrics.csv"
    pred_path = results_dir / "phase3_one_qrc_two_head_predictions.csv"
    test_path = results_dir / "phase3_one_qrc_two_head_test_summary.csv"

    reg_metrics.to_csv(reg_path, index=False)
    clf_metrics.to_csv(clf_path, index=False)
    preds.to_csv(pred_path, index=False)

    test_summary = pd.concat([
        reg_metrics.loc[reg_metrics["split"].eq("test")],
        clf_metrics.loc[clf_metrics["split"].eq("test")],
    ], ignore_index=True, sort=False)
    test_summary.to_csv(test_path, index=False)

    print("\nSaved:")
    print(reg_path)
    print(clf_path)
    print(pred_path)
    print(test_path)

    print("\nRegression test summary:")
    print(reg_metrics.loc[reg_metrics["split"].eq("test")].to_string(index=False))
    print("\nClassifier test summary:")
    show_cols = ["head", "event", "precision", "recall", "f1", "average_precision", "roc_auc", "called_rate", "event_rate", "decision_threshold"]
    print(clf_metrics.loc[clf_metrics["split"].eq("test"), show_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
