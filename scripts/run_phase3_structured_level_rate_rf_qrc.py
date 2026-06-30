"""Phase 3 structured level-rate RF-QRC experiment.

Option B test:
    keep level and rate inputs on separate qubit blocks, and apply entanglement
    primarily across the blocks so the RF-QRC feature map computes level x rate
    couplings instead of indiscriminate ring correlations.

This is simulator-only and small: 6 qubits, statevector features, ridge readout.
Outputs are written to root-level results/tables and results/figures by the
notebook runner.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_squared_error, precision_score, recall_score
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
    raise FileNotFoundError(
        "No Phase 2 modeling dataset found. Run scripts/prepare_phase2_spy_vix_dataset.py "
        "or pass --data-path with a CSV containing future_rv_20d."
    )


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
    return work, X, y, split, dates, feature_cols


def make_level_rate_inputs(X: np.ndarray, split: np.ndarray, n_qubits: int) -> np.ndarray:
    if n_qubits % 2 != 0:
        raise ValueError("Structured level-rate experiment requires an even number of qubits.")
    k = n_qubits // 2
    train = split == "train"
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


def entangler_pairs(mode: str, n: int) -> list[tuple[int, int]]:
    k = n // 2
    level = list(range(k))
    rate = list(range(k, n))

    if mode == "ring":
        return [(i, (i + 1) % n) for i in range(n)]
    if mode == "cross_matched":
        return [(level[i], rate[i]) for i in range(k)]
    if mode == "cross_all":
        return [(i, j) for i in level for j in rate]
    if mode == "block_plus_cross":
        within_level = [(level[i], level[i + 1]) for i in range(k - 1)]
        within_rate = [(rate[i], rate[i + 1]) for i in range(k - 1)]
        cross = [(i, j) for i in level for j in rate]
        return within_level + within_rate + cross
    if mode == "none":
        return []
    raise ValueError(f"Unknown entangler mode: {mode}")


class StructuredLevelRateRFQRCMap:
    def __init__(
        self,
        n_qubits: int,
        entangler: str,
        input_scale: float,
        level_scale: float,
        rate_scale: float,
        random_scale: float,
        cross_zz_boost: float,
        weak_within_boost: float,
        seed: int,
    ):
        if n_qubits % 2 != 0:
            raise ValueError("n_qubits must be even.")
        self.n = n_qubits
        self.k = n_qubits // 2
        self.entangler = entangler
        self.input_scale = input_scale
        self.level_scale = level_scale
        self.rate_scale = rate_scale
        self.random_scale = random_scale
        self.cross_zz_boost = cross_zz_boost
        self.weak_within_boost = weak_within_boost

        rng = np.random.default_rng(seed)
        self.rz_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.ry_angles = rng.normal(0.0, random_scale, size=n_qubits)
        self.zz_angles = np.triu(rng.normal(0.0, random_scale, size=(n_qubits, n_qubits)), 1)

    def _encode(self, state: np.ndarray, u: np.ndarray, second: bool = False) -> np.ndarray:
        second_factor = 0.65 if second else 1.0
        for q, val in enumerate(u):
            channel_scale = self.level_scale if q < self.k else self.rate_scale
            theta = self.input_scale * channel_scale * second_factor * float(val)
            state = apply_one(state, ry(theta), q, self.n)
        return state

    def _entangle(self, state: np.ndarray) -> np.ndarray:
        for i, j in entangler_pairs(self.entangler, self.n):
            state = apply_cnot(state, i, j, self.n)
        return state

    def _random_layer(self, state: np.ndarray) -> np.ndarray:
        for q in range(self.n):
            state = apply_one(state, rz(float(self.rz_angles[q])), q, self.n)
            state = apply_one(state, ry(float(self.ry_angles[q])), q, self.n)

        for i in range(self.n):
            for j in range(i + 1, self.n):
                is_cross = (i < self.k <= j) or (j < self.k <= i)
                boost = self.cross_zz_boost if is_cross else self.weak_within_boost
                state = apply_zz_phase(state, i, j, float(boost * self.zz_angles[i, j]), self.n)
        return state

    def one(self, u: np.ndarray) -> np.ndarray:
        state = np.zeros(2**self.n, dtype=complex)
        state[0] = 1.0
        state = self._encode(state, u, second=False)
        state = self._entangle(state)
        state = self._encode(state, u, second=True)
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


def fit_eval(F: np.ndarray, y: np.ndarray, split: np.ndarray, alpha: float) -> tuple[pd.DataFrame, np.ndarray]:
    train = split == "train"
    scaler = StandardScaler().fit(F[train])
    Fs = scaler.transform(F)
    model = Ridge(alpha=alpha).fit(Fs[train], np.log(np.maximum(y[train], 1e-8)))
    pred = np.maximum(np.exp(model.predict(Fs)), 1e-8)

    qs = {
        "q80": float(np.quantile(y[train], 0.80)),
        "q90": float(np.quantile(y[train], 0.90)),
        "q95": float(np.quantile(y[train], 0.95)),
    }
    rows = []
    for sp in ["train", "val", "test"]:
        m = split == sp
        row = {
            "split": sp,
            "n": int(m.sum()),
            "rmse": float(np.sqrt(mean_squared_error(y[m], pred[m]))),
            "qlike": repo_qlike(y[m], pred[m]),
            "corr": corr(y[m], pred[m]),
            "actual_mean": float(y[m].mean()),
            "actual_std": float(y[m].std()),
            "pred_mean": float(pred[m].mean()),
            "pred_std": float(pred[m].std()),
            "effective_rank": effective_rank(F[m]),
        }
        for label, thr in qs.items():
            actual = y[m] >= thr
            called = pred[m] >= thr
            row[f"{label}_actual_rate"] = float(actual.mean())
            row[f"{label}_pred_rate"] = float(called.mean())
            row[f"{label}_f1"] = float(f1_score(actual, called, zero_division=0))
            row[f"{label}_precision"] = float(precision_score(actual, called, zero_division=0))
            row[f"{label}_recall"] = float(recall_score(actual, called, zero_division=0))
        if sp == "test":
            top = y[m] >= np.quantile(y[m], 0.80)
            row["top20_actual_mean"] = float(y[m][top].mean())
            row["top20_pred_mean"] = float(pred[m][top].mean())
            row["top20_pred_actual_ratio"] = row["top20_pred_mean"] / row["top20_actual_mean"]
        rows.append(row)
    return pd.DataFrame(rows), pred


def run_variant(name: str, U: np.ndarray, y: np.ndarray, split: np.ndarray, entangler: str, args):
    fmap = StructuredLevelRateRFQRCMap(
        n_qubits=U.shape[1],
        entangler=entangler,
        input_scale=args.input_scale,
        level_scale=args.level_scale,
        rate_scale=args.rate_scale,
        random_scale=args.random_scale,
        cross_zz_boost=args.cross_zz_boost,
        weak_within_boost=args.weak_within_boost,
        seed=args.seed,
    )
    F = fmap.transform(U)
    metrics, pred = fit_eval(F, y, split, args.ridge_alpha)
    metrics.insert(0, "run_name", name)
    metrics.insert(1, "entangler", entangler)
    return metrics, pred


def parse_args(argv: Iterable[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default=None)
    p.add_argument("--target-col", default=TARGET_COL)
    p.add_argument("--n-qubits", type=int, default=6)
    p.add_argument("--leak", type=float, default=0.3)
    p.add_argument("--input-scale", type=float, default=math.pi / 3)
    p.add_argument("--level-scale", type=float, default=1.0)
    p.add_argument("--rate-scale", type=float, default=0.75)
    p.add_argument("--random-scale", type=float, default=0.35)
    p.add_argument("--cross-zz-boost", type=float, default=1.0)
    p.add_argument("--weak-within-boost", type=float, default=0.35)
    p.add_argument("--ridge-alpha", type=float, default=3000.0)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument(
        "--entanglers",
        nargs="+",
        default=["ring", "cross_matched", "cross_all", "block_plus_cross"],
        choices=["none", "ring", "cross_matched", "cross_all", "block_plus_cross"],
    )
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    root = project_root_from_cwd()
    results_dir = root / "results" / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_path = find_dataset(root, args.data_path)
    print(f"Project root: {root}")
    print(f"Dataset: {data_path}")
    print(f"Entanglers: {args.entanglers}")

    _, X, y, split, dates, feature_cols = load_matrix(data_path, args.target_col)
    U = make_level_rate_inputs(X, split, args.n_qubits)
    U = leaky_filter(U, args.leak)

    metrics_all = []
    preds = pd.DataFrame({"date": dates, "actual_future_rv_20d": y, "split": split})

    for entangler in args.entanglers:
        run_name = f"structured_level_rate_rf_qrc_{entangler}"
        print(f"Running {run_name}")
        metrics, pred = run_variant(run_name, U, y, split, entangler, args)
        metrics_all.append(metrics)
        preds[f"{run_name}_pred"] = pred

    metrics_df = pd.concat(metrics_all, ignore_index=True)

    metrics_path = results_dir / "phase3_structured_level_rate_rf_qrc_metrics.csv"
    pred_path = results_dir / "phase3_structured_level_rate_rf_qrc_predictions.csv"
    summary_path = results_dir / "phase3_structured_level_rate_rf_qrc_test_summary.csv"

    metrics_df.to_csv(metrics_path, index=False)
    preds.to_csv(pred_path, index=False)
    metrics_df.loc[metrics_df["split"].eq("test")].to_csv(summary_path, index=False)

    print("\nSaved:")
    print(metrics_path)
    print(pred_path)
    print(summary_path)

    show_cols = [
        "run_name",
        "rmse",
        "qlike",
        "corr",
        "actual_std",
        "pred_std",
        "q80_f1",
        "q90_f1",
        "q95_f1",
        "top20_pred_actual_ratio",
        "effective_rank",
    ]
    print("\nTest summary:")
    print(metrics_df.loc[metrics_df["split"].eq("test"), show_cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
