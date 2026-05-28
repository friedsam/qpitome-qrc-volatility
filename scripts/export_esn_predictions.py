from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS, make_sequence_arrays
from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.pca import fit_transform_pca_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast


def spectral_scale(W: np.ndarray, radius: float) -> np.ndarray:
    eig = np.linalg.eigvals(W)
    rho = np.max(np.abs(eig))
    return W * (radius / max(rho, 1e-12))


def make_esn_weights(n_inputs: int, n_reservoir: int, spectral_radius: float, input_scale: float, seed: int):
    rng = np.random.default_rng(seed)
    W_in = rng.normal(0.0, input_scale, size=(n_reservoir, n_inputs))
    W = rng.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    mask = rng.random(W.shape) < 0.10
    W = spectral_scale(W * mask, spectral_radius)
    return W_in, W


def esn_states(X: np.ndarray, W_in: np.ndarray, W: np.ndarray, leak: float) -> np.ndarray:
    rows = []
    for window in X:
        h = np.zeros(W.shape[0])
        for u_t in window:
            h_new = np.tanh(W_in @ u_t + W @ h)
            h = (1.0 - leak) * h + leak * h_new
        rows.append(np.concatenate([h, window[-1]]))
    return np.asarray(rows)


def fit_predict(H_train, H_val, H_test, y_train, alpha: float):
    scaler = StandardScaler()
    H_train_s = scaler.fit_transform(H_train)
    H_val_s = scaler.transform(H_val)
    H_test_s = scaler.transform(H_test)
    model = Ridge(alpha=alpha)
    model.fit(H_train_s, np.log(np.maximum(y_train, 1e-8)))
    return (
        np.exp(model.predict(H_train_s)),
        np.exp(model.predict(H_val_s)),
        np.exp(model.predict(H_test_s)),
    )


def metric_row(run_name, split, y, pred):
    m = evaluate_volatility_forecast(y, pred)
    return {
        "run_name": run_name,
        "split": split,
        "rmse": m.rmse,
        "qlike": m.qlike,
        "mz_r2": m.mz_r2,
        "corr": float(np.corrcoef(y, pred)[0, 1]),
        "pred_std": float(np.std(pred)),
    }


def main():
    target = "future_rv_20d"
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_phase2_volatility_data()
    splits = chronological_tabular_split(df)
    pca6 = fit_transform_pca_splits_train_only(
        splits,
        feature_columns=FEATURE_COLUMNS,
        target_columns=[target],
        n_components=6,
        prefix="pca6",
    )
    seq = {
        name: make_sequence_arrays(split, feature_columns=pca6.feature_columns, target_column=target, lookback=40)
        for name, split in pca6.splits.items()
    }
    X_train, y_train, train_dates = seq["train"]
    X_val, y_val, val_dates = seq["val"]
    X_test, y_test, test_dates = seq["test"]
    print({k: (v[0].shape, v[1].shape) for k, v in seq.items()})

    # Small validation-selected ESN grid. The best val RMSE run is exported.
    grid = [
        {"n": 300, "sr": 0.70, "inp": 0.30, "leak": 0.30, "alpha": 300.0, "seed": 42},
        {"n": 300, "sr": 0.90, "inp": 0.30, "leak": 0.30, "alpha": 1000.0, "seed": 42},
        {"n": 500, "sr": 0.70, "inp": 0.20, "leak": 0.50, "alpha": 1000.0, "seed": 42},
        {"n": 500, "sr": 0.90, "inp": 0.20, "leak": 0.50, "alpha": 3000.0, "seed": 42},
    ]

    rows = []
    best = None
    for cfg in grid:
        run_name = f"esn_n{cfg['n']}_sr{cfg['sr']}_inp{cfg['inp']}_leak{cfg['leak']}_alpha{cfg['alpha']}"
        print("Running", run_name)
        W_in, W = make_esn_weights(X_train.shape[2], cfg["n"], cfg["sr"], cfg["inp"], cfg["seed"])
        H_train = esn_states(X_train, W_in, W, cfg["leak"])
        H_val = esn_states(X_val, W_in, W, cfg["leak"])
        H_test = esn_states(X_test, W_in, W, cfg["leak"])
        pred_train, pred_val, pred_test = fit_predict(H_train, H_val, H_test, y_train, cfg["alpha"])
        for split, y, pred in [("train", y_train, pred_train), ("val", y_val, pred_val), ("test", y_test, pred_test)]:
            row = metric_row(run_name, split, y, pred)
            row.update(cfg)
            rows.append(row)
        val_rmse = rows[-2]["rmse"]
        if best is None or val_rmse < best["val_rmse"]:
            best = {"run_name": run_name, "cfg": cfg, "val_rmse": val_rmse, "pred": (pred_train, pred_val, pred_test)}

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "phase2_esn_prediction_export_metrics.csv", index=False)

    pred_train, pred_val, pred_test = best["pred"]
    pred_table = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "actual_future_rv_20d": y_test,
            "esn_pred_future_rv_20d": pred_test,
            "run_name": best["run_name"],
        }
    )
    pred_table.to_csv(out_dir / "phase2_esn_predictions.csv", index=False)

    print("Best ESN:", best["run_name"])
    print(metrics[metrics["run_name"] == best["run_name"]].to_string(index=False))
    print(pred_table.head().to_string(index=False))


if __name__ == "__main__":
    main()
