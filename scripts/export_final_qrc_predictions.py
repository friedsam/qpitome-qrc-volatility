from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.pca import fit_transform_pca_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.evaluation.metrics import evaluate_volatility_forecast
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig,
    _safe_feature_target_correlations,
    build_qrc_feature_matrix,
    make_qrc_sequence_splits,
)


def leaky_integrate_windows(X, leak=0.3):
    out = np.zeros_like(X, dtype=float)
    for i, window in enumerate(X):
        h = np.zeros(window.shape[1], dtype=float)
        for t, u_t in enumerate(window):
            h = (1.0 - leak) * h + leak * u_t
            out[i, t] = h
    return out


def fit_readout(H_train_raw, H_val_raw, H_test_raw, y_train, top_k=240, alpha=1000.0):
    lower = np.percentile(H_train_raw, 1.0, axis=0)
    upper = np.percentile(H_train_raw, 99.0, axis=0)

    H_train = np.clip(H_train_raw, lower, upper)
    H_val = np.clip(H_val_raw, lower, upper)
    H_test = np.clip(H_test_raw, lower, upper)

    corr = _safe_feature_target_correlations(H_train, y_train)
    idx = np.argsort(np.abs(corr))[-min(top_k, H_train.shape[1]):]

    H_train = H_train[:, idx]
    H_val = H_val[:, idx]
    H_test = H_test[:, idx]

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


def main():
    target = "future_rv_20d"
    run_name = "linear_clip_top240_alpha1000"

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_phase2_volatility_data()
    splits = chronological_tabular_split(df)

    print("Fitting PCA-6...")
    pca6 = fit_transform_pca_splits_train_only(
        splits,
        feature_columns=FEATURE_COLUMNS,
        target_columns=[target],
        n_components=6,
        prefix="pca6",
    )

    raw_seq = make_qrc_sequence_splits(
        pca6.splits,
        feature_columns=pca6.feature_columns,
        target_column=target,
        lookback_days=40,
    )

    leaky_seq = {
        split: (leaky_integrate_windows(X, leak=0.3), y, dates)
        for split, (X, y, dates) in raw_seq.items()
    }

    X_train, y_train, train_dates = leaky_seq["train"]
    X_val, y_val, val_dates = leaky_seq["val"]
    X_test, y_test, test_dates = leaky_seq["test"]

    print({k: (v[0].shape, v[1].shape) for k, v in leaky_seq.items()})

    config = TFIMQRCConfig(
        qubits=6,
        pca_components=6,
        lookback_days=40,
        anchor_count=10,
        anchor_policy="recent",
        observable_mode="zxzz",
        collect_anchor_features=True,
        topology="full",
        trotter_steps_per_anchor=3,
        virtual_nodes_per_anchor=3,
        coupling_scale=0.7,
        transverse_field=0.5,
        evolution_time=0.5,
        angle_max=np.pi / 2,
        ridge_alpha=1000.0,
        target_transform="log",
        seed=42,
        use_disorder=True,
        disorder_strength=0.20,
    )

    print("Building QRC features. This may take a few minutes...")
    H_train_raw = build_qrc_feature_matrix(X_train, config, verbose=True)
    H_val_raw = build_qrc_feature_matrix(X_val, config, verbose=True)
    H_test_raw = build_qrc_feature_matrix(X_test, config, verbose=True)

    pred_train, pred_val, pred_test = fit_readout(
        H_train_raw,
        H_val_raw,
        H_test_raw,
        y_train,
        top_k=240,
        alpha=1000.0,
    )

    metrics = []
    for split_name, y, pred in [
        ("train", y_train, pred_train),
        ("val", y_val, pred_val),
        ("test", y_test, pred_test),
    ]:
        m = evaluate_volatility_forecast(y, pred)
        metrics.append(
            {
                "run_name": run_name,
                "split": split_name,
                "rmse": m.rmse,
                "qlike": m.qlike,
                "mz_r2": m.mz_r2,
                "corr": float(np.corrcoef(y, pred)[0, 1]),
                "pred_std": float(np.std(pred)),
            }
        )

    metrics = pd.DataFrame(metrics)
    metrics.to_csv(
        out_dir / "phase2_qrc_final_encoding_readout_prediction_export_metrics.csv",
        index=False,
    )

    pred_table = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "actual_future_rv_20d": y_test,
            "qrc_pred_future_rv_20d": pred_test,
            "run_name": run_name,
        }
    )

    pred_table.to_csv(
        out_dir / "phase2_qrc_final_encoding_readout_predictions.csv",
        index=False,
    )

    print(metrics)
    print(pred_table.head())
    print("Saved row-level predictions for Section 9.")


if __name__ == "__main__":
    main()