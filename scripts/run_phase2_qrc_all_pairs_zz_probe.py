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
    _fixed_disorder_factors,
    _safe_feature_target_correlations,
    encode_input_angles,
    evolve_tfim_step,
    expectation_x,
    expectation_z,
    expectation_zz,
    initialize_zero_state,
    make_qrc_sequence_splits,
    select_anchor_indices,
)


def leaky_integrate_windows(X: np.ndarray, leak: float = 0.3) -> np.ndarray:
    """Apply causal leaky integration within each rolling window."""
    out = np.zeros_like(X, dtype=float)
    for i, window in enumerate(X):
        h = np.zeros(window.shape[1], dtype=float)
        for t, u_t in enumerate(window):
            h = (1.0 - leak) * h + leak * u_t
            out[i, t] = h
    return out


def zz_pairs(n_qubits: int, mode: str) -> list[tuple[int, int]]:
    """Return ZZ readout pairs for nearest-neighbor or all-pairs readout."""
    if mode == "nearest":
        return [(q, q + 1) for q in range(n_qubits - 1)]
    if mode == "all":
        return [(qa, qb) for qa in range(n_qubits - 1) for qb in range(qa + 1, n_qubits)]
    raise ValueError(f"Unknown zz readout mode: {mode}")


def observable_features_with_zz_mode(
    state: np.ndarray,
    config: TFIMQRCConfig,
    *,
    zz_mode: str,
) -> np.ndarray:
    """Extract Z, X, and ZZ expectations with configurable ZZ pair coverage."""
    n = config.qubits
    feats: list[float] = []

    feats.extend(expectation_z(state, q, n) for q in range(n))

    if config.observable_mode in {"zx", "zxzz"}:
        feats.extend(expectation_x(state, q, n) for q in range(n))

    if config.observable_mode == "zxzz":
        feats.extend(expectation_zz(state, qa, qb, n) for qa, qb in zz_pairs(n, zz_mode))

    return np.asarray(feats, dtype=float)


def feature_names(config: TFIMQRCConfig, *, zz_mode: str) -> list[str]:
    """Names matching one observable readout block before temporal concatenation."""
    names = [f"Z{q}" for q in range(config.qubits)]
    if config.observable_mode in {"zx", "zxzz"}:
        names.extend(f"X{q}" for q in range(config.qubits))
    if config.observable_mode == "zxzz":
        names.extend(f"ZZ{qa}_{qb}" for qa, qb in zz_pairs(config.qubits, zz_mode))
    return names


def expanded_feature_names(config: TFIMQRCConfig, *, zz_mode: str) -> list[str]:
    """Feature names after anchor/virtual-node concatenation."""
    block = feature_names(config, zz_mode=zz_mode)
    if not config.collect_anchor_features:
        return block
    n_blocks = config.anchor_count * config.virtual_nodes_per_anchor
    return [f"block{b:02d}:{name}" for b in range(n_blocks) for name in block]


def run_window_with_zz_mode(
    window: np.ndarray,
    config: TFIMQRCConfig,
    *,
    zz_mode: str,
) -> np.ndarray:
    """Run one window through the TFIM reservoir with configurable ZZ readout."""
    if window.ndim != 2:
        raise ValueError(f"Expected window shape (lookback, features), got {window.shape}")
    if config.virtual_nodes_per_anchor < 1:
        raise ValueError("virtual_nodes_per_anchor must be >= 1")
    if config.virtual_nodes_per_anchor > config.trotter_steps_per_anchor:
        raise ValueError("virtual_nodes_per_anchor must be <= trotter_steps_per_anchor")

    state = initialize_zero_state(config.qubits)
    anchor_indices = select_anchor_indices(
        config.lookback_days,
        config.anchor_count,
        config.anchor_policy,
    )
    edge_factors, field_factors = _fixed_disorder_factors(config)

    anchor_features: list[np.ndarray] = []
    readout_steps = set(
        np.linspace(
            1,
            config.trotter_steps_per_anchor,
            config.virtual_nodes_per_anchor,
        ).round().astype(int)
    )

    for idx in anchor_indices:
        state = encode_input_angles(
            state,
            window[idx],
            n_qubits=config.qubits,
            angle_max=config.angle_max,
        )
        for step in range(1, config.trotter_steps_per_anchor + 1):
            state = evolve_tfim_step(
                state,
                config,
                edge_factors=edge_factors,
                field_factors=field_factors,
            )
            if config.collect_anchor_features and step in readout_steps:
                anchor_features.append(observable_features_with_zz_mode(state, config, zz_mode=zz_mode))

    if config.collect_anchor_features:
        return np.concatenate(anchor_features)

    return observable_features_with_zz_mode(state, config, zz_mode=zz_mode)


def build_feature_matrix_with_zz_mode(
    X_windows: np.ndarray,
    config: TFIMQRCConfig,
    *,
    zz_mode: str,
    verbose: bool = False,
) -> np.ndarray:
    rows = []
    for i, window in enumerate(X_windows):
        if verbose and i % 250 == 0:
            print(f"QRC {zz_mode} sample {i}/{len(X_windows)}")
        rows.append(run_window_with_zz_mode(window, config, zz_mode=zz_mode))
    return np.asarray(rows, dtype=float)


def fit_readout(
    H_train_raw: np.ndarray,
    H_val_raw: np.ndarray,
    H_test_raw: np.ndarray,
    y_train: np.ndarray,
    *,
    top_k: int = 240,
    alpha: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train-only winsorization, top-k feature selection, scaling, and ridge readout."""
    lower = np.percentile(H_train_raw, 1.0, axis=0)
    upper = np.percentile(H_train_raw, 99.0, axis=0)

    H_train = np.clip(H_train_raw, lower, upper)
    H_val = np.clip(H_val_raw, lower, upper)
    H_test = np.clip(H_test_raw, lower, upper)

    corr = _safe_feature_target_correlations(H_train, y_train)
    selected_idx = np.argsort(np.abs(corr))[-min(top_k, H_train.shape[1]) :]

    H_train = H_train[:, selected_idx]
    H_val = H_val[:, selected_idx]
    H_test = H_test[:, selected_idx]

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
        selected_idx,
    )


def high_vol_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, float]:
    """Binary high-vol diagnostics at a train-defined actual/predicted threshold."""
    actual = y_true >= threshold
    pred = y_pred >= threshold
    tp = int(np.sum(actual & pred))
    fp = int(np.sum(~actual & pred))
    fn = int(np.sum(actual & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "support": int(np.sum(actual))}


def summarize_feature_selection(selected_idx: np.ndarray, names: list[str]) -> dict[str, int | float]:
    selected_names = [names[i] for i in selected_idx]
    is_all_zz = [name.split(":", 1)[-1].startswith("ZZ") for name in selected_names]
    is_long_range_zz = []
    for name in selected_names:
        obs = name.split(":", 1)[-1]
        if not obs.startswith("ZZ"):
            is_long_range_zz.append(False)
            continue
        qa, qb = obs.replace("ZZ", "").split("_")
        is_long_range_zz.append(abs(int(qb) - int(qa)) > 1)

    return {
        "selected_features": int(len(selected_idx)),
        "selected_zz_features": int(np.sum(is_all_zz)),
        "selected_long_range_zz_features": int(np.sum(is_long_range_zz)),
        "selected_long_range_zz_share": float(np.mean(is_long_range_zz)) if selected_idx.size else 0.0,
    }


def main() -> None:
    target = "future_rv_20d"
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

    metrics_rows: list[dict] = []
    high_vol_rows: list[dict] = []
    selection_rows: list[dict] = []
    prediction_tables: list[pd.DataFrame] = []

    thresholds = {f"q{q}": float(np.quantile(y_train, q / 100.0)) for q in [80, 90, 95]}

    for zz_mode in ["nearest", "all"]:
        run_name = f"linear_clip_top240_alpha1000_zz_{zz_mode}"
        print(f"\nBuilding QRC features: {zz_mode} ZZ readout")
        H_train_raw = build_feature_matrix_with_zz_mode(X_train, config, zz_mode=zz_mode, verbose=True)
        H_val_raw = build_feature_matrix_with_zz_mode(X_val, config, zz_mode=zz_mode, verbose=True)
        H_test_raw = build_feature_matrix_with_zz_mode(X_test, config, zz_mode=zz_mode, verbose=True)

        pred_train, pred_val, pred_test, selected_idx = fit_readout(
            H_train_raw,
            H_val_raw,
            H_test_raw,
            y_train,
            top_k=240,
            alpha=1000.0,
        )

        names = expanded_feature_names(config, zz_mode=zz_mode)
        selection_summary = summarize_feature_selection(selected_idx, names)
        selection_summary.update(
            {
                "run_name": run_name,
                "zz_mode": zz_mode,
                "n_raw_features": int(H_train_raw.shape[1]),
            }
        )
        selection_rows.append(selection_summary)

        for split_name, y, pred in [
            ("train", y_train, pred_train),
            ("val", y_val, pred_val),
            ("test", y_test, pred_test),
        ]:
            m = evaluate_volatility_forecast(y, pred)
            metrics_rows.append(
                {
                    "run_name": run_name,
                    "zz_mode": zz_mode,
                    "split": split_name,
                    "rmse": m.rmse,
                    "qlike": m.qlike,
                    "mz_r2": m.mz_r2,
                    "corr": float(np.corrcoef(y, pred)[0, 1]),
                    "pred_std": float(np.std(pred)),
                }
            )

            for q_name, threshold in thresholds.items():
                row = high_vol_metrics(y, pred, threshold)
                row.update(
                    {
                        "run_name": run_name,
                        "zz_mode": zz_mode,
                        "split": split_name,
                        "quantile": q_name,
                        "threshold": threshold,
                    }
                )
                high_vol_rows.append(row)

        prediction_tables.append(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(test_dates),
                    "actual_future_rv_20d": y_test,
                    "qrc_pred_future_rv_20d": pred_test,
                    "run_name": run_name,
                    "zz_mode": zz_mode,
                }
            )
        )

    metrics = pd.DataFrame(metrics_rows)
    high_vol = pd.DataFrame(high_vol_rows)
    selections = pd.DataFrame(selection_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)

    metrics_path = out_dir / "phase2_qrc_all_pairs_zz_probe_metrics.csv"
    high_vol_path = out_dir / "phase2_qrc_all_pairs_zz_probe_high_vol.csv"
    selections_path = out_dir / "phase2_qrc_all_pairs_zz_probe_feature_selection.csv"
    predictions_path = out_dir / "phase2_qrc_all_pairs_zz_probe_predictions.csv"

    metrics.to_csv(metrics_path, index=False)
    high_vol.to_csv(high_vol_path, index=False)
    selections.to_csv(selections_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    print("\nMetrics:")
    print(metrics)
    print("\nHigh-vol diagnostics:")
    print(high_vol)
    print("\nFeature-selection diagnostics:")
    print(selections)
    print("\nSaved:")
    print(metrics_path)
    print(high_vol_path)
    print(selections_path)
    print(predictions_path)


if __name__ == "__main__":
    main()
