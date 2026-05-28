from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.pca import fit_transform_pca_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.qrc.tfim_reservoir import (
    TFIMQRCConfig,
    fit_tfim_qrc_regressor,
    make_qrc_sequence_splits,
)


def metrics_row(run_name: str, split: str, metrics, pred: np.ndarray) -> dict:
    return {
        "run_name": run_name,
        "split": split,
        "rmse": metrics.rmse,
        "qlike": metrics.qlike,
        "mz_r2": metrics.mz_r2,
        "corr": np.nan,  # filled below when actual arrays are available
        "pred_std": float(np.std(pred)),
    }


def corr_safe(y: np.ndarray, pred: np.ndarray) -> float:
    if np.std(y) == 0 or np.std(pred) == 0:
        return float("nan")
    return float(np.corrcoef(y, pred)[0, 1])


def main():
    target = "future_rv_20d"
    run_name = "qrc_v0_default_tfim_anchor_snapshot"

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

    seq = make_qrc_sequence_splits(
        pca6.splits,
        feature_columns=pca6.feature_columns,
        target_column=target,
        lookback_days=40,
    )
    print({k: (v[0].shape, v[1].shape) for k, v in seq.items()})

    # Frozen initial QRC prototype.
    # This intentionally uses the TFIMQRCConfig defaults preserved in the module
    # as the May-25/v1 behavior: chain topology, even temporal anchors,
    # Z-only readout, one Trotter step per anchor, no disorder, no virtual nodes.
    # Do not tune this script; it exists only for progress/ablation storytelling.
    config = TFIMQRCConfig(
        qubits=6,
        pca_components=6,
        lookback_days=40,
        anchor_count=6,
        anchor_policy="even",
        observable_mode="z",
        trotter_steps_per_anchor=1,
        virtual_nodes_per_anchor=1,
        topology="chain",
        coupling_scale=0.7,
        transverse_field=0.5,
        evolution_time=0.5,
        angle_max=np.pi / 2,
        ridge_alpha=10.0,
        target_transform="log",
        seed=42,
        collect_anchor_features=False,
        use_disorder=False,
        disorder_strength=0.0,
    )

    print("Fitting frozen QRC v0 prototype. This may take a few minutes...")
    result = fit_tfim_qrc_regressor(seq, config=config, target=target, verbose=True)

    X_train, y_train, train_dates = seq["train"]
    X_val, y_val, val_dates = seq["val"]
    X_test, y_test, test_dates = seq["test"]

    metrics = pd.DataFrame(
        [
            {
                **metrics_row(run_name, "train", result.train_metrics, result.train_predictions),
                "corr": corr_safe(y_train, result.train_predictions),
                "n_reservoir_features": result.train_features.shape[1],
            },
            {
                **metrics_row(run_name, "val", result.val_metrics, result.val_predictions),
                "corr": corr_safe(y_val, result.val_predictions),
                "n_reservoir_features": result.val_features.shape[1],
            },
            {
                **metrics_row(run_name, "test", result.test_metrics, result.test_predictions),
                "corr": corr_safe(y_test, result.test_predictions),
                "n_reservoir_features": result.test_features.shape[1],
            },
        ]
    )
    metrics.to_csv(out_dir / "phase2_qrc_anchor_snapshot_prediction_export_metrics.csv", index=False)

    pred_table = pd.DataFrame(
        {
            "date": pd.to_datetime(test_dates),
            "actual_future_rv_20d": y_test,
            "qrc_v0_pred_future_rv_20d": result.test_predictions,
            "run_name": run_name,
        }
    )
    pred_table.to_csv(out_dir / "phase2_qrc_anchor_snapshot_predictions.csv", index=False)

    print(metrics.to_string(index=False))
    print(pred_table.head().to_string(index=False))
    print("Saved frozen QRC v0 row-level predictions.")


if __name__ == "__main__":
    main()
