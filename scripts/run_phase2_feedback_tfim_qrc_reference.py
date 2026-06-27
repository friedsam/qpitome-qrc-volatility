from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.data.features import FEATURE_COLUMNS
from qpitome_qrc.data.loaders import load_phase2_volatility_data
from qpitome_qrc.data.pca import fit_transform_pca_splits_train_only
from qpitome_qrc.data.splits import chronological_tabular_split
from qpitome_qrc.qrc.feedback_tfim_reservoir import (
    FeedbackTFIMQRCConfig,
    build_feedback_qrc_feature_matrix,
    fit_feedback_tfim_qrc_regressor,
    make_qrc_sequence_splits,
    summarize_feedback_qrc_result,
)

TARGET_COLUMN = "future_rv_20d"
DEFAULT_OUTPUT_DIR = Path("results/tables")


def make_reference_config() -> FeedbackTFIMQRCConfig:
    """Return the canonical Phase 2 feedback-TFIM QRC reference config."""
    return FeedbackTFIMQRCConfig(
        qubits=6,
        pca_components=6,
        lookback_days=40,
        temporal_steps=10,
        temporal_policy="recent",
        input_qubits=(0, 1),
        memory_qubits=(2, 3, 4),
        readout_qubits=(5,),
        observable_mode="zxzz",
        feature_collection="trajectory",
        trotter_steps_per_time=3,
        evolution_time=0.50,
        input_scale=np.pi / 2,
        transverse_field=0.5,
        input_memory_coupling_scale=1.2,
        memory_coupling_scale=1.0,
        readout_coupling_scale=0.7,
        weak_background_coupling_scale=0.15,
        feedback_gain=0.0,
        ridge_alpha=1000.0,
        target_transform="log",
        seed=42,
        disorder_strength=0.20,
    )


def make_tiny_validation_config() -> FeedbackTFIMQRCConfig:
    """Return a fast config for command-line validation and tests."""
    return FeedbackTFIMQRCConfig(
        qubits=3,
        pca_components=2,
        lookback_days=4,
        temporal_steps=2,
        input_qubits=(0,),
        memory_qubits=(1,),
        readout_qubits=(2,),
        observable_mode="zxzz",
        feature_collection="trajectory",
        trotter_steps_per_time=1,
        evolution_time=0.10,
        disorder_strength=0.0,
        seed=42,
        ridge_alpha=1.0,
    )


def leaky_integrate_windows(X: np.ndarray, leak: float = 0.3) -> np.ndarray:
    """Apply train-independent leaky integration to sequence windows."""
    out = np.zeros_like(X, dtype=float)
    for i, window in enumerate(X):
        h = np.zeros(window.shape[1], dtype=float)
        for t, u_t in enumerate(window):
            h = (1.0 - leak) * h + leak * u_t
            out[i, t] = h
    return out


def prepare_phase2_sequence_splits(
    *,
    target_column: str = TARGET_COLUMN,
    leak: float = 0.3,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.Series]]:
    """Load data, make chronological splits, apply train-only PCA, and build QRC windows."""
    df = load_phase2_volatility_data()
    splits = chronological_tabular_split(df)

    pca = fit_transform_pca_splits_train_only(
        splits,
        feature_columns=FEATURE_COLUMNS,
        target_columns=[target_column],
        n_components=6,
        prefix="pca6",
    )

    raw_sequence_splits = make_qrc_sequence_splits(
        pca.splits,
        feature_columns=pca.feature_columns,
        target_column=target_column,
        lookback_days=40,
    )

    return {
        split_name: (leaky_integrate_windows(X, leak=leak), y, dates)
        for split_name, (X, y, dates) in raw_sequence_splits.items()
    }


def make_tiny_sequence_splits() -> dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    """Build a deterministic tiny dataset for validating the script path."""
    base_window = np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [0.1, 0.0],
            [-0.2, 0.3],
        ],
        dtype=float,
    )
    offsets = np.arange(12, dtype=float) * 0.05
    X = np.stack([base_window + offset for offset in offsets], axis=0)
    y = np.array(
        [0.10, 0.11, 0.13, 0.15, 0.18, 0.20, 0.23, 0.26, 0.30, 0.34, 0.38, 0.42],
        dtype=float,
    )
    dates = pd.date_range("2020-01-01", periods=len(y), freq="D")
    return {
        "train": (X[:6], y[:6], dates[:6]),
        "val": (X[6:9], y[6:9], dates[6:9]),
        "test": (X[9:], y[9:], dates[9:]),
    }


def prediction_frame(
    *,
    split_name: str,
    dates,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_name: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split_name,
            "date": pd.to_datetime(dates),
            "actual_future_rv_20d": y_true,
            "qrc_pred_future_rv_20d": y_pred,
            "run_name": run_name,
        }
    )


def run_reference(
    *,
    quick_check: bool = False,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_outputs: bool = True,
) -> tuple[dict, pd.DataFrame]:
    """Run either the tiny validation path or the full Phase 2 feedback-QRC reference path."""
    if quick_check:
        run_name = "quick_check_feedback_tfim_qrc"
        config = make_tiny_validation_config()
        sequence_splits = make_tiny_sequence_splits()
    else:
        run_name = "phase2_feedback_tfim_qrc_reference"
        config = make_reference_config()
        sequence_splits = prepare_phase2_sequence_splits()

    result = fit_feedback_tfim_qrc_regressor(
        sequence_splits,
        config=config,
        target=TARGET_COLUMN,
        verbose=not quick_check,
    )
    summary = summarize_feedback_qrc_result(result)
    summary["run_name"] = run_name

    prediction_parts = []
    for split_name, predictions in [
        ("train", result.train_predictions),
        ("val", result.val_predictions),
        ("test", result.test_predictions),
    ]:
        _, y_true, dates = sequence_splits[split_name]
        prediction_parts.append(
            prediction_frame(
                split_name=split_name,
                dates=dates,
                y_true=y_true,
                y_pred=predictions,
                run_name=run_name,
            )
        )
    predictions = pd.concat(prediction_parts, ignore_index=True)

    if write_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([summary]).to_csv(
            output_dir / f"{run_name}_metrics.csv",
            index=False,
        )
        predictions.to_csv(
            output_dir / f"{run_name}_predictions.csv",
            index=False,
        )

    return summary, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical Phase 2 feedback-TFIM QRC reference path."
    )
    parser.add_argument(
        "--quick-check",
        action="store_true",
        help="Run a tiny deterministic validation case instead of the full Phase 2 data path.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run without writing result CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for metrics and prediction CSV outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, predictions = run_reference(
        quick_check=args.quick_check,
        output_dir=args.output_dir,
        write_outputs=not args.no_write,
    )
    print(pd.DataFrame([summary]).T)
    print(predictions.head())


if __name__ == "__main__":
    main()
