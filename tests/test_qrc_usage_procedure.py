import numpy as np
import pandas as pd

from qpitome_qrc.qrc.feedback_tfim_reservoir import (
    FeedbackTFIMQRCConfig,
    build_feedback_qrc_feature_matrix,
    fit_feedback_tfim_qrc_regressor,
    run_feedback_tfim_reservoir_for_window,
    select_temporal_indices,
    summarize_feedback_qrc_result,
)


def _tiny_config() -> FeedbackTFIMQRCConfig:
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


def _tiny_window() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.1],
            [0.2, -0.1],
            [0.1, 0.0],
            [-0.2, 0.3],
        ],
        dtype=float,
    )


def test_select_temporal_indices_even_policy_is_stable():
    indices = select_temporal_indices(lookback_days=4, temporal_steps=2, policy="even")

    np.testing.assert_array_equal(indices, np.array([0, 3]))


def test_single_window_qrc_features_are_finite_and_deterministic():
    config = _tiny_config()
    window = _tiny_window()

    features_1 = run_feedback_tfim_reservoir_for_window(window, config)
    features_2 = run_feedback_tfim_reservoir_for_window(window, config)

    assert features_1.ndim == 1
    assert features_1.shape == features_2.shape
    assert np.isfinite(features_1).all()
    np.testing.assert_allclose(features_1, features_2)


def test_qrc_feature_matrix_has_one_row_per_window():
    config = _tiny_config()
    window = _tiny_window()
    X_windows = np.stack([window, window + 0.05], axis=0)

    single_features = run_feedback_tfim_reservoir_for_window(window, config)
    H = build_feedback_qrc_feature_matrix(X_windows, config)

    assert H.shape == (2, single_features.shape[0])
    assert np.isfinite(H).all()


def test_tiny_qrc_regressor_returns_valid_predictions_and_summary():
    config = _tiny_config()
    base_window = _tiny_window()
    X = np.stack(
        [
            base_window,
            base_window + 0.05,
            base_window + 0.10,
            base_window + 0.15,
            base_window + 0.20,
            base_window + 0.25,
            base_window + 0.30,
            base_window + 0.35,
            base_window + 0.40,
        ],
        axis=0,
    )
    y = np.array([0.10, 0.11, 0.13, 0.15, 0.18, 0.20, 0.23, 0.26, 0.30], dtype=float)
    dates = pd.date_range("2020-01-01", periods=len(y), freq="D")

    sequence_splits = {
        "train": (X[:5], y[:5], dates[:5]),
        "val": (X[5:7], y[5:7], dates[5:7]),
        "test": (X[7:], y[7:], dates[7:]),
    }

    result = fit_feedback_tfim_qrc_regressor(
        sequence_splits,
        config=config,
        target="future_rv_20d",
        verbose=False,
    )
    summary = summarize_feedback_qrc_result(result)

    assert result.train_predictions.shape == y[:5].shape
    assert result.val_predictions.shape == y[5:7].shape
    assert result.test_predictions.shape == y[7:].shape
    assert np.isfinite(result.train_predictions).all()
    assert np.isfinite(result.val_predictions).all()
    assert np.isfinite(result.test_predictions).all()
    assert (result.train_predictions > 0).all()
    assert (result.val_predictions > 0).all()
    assert (result.test_predictions > 0).all()

    expected_summary_keys = {
        "model",
        "target",
        "train_n",
        "val_n",
        "test_n",
        "n_reservoir_features",
        "train_rmse",
        "val_rmse",
        "test_rmse",
        "train_qlike",
        "val_qlike",
        "test_qlike",
        "train_mz_r2",
        "val_mz_r2",
        "test_mz_r2",
    }
    assert expected_summary_keys.issubset(summary.keys())
    assert summary["model"] == "feedback_tfim_qrc_exact"
    assert summary["target"] == "future_rv_20d"
    assert summary["train_n"] == 5
    assert summary["val_n"] == 2
    assert summary["test_n"] == 2
