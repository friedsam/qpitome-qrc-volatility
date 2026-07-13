from __future__ import annotations

import qpitome_qrc.baselines as baselines
import qpitome_qrc.day5 as day5
import qpitome_qrc.evaluation as evaluation
import qpitome_qrc.qrc as qrc


def test_baseline_public_api_exports_reusable_components() -> None:
    expected = {
        "DEFAULT_RESERVOIR_SIZE",
        "DEFAULT_SEED",
        "SPECTRAL_RADIUS",
        "esn_state",
        "fit_esn_predict",
        "fixed_esn_weights",
        "predict_empirical_prior",
    }

    assert set(baselines.__all__) == expected
    for name in expected:
        assert hasattr(baselines, name)


def test_qrc_public_api_exports_feature_maps() -> None:
    expected = {
        "quantum_features",
        "rydberg_features",
    }

    assert set(qrc.__all__) == expected
    for name in expected:
        assert hasattr(qrc, name)


def test_day5_public_api_exports_locked_helpers() -> None:
    expected = {
        "D1",
        "EVAL_START",
        "MIN_TRAIN",
        "PATH_SHAPE_BLOCKS",
        "STATIC",
        "add_extrema",
        "add_path_shape_features",
        "attach_cluster_start",
        "differential_patterns",
        "eligible_rows",
        "feature_diagnostics",
        "load_frame",
        "pca_block",
        "rydberg_config",
        "split_blocks",
    }

    assert set(day5.__all__) == expected
    for name in expected:
        assert hasattr(day5, name)


def test_evaluation_public_api_exports_shared_helpers() -> None:
    expected = {
        "DEFAULT_N_SPLITS",
        "binary_summary",
        "cluster_weighted_summary",
        "d1_basis",
        "fit_feature_map_predict",
        "fit_feature_only_predict",
        "fit_joint_predict",
        "fit_offset_predict",
        "fit_residualizer",
        "fit_train_test_probabilities",
        "fit_transformed_predict",
        "historical_crossfit_d1_logits",
        "logistic_pipeline",
        "proper_score_deltas",
        "residual_diagnostics",
        "residualize_train_test",
        "ridge_pipeline",
        "transformed_logistic_pipeline",
    }

    assert set(evaluation.__all__) == expected
    for name in expected:
        assert hasattr(evaluation, name)
