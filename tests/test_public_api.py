from __future__ import annotations

import qpitome_qrc.day5 as day5
import qpitome_qrc.evaluation as evaluation


def test_day5_public_api_exports_locked_helpers() -> None:
    expected = {
        "D1",
        "EVAL_START",
        "MIN_TRAIN",
        "STATIC",
        "add_extrema",
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
        "d1_basis",
        "fit_feature_only_predict",
        "fit_joint_predict",
        "fit_offset_predict",
        "fit_residualizer",
        "historical_crossfit_d1_logits",
        "logistic_pipeline",
        "residual_diagnostics",
        "residualize_train_test",
    }

    assert set(evaluation.__all__) == expected
    for name in expected:
        assert hasattr(evaluation, name)
