from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.day5 import protocol
from qpitome_qrc.day5.features import split_blocks
from qpitome_qrc.evaluation.binary import fit_offset_predict, logistic_pipeline
from qpitome_qrc.evaluation.residualization import d1_basis, residualize_train_test, ridge_pipeline


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_add_extrema_preserves_input_and_matches_locked_geometry() -> None:
    frame = pd.DataFrame(
        {
            "r_d1": [-0.03, 0.01],
            "r_d2": [-0.05, 0.02],
            "r_d3": [-0.02, 0.00],
            "r_d4": [0.01, -0.01],
            "r_d5": [0.02, 0.03],
            "current_return_5d_from_branch": [0.02, 0.03],
            "distance_to_relapse_barrier": [0.10, 0.08],
            "distance_to_recovery_barrier": [0.06, 0.07],
        }
    )

    result = protocol.add_extrema(frame)

    assert "closest_to_relapse" not in frame.columns
    np.testing.assert_allclose(result["closest_to_relapse"], [0.03, 0.04])
    np.testing.assert_allclose(result["closest_to_recovery"], [0.06, 0.07])


def test_differential_patterns_are_complementary_pairs() -> None:
    train = np.array(
        [
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [-1.0, 0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0, 4.0],
        ]
    )
    values = np.array([[0.5, 0.5, 0.5, 0.5, 0.5]])

    patterns = protocol.differential_patterns(train, values)

    assert patterns.shape == (1, 10)
    np.testing.assert_allclose(patterns[:, 0::2] + patterns[:, 1::2], 1.0)
    assert np.all((patterns >= 0.0) & (patterns <= 1.0))


def test_split_blocks_constructs_connected_pairs() -> None:
    occupations = np.linspace(0.1, 1.0, 10)[None, :]
    raw_pairs = np.arange(45, dtype=float)[None, :] / 100.0
    features = np.column_stack([occupations, raw_pairs])

    blocks = split_blocks(features)

    assert blocks["occupations"].shape == (1, 10)
    assert blocks["raw_pairs"].shape == (1, 45)
    assert blocks["connected_pairs"].shape == (1, 45)
    assert blocks["occ_plus_connected"].shape == (1, 55)
    np.testing.assert_allclose(
        blocks["connected_pairs"][0, 0],
        raw_pairs[0, 0] - occupations[0, 0] * occupations[0, 1],
    )


def test_residualize_train_test_shapes_and_finite_values() -> None:
    X_train = np.arange(24, dtype=float).reshape(6, 4)
    H_train = np.column_stack([X_train[:, 0] + 0.5 * X_train[:, 1], X_train[:, 2] - X_train[:, 3]])
    X_test = np.array([[24.0, 25.0, 26.0, 27.0]])
    H_test = np.array([[36.5, -1.0]])

    R_train, R_test = residualize_train_test(X_train, H_train, X_test, H_test, alpha=1.0, n_splits=3)

    assert R_train.shape == H_train.shape
    assert R_test.shape == H_test.shape
    assert np.isfinite(R_train).all()
    assert np.isfinite(R_test).all()


def test_eligible_rows_uses_cluster_start_and_two_class_history() -> None:
    dates = pd.date_range("1989-01-01", periods=33, freq="YS")
    frame = pd.DataFrame(
        {
            "landmark_date": dates,
            "cluster_start": dates,
            "y_recovery": [0, 1] * 16 + [1],
        }
    )

    assert protocol.eligible_rows(frame) == [30, 31, 32]


def test_spatial_runner_reexports_protocol_symbols() -> None:
    module = load_script(
        "day5_spatial_runner",
        "scripts/modeling/day5_branching/spatial_rydberg/run_day5_spatial_rydberg_assay_shard.py",
    )

    assert module.load_frame is protocol.load_frame
    assert module.add_extrema is protocol.add_extrema
    assert module.differential_patterns is protocol.differential_patterns
    assert module.eligible_rows is protocol.eligible_rows
    assert module.rydberg_config is protocol.rydberg_config
    assert module.D1 == protocol.D1
    assert module.STATIC == protocol.STATIC
    assert module.split_blocks is split_blocks
    assert module.fit_offset_predict is fit_offset_predict
    assert module.logistic_pipeline is logistic_pipeline


def test_input_audit_uses_package_helpers_directly() -> None:
    module = load_script(
        "day5_input_audit",
        "scripts/modeling/day5_branching/falsification/run_day5_input_audit.py",
    )

    assert module.load_frame is protocol.load_frame
    assert module.eligible_rows is protocol.eligible_rows
    assert module.D1 == protocol.D1
    assert module.logistic_pipeline is logistic_pipeline
    assert module.ridge_pipeline is ridge_pipeline


def test_residualized_shard_uses_package_helpers_directly() -> None:
    module = load_script(
        "day5_residualized_shard",
        "scripts/modeling/day5_branching/residual_confirmation/run_day5_residualized_rydberg_shard.py",
    )

    assert module.load_frame is protocol.load_frame
    assert module.eligible_rows is protocol.eligible_rows
    assert module.differential_patterns is protocol.differential_patterns
    assert module.rydberg_config is protocol.rydberg_config
    assert module.D1 == protocol.D1
    assert module.STATIC == protocol.STATIC
    assert module.split_blocks is split_blocks
    assert module.fit_offset_predict is fit_offset_predict
    assert module.logistic_pipeline is logistic_pipeline
    assert module.d1_basis is d1_basis
    assert module.residualize_train_test is residualize_train_test


def test_crossfit_shard_uses_package_helpers_directly() -> None:
    module = load_script(
        "day5_residualized_crossfit_shard",
        "scripts/modeling/day5_branching/residual_confirmation/run_day5_residualized_rydberg_crossfit_shard.py",
    )

    assert module.load_frame is protocol.load_frame
    assert module.eligible_rows is protocol.eligible_rows
    assert module.differential_patterns is protocol.differential_patterns
    assert module.rydberg_config is protocol.rydberg_config
    assert module.D1 == protocol.D1
    assert module.STATIC == protocol.STATIC
    assert module.split_blocks is split_blocks
    assert module.fit_offset_predict is fit_offset_predict
    assert module.logistic_pipeline is logistic_pipeline
    assert module.d1_basis is d1_basis
    assert module.residualize_train_test is residualize_train_test
