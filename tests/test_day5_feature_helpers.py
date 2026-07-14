from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from qpitome_qrc.day5.features import feature_diagnostics, pca_block
from qpitome_qrc.evaluation.binary import fit_feature_only_predict, fit_joint_predict


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_feature_diagnostics_constant_block() -> None:
    diagnostics = feature_diagnostics(np.ones((6, 4)))

    assert diagnostics == {
        "n_features": 4,
        "participation_ratio": 0.0,
        "n95": 0,
        "condition_number": np.inf,
        "near_constant": 4,
    }


def test_pca_block_preserves_train_test_row_counts() -> None:
    train = np.arange(40, dtype=float).reshape(8, 5)
    test = np.arange(10, dtype=float).reshape(2, 5)

    train_pca, test_pca = pca_block(train, test, rank=3)

    assert train_pca.shape == (8, 3)
    assert test_pca.shape == (2, 3)
    assert np.isfinite(train_pca).all()
    assert np.isfinite(test_pca).all()


def test_spatial_runner_reexports_extracted_helpers() -> None:
    module = load_script(
        "day5_spatial_runner_helpers",
        "scripts/modeling/day5_branching/spatial_rydberg/run_day5_spatial_rydberg_assay_shard.py",
    )

    assert module.feature_diagnostics is feature_diagnostics
    assert module.pca_block is pca_block
    assert module.fit_joint_predict is fit_joint_predict
    assert module.fit_rydberg_only_predict is fit_feature_only_predict
