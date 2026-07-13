from __future__ import annotations

import importlib.util
from pathlib import Path

from qpitome_qrc.baselines.fixed_esn import fit_esn_predict
from qpitome_qrc.day5 import protocol
from qpitome_qrc.evaluation.binary import (
    fit_feature_map_predict,
    fit_feature_only_predict,
    logistic_pipeline,
)
from qpitome_qrc.evaluation.scoring import binary_summary, cluster_weighted_summary


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_rydberg_baseline_uses_shared_day5_protocol() -> None:
    module = load_script(
        "cross_market_day5_fixed_rydberg_feature",
        "scripts/modeling/run_cross_market_day5_fixed_rydberg_feature.py",
    )

    assert module.D1 is protocol.D1
    assert module.MIN_TRAIN == protocol.MIN_TRAIN
    assert module.EVAL_START == protocol.EVAL_START
    assert module.logistic_pipeline is logistic_pipeline
    assert module.fit_feature_only_predict is fit_feature_only_predict


def test_fixed_quantum_baseline_uses_shared_components() -> None:
    module = load_script(
        "cross_market_day5_fixed_quantum_feature",
        "scripts/modeling/run_cross_market_day5_fixed_quantum_feature.py",
    )

    assert module.D1 is protocol.D1
    assert module.MIN_TRAIN == protocol.MIN_TRAIN
    assert module.EVAL_START == protocol.EVAL_START
    assert module.fit_feature_only_predict is fit_feature_only_predict
    assert module.fit_feature_map_predict is fit_feature_map_predict
    assert module.score is binary_summary
    assert module.binary_summary is binary_summary
    assert module.cluster_weighted_summary is cluster_weighted_summary


def test_fixed_esn_baseline_uses_shared_components() -> None:
    module = load_script(
        "cross_market_day5_fixed_esn",
        "scripts/modeling/run_cross_market_day5_fixed_esn.py",
    )

    assert module.D1 is protocol.D1
    assert module.MIN_TRAIN == protocol.MIN_TRAIN
    assert module.EVAL_START == protocol.EVAL_START
    assert module.fit_feature_only_predict is fit_feature_only_predict
    assert module.fit_esn_predict is fit_esn_predict
    assert module.score is binary_summary
    assert module.binary_summary is binary_summary
    assert module.cluster_weighted_summary is cluster_weighted_summary
