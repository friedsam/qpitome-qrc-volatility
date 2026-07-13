from __future__ import annotations

import importlib.util
from pathlib import Path

from qpitome_qrc.day5 import protocol
from qpitome_qrc.evaluation.binary import logistic_pipeline


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
