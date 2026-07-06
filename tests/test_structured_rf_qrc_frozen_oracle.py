from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qpitome_qrc.qrc.rf_qrc_structured import StructuredLevelRateRFQRCMap

REFERENCE = Path(__file__).with_name("legacy_structured_rf_qrc_reference.py")


def load_reference():
    spec = importlib.util.spec_from_file_location("structured_reference", REFERENCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {REFERENCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "mode",
    ["none", "ring", "cross_matched", "cross_all", "block_plus_cross"],
)
def test_frozen_structured_oracle(mode: str) -> None:
    reference = load_reference()
    rng = np.random.default_rng(1234)
    inputs = rng.normal(size=(6, 6))
    kwargs = {
        "n_qubits": 6,
        "entangler": mode,
        "input_scale": np.pi / 3,
        "level_scale": 1.0,
        "rate_scale": 0.75,
        "random_scale": 0.35,
        "cross_zz_boost": 1.0,
        "weak_within_boost": 0.35,
        "seed": 42,
    }
    expected = reference.StructuredLevelRateRFQRCMap(**kwargs).transform(inputs)
    actual = StructuredLevelRateRFQRCMap(**kwargs).transform(inputs)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
