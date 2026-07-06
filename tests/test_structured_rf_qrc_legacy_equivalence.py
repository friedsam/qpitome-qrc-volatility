"""Equivalence tests for the structured level/rate RF-QRC extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from qpitome_qrc.qrc.rf_qrc_structured import (
    StructuredLevelRateRFQRCMap,
    structured_entangler_pairs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = REPO_ROOT / "scripts" / "run_phase3_structured_level_rate_rf_qrc.py"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_structured_rf_qrc", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy script: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def legacy():
    return load_legacy_module()


@pytest.mark.parametrize(
    "mode",
    ["none", "ring", "cross_matched", "cross_all", "block_plus_cross"],
)
def test_structured_entangler_pairs_match_legacy(legacy, mode: str) -> None:
    assert structured_entangler_pairs(mode, 6) == legacy.entangler_pairs(mode, 6)


@pytest.mark.parametrize(
    "mode",
    ["none", "ring", "cross_matched", "cross_all", "block_plus_cross"],
)
def test_structured_feature_map_matches_legacy(legacy, mode: str) -> None:
    rng = np.random.default_rng(1234)
    inputs = rng.normal(size=(6, 6))
    kwargs = dict(
        n_qubits=6,
        entangler=mode,
        input_scale=np.pi / 3,
        level_scale=1.0,
        rate_scale=0.75,
        random_scale=0.35,
        cross_zz_boost=1.0,
        weak_within_boost=0.35,
        seed=42,
    )

    old_map = legacy.StructuredLevelRateRFQRCMap(**kwargs)
    new_map = StructuredLevelRateRFQRCMap(**kwargs)

    np.testing.assert_allclose(new_map.rz_angles, old_map.rz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.ry_angles, old_map.ry_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.zz_angles, old_map.zz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        new_map.transform(inputs),
        old_map.transform(inputs),
        rtol=1e-13,
        atol=1e-13,
    )


def test_structured_model_rejects_odd_qubit_count() -> None:
    with pytest.raises(ValueError):
        StructuredLevelRateRFQRCMap(
            n_qubits=5,
            entangler="ring",
            input_scale=np.pi / 3,
            level_scale=1.0,
            rate_scale=0.75,
            random_scale=0.35,
            cross_zz_boost=1.0,
            weak_within_boost=0.35,
            seed=42,
        )
