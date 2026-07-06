"""Equivalence tests for the one-QRC/two-head Phase 3 extraction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from qpitome_qrc.baselines.reservoir_readouts import (
    best_pr_f1_threshold,
    fit_event_logistic_head,
    fit_volatility_ridge_head,
)
from qpitome_qrc.qrc.rf_qrc_reservoir import RFQRCMap


REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCRIPT = REPO_ROOT / "scripts" / "run_phase3_one_qrc_two_head_readout.py"


def load_legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_two_head", LEGACY_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load legacy script: {LEGACY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ring_feature_map_matches_two_head_legacy() -> None:
    legacy = load_legacy_module()
    rng = np.random.default_rng(87)
    inputs = rng.normal(size=(8, 6))

    old_map = legacy.RFQRCRingMap(6, np.pi / 3, 0.41, 42)
    new_map = RFQRCMap(
        6,
        True,
        "ring",
        np.pi / 3,
        42,
        random_scale=0.41,
    )

    np.testing.assert_allclose(new_map.rz_angles, old_map.rz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.ry_angles, old_map.ry_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(new_map.zz_angles, old_map.zz_angles, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        new_map.transform(inputs),
        old_map.transform(inputs),
        rtol=1e-13,
        atol=1e-13,
    )


def synthetic_readout_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(19)
    features = rng.normal(size=(180, 12))
    latent = 0.5 * features[:, 0] - 0.3 * features[:, 3] + 0.2 * features[:, 7]
    target = np.exp(-2.4 + 0.35 * latent + 0.12 * rng.normal(size=len(features)))
    split = np.array(["train"] * 100 + ["val"] * 40 + ["test"] * 40)
    return features, target, split


def test_volatility_ridge_head_matches_legacy() -> None:
    legacy = load_legacy_module()
    features, target, split = synthetic_readout_data()

    expected = legacy.regression_head(features, target, split, alpha=3000.0)
    actual = fit_volatility_ridge_head(
        features,
        target,
        split,
        alpha=3000.0,
    ).predictions

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_pr_f1_threshold_matches_legacy() -> None:
    legacy = load_legacy_module()
    y_true = np.array([False, True, False, True, True, False, True, False])
    probabilities = np.array([0.12, 0.72, 0.31, 0.61, 0.83, 0.29, 0.56, 0.18])

    expected = legacy.best_threshold_from_val(y_true, probabilities)
    actual = best_pr_f1_threshold(y_true, probabilities)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_event_logistic_head_matches_legacy() -> None:
    legacy = load_legacy_module()
    features, target, split = synthetic_readout_data()
    event_threshold = float(np.quantile(target[split == "train"], 0.90))

    _, expected_prob, expected_threshold = legacy.classifier_head(
        features,
        target,
        split,
        threshold=event_threshold,
        event_name="q90",
        C=1.0,
    )
    result = fit_event_logistic_head(
        features,
        target,
        split,
        event_threshold=event_threshold,
        C=1.0,
        random_state=42,
    )

    np.testing.assert_allclose(
        result.probabilities,
        expected_prob,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        result.decision_threshold,
        expected_threshold,
        rtol=0.0,
        atol=0.0,
    )
