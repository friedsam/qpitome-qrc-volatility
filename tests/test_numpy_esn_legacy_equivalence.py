import numpy as np

from qpitome_qrc.baselines.numpy_esn import (
    esn_states,
    fit_log_ridge_scores,
    make_esn_weights,
    spectral_scale,
)
from tests.legacy_numpy_esn_reference import (
    legacy_esn_states,
    legacy_fit_ridge_scores,
    legacy_make_esn_weights,
    legacy_spectral_scale,
)


def test_spectral_scaling_matches_frozen_oracle():
    rng = np.random.default_rng(7)
    W = rng.normal(size=(12, 12))

    expected = legacy_spectral_scale(W.copy(), 0.83)
    actual = spectral_scale(W.copy(), 0.83)

    np.testing.assert_array_equal(actual, expected)


def test_weight_generation_matches_frozen_oracle():
    kwargs = dict(
        n_inputs=6,
        n_reservoir=40,
        spectral_radius=0.9,
        input_scale=0.3,
        seed=42,
    )

    expected_in, expected_W = legacy_make_esn_weights(**kwargs)
    actual_in, actual_W = make_esn_weights(**kwargs)

    np.testing.assert_array_equal(actual_in, expected_in)
    np.testing.assert_array_equal(actual_W, expected_W)


def test_state_features_match_frozen_oracle():
    rng = np.random.default_rng(123)

    X = rng.normal(size=(11, 9, 6))

    W_in, W = legacy_make_esn_weights(
        n_inputs=6,
        n_reservoir=30,
        spectral_radius=0.7,
        input_scale=0.2,
        seed=42,
    )

    expected = legacy_esn_states(X, W_in, W, leak=0.5)
    actual = esn_states(X, W_in, W, leak=0.5)

    np.testing.assert_array_equal(actual, expected)


def test_log_ridge_scores_match_frozen_oracle():
    rng = np.random.default_rng(321)

    features = {
        "train": rng.normal(size=(60, 25)),
        "val": rng.normal(size=(20, 25)),
        "test": rng.normal(size=(25, 25)),
    }

    targets = {
        "train": np.exp(rng.normal(-2.0, 0.4, size=60)),
        "val": np.exp(rng.normal(-2.0, 0.4, size=20)),
        "test": np.exp(rng.normal(-2.0, 0.4, size=25)),
    }

    expected = legacy_fit_ridge_scores(
        features,
        targets,
        alpha=1000.0,
    )

    actual = fit_log_ridge_scores(
        features,
        targets,
        alpha=1000.0,
    )

    for split in ("train", "val", "test"):
        np.testing.assert_array_equal(actual[split], expected[split])


def test_full_reservoir_to_readout_path_matches_frozen_oracle():
    rng = np.random.default_rng(987)

    X = {
        "train": rng.normal(size=(50, 12, 6)),
        "val": rng.normal(size=(18, 12, 6)),
        "test": rng.normal(size=(22, 12, 6)),
    }

    y = {
        "train": np.exp(rng.normal(-2.0, 0.5, size=50)),
        "val": np.exp(rng.normal(-2.0, 0.5, size=18)),
        "test": np.exp(rng.normal(-2.0, 0.5, size=22)),
    }

    old_in, old_W = legacy_make_esn_weights(
        6,
        35,
        0.9,
        0.3,
        42,
    )

    new_in, new_W = make_esn_weights(
        6,
        35,
        0.9,
        0.3,
        42,
    )

    old_H = {
        split: legacy_esn_states(X[split], old_in, old_W, 0.3)
        for split in X
    }

    new_H = {
        split: esn_states(X[split], new_in, new_W, 0.3)
        for split in X
    }

    old_scores = legacy_fit_ridge_scores(old_H, y, 300.0)
    new_scores = fit_log_ridge_scores(new_H, y, 300.0)

    for split in ("train", "val", "test"):
        np.testing.assert_array_equal(new_H[split], old_H[split])
        np.testing.assert_array_equal(new_scores[split], old_scores[split])


def test_historical_grid_matches_frozen_phase3_configs():
    from qpitome_qrc.baselines.numpy_esn import historical_numpy_esn_grid

    actual = historical_numpy_esn_grid([42])

    expected = [
        {
            "n": 300,
            "sr": 0.70,
            "inp": 0.30,
            "leak": 0.30,
            "alpha": 300.0,
            "seed": 42,
            "config_id": "esn_n300_sr0.7_inp0.3_leak0.3_alpha300.0_seed42",
        },
        {
            "n": 300,
            "sr": 0.90,
            "inp": 0.30,
            "leak": 0.30,
            "alpha": 1000.0,
            "seed": 42,
            "config_id": "esn_n300_sr0.9_inp0.3_leak0.3_alpha1000.0_seed42",
        },
        {
            "n": 500,
            "sr": 0.70,
            "inp": 0.20,
            "leak": 0.50,
            "alpha": 1000.0,
            "seed": 42,
            "config_id": "esn_n500_sr0.7_inp0.2_leak0.5_alpha1000.0_seed42",
        },
        {
            "n": 500,
            "sr": 0.90,
            "inp": 0.20,
            "leak": 0.50,
            "alpha": 3000.0,
            "seed": 42,
            "config_id": "esn_n500_sr0.9_inp0.2_leak0.5_alpha3000.0_seed42",
        },
    ]

    assert actual == expected
