from __future__ import annotations

import numpy as np

from baselines.esn_representation import pool_trajectory, reservoir_trajectory
from baselines.numpy_esn import make_esn_weights
from transition_forecasting.modeling.classical_benchmarks.esn import pooled_features_vectorized


def test_vectorized_pooling_matches_historical_loop() -> None:
    rng = np.random.default_rng(4)
    inputs = rng.normal(size=(7, 12, 3))
    w_in, w = make_esn_weights(3, 20, 0.7, 0.3, 11, connectivity=0.2)
    expected = pool_trajectory(reservoir_trajectory(inputs, w_in, w, 0.4), "final_mean_std", washout=3)
    actual = pooled_features_vectorized(inputs, w_in=w_in, w=w, leak=0.4, washout=3)
    assert np.allclose(actual, expected, atol=1e-11)


def test_svd_ridge_path_matches_sklearn() -> None:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from transition_forecasting.modeling.classical_benchmarks.esn import _ridge_path_predictions

    rng = np.random.default_rng(14)
    features = rng.normal(size=(60, 15))
    target = rng.normal(size=(60, 4))
    train = np.arange(60) < 45
    val = ~train
    alpha = 300.0
    actual = _ridge_path_predictions(features, target, train, val, (alpha,))[alpha]
    scaler = StandardScaler()
    model = Ridge(alpha=alpha)
    model.fit(scaler.fit_transform(features[train]), target[train])
    expected = model.predict(scaler.transform(features[val]))
    assert np.allclose(actual, expected, atol=1e-10)
