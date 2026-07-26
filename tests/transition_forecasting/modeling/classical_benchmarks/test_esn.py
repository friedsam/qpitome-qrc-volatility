import numpy as np

from baselines.numpy_esn import make_esn_weights
from transition_forecasting.modeling.classical_benchmarks.esn import (
    _ridge_path_predictions,
    pooled_features_vectorized,
)


def test_sparse_esn_is_deterministic_and_ridge_path_shapes():
    w_in, recurrent = make_esn_weights(
        3,
        20,
        0.55,
        0.2,
        1,
        connectivity=0.02,
    )
    w_in_2, recurrent_2 = make_esn_weights(
        3,
        20,
        0.55,
        0.2,
        1,
        connectivity=0.02,
    )
    assert np.array_equal(w_in, w_in_2)
    assert np.array_equal(recurrent, recurrent_2)
    inputs = np.random.default_rng(2).normal(size=(12, 40, 3))
    features = pooled_features_vectorized(
        inputs,
        w_in=w_in,
        w=recurrent,
        leak=0.95,
    )
    target = np.random.default_rng(3).normal(size=(12, 10))
    train_mask = np.arange(12) < 8
    validation_mask = ~train_mask
    prediction = _ridge_path_predictions(
        features,
        target,
        train_mask,
        validation_mask,
        (120000.0,),
    )[120000.0]
    assert prediction.shape == (4, 10)
