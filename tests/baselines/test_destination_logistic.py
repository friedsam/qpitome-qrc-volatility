import numpy as np

from baselines.destination_logistic import (
    fit_predict_probability,
    fit_ridge_logistic,
    sigmoid,
    standardize_train_test,
)


def test_sigmoid_is_bounded_and_monotone() -> None:
    values = sigmoid(np.array([-100.0, 0.0, 100.0]))
    assert np.all(values > 0.0)
    assert np.all(values < 1.0)
    assert np.all(np.diff(values) > 0.0)


def test_ridge_logistic_separates_simple_signal() -> None:
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    beta = fit_ridge_logistic(x, y, penalty=1.0)
    probabilities = sigmoid(np.column_stack([np.ones(len(x)), x]) @ beta)
    assert probabilities[:2].max() < probabilities[2:].min()


def test_standardization_uses_training_statistics_only() -> None:
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    test = np.array([4.0, 6.0])
    train_z, test_z, mean, scale = standardize_train_test(train, test)
    assert np.allclose(mean, [1.0, 3.0])
    assert np.allclose(scale, [1.0, 1.0])
    assert np.allclose(train_z.mean(axis=0), 0.0)
    assert np.allclose(test_z, [3.0, 3.0])


def test_offset_changes_prediction() -> None:
    x = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    direct = fit_predict_probability(x, y, np.array([0.5]), penalty=1.0)
    shifted = fit_predict_probability(
        x,
        y,
        np.array([0.5]),
        penalty=1.0,
        train_offset=np.zeros(len(y)),
        test_offset=1.0,
    )
    assert shifted > direct
