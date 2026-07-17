import numpy as np
import pandas as pd

from baselines.destination_esn import make_reservoir, reservoir_features
from evaluation.paired import compare_predictions
from reservoirs.destination_rydberg import build_operators
from transition_destination.core import encode_rydberg_inputs


def test_esn_feature_shape() -> None:
    paths = np.zeros((3, 4, 2))
    w_in, w, bias = make_reservoir(2, 5, 0.9, 0.35, 7)
    features = reservoir_features(paths, w_in, w, bias, 0.5)
    assert features.shape == (3, 10)


def test_rydberg_operator_shapes() -> None:
    drive, number, n_ops, pair_ops, interaction = build_operators(3)
    assert drive.shape == number.shape == interaction.shape == (8, 8)
    assert len(n_ops) == 3
    assert len(pair_ops) == 2


def test_encoding_shapes() -> None:
    channels = ["return_pct", "long_regime_uncertainty", "rolling_vol_13w"]
    paths = np.zeros((2, 13, 3))
    standardization = pd.DataFrame({"channel": channels, "mean": 0.0, "scale": 1.0})
    encodings, summary = encode_rydberg_inputs(paths, standardization, 2.0)
    assert encodings["return_only"].shape == (2, 13, 1)
    assert encodings["return_uncertainty"].shape == (2, 13, 2)
    assert not summary.empty


def test_paired_comparison_has_no_fixed_episode_count() -> None:
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01", "2020-02-01"]),
        "model": ["base", "candidate", "base", "candidate"],
        "y_true": [0, 0, 1, 1],
        "origin_regime": ["bear", "bear", "bull", "bull"],
        "probability_positive": [0.3, 0.2, 0.7, 0.8],
    })
    paired, summary, thresholds, correction = compare_predictions(frame, "base", "candidate", 100, 3)
    assert len(paired) == 2
    assert set(summary["subgroup"]) == {"all", "positive", "negative"}
    assert not thresholds.empty
    assert not correction.empty
