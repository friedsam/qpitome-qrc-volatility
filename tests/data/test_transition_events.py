import numpy as np
import pandas as pd

from data import transition_events as events


def test_detect_persistent_transition_not_single_spike():
    index = pd.date_range("2020-01-01", periods=120, freq="B")
    values = np.zeros(120)
    values[30] = 2
    values[70:82] = 2
    series = pd.Series(values, index=index)
    onsets = events.detect_onsets(series, 1)
    assert 30 not in onsets
    assert 70 in onsets


def test_matching_features_are_causal():
    index = pd.date_range("2020-01-01", periods=30, freq="B")
    series = pd.Series(np.arange(30, dtype=float), index=index)
    features = events.causal_features(series, 20)
    assert features is not None
    assert features["level"] == 20
    assert features["max20"] == 20


def test_global_episode_chaining():
    catalogue = pd.DataFrame(
        {
            "onset_date": pd.to_datetime(["2020-02-20", "2020-02-25", "2020-04-01"]),
            "index": ["A", "B", "A"],
        }
    )
    clustered = events.cluster_events(catalogue)
    assert clustered.episode_id.nunique() == 2
