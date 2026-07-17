from __future__ import annotations

import pandas as pd
import pytest

from transition_forecasting.modeling.global_stage_d_dataset import (
    NEG_PER_POS,
    _episode_split_assignments,
    _validate_manifest,
)


def test_episode_split_is_assigned_once_from_earliest_onset():
    catalogue = pd.DataFrame(
        {
            "episode_id": ["E1", "E1", "E2"],
            "onset_date": pd.to_datetime(["2015-12-31", "2016-01-04", "2016-01-05"]),
        }
    )

    assignments = _episode_split_assignments(catalogue)

    assert assignments == {"E1": "train", "E2": "test"}


def test_manifest_rejects_episode_split_overlap():
    manifest = pd.DataFrame(
        {
            "sample_id": ["P1", "P2"],
            "label": [1, 1],
            "episode_id": ["E1", "E1"],
            "split": ["train", "test"],
            "matched_positive_id": [None, None],
        }
    )

    with pytest.raises(ValueError, match="span train and test"):
        _validate_manifest(manifest)


def test_manifest_requires_exact_control_count_per_positive():
    rows = [
        {
            "sample_id": "P1",
            "label": 1,
            "episode_id": "E1",
            "split": "train",
            "matched_positive_id": None,
        }
    ]
    for control_number in range(NEG_PER_POS - 1):
        rows.append(
            {
                "sample_id": f"N{control_number}",
                "label": 0,
                "episode_id": "E1",
                "split": "train",
                "matched_positive_id": "P1",
            }
        )

    with pytest.raises(ValueError, match="without exactly"):
        _validate_manifest(pd.DataFrame(rows))


def test_manifest_accepts_complete_nonleaking_groups():
    rows = [
        {
            "sample_id": "P1",
            "label": 1,
            "episode_id": "E1",
            "split": "train",
            "matched_positive_id": None,
        }
    ]
    for control_number in range(NEG_PER_POS):
        rows.append(
            {
                "sample_id": f"N{control_number}",
                "label": 0,
                "episode_id": "E1",
                "split": "train",
                "matched_positive_id": "P1",
            }
        )

    _validate_manifest(pd.DataFrame(rows))
