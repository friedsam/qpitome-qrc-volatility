from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    load_rolling_fold_dataset,
)


def test_load_rolling_fold_dataset_requires_alignment(
    tmp_path: Path,
) -> None:
    manifest = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label": [0, 1, 0],
            "episode_id": ["e1", "e2", "e3"],
            "origin_date": [
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ],
            "lead": [1, 1, 1],
            "fold": [1, 1, 1],
            "fold_split": ["train", "val", "test"],
        }
    )
    manifest.to_csv(
        tmp_path / "rematched_rolling_manifest.csv",
        index=False,
    )
    values = np.arange(3 * 4, dtype=float).reshape(3, 4, 1)
    np.savez_compressed(
        tmp_path / "rematched_rolling_tensors.npz",
        X=values,
        sample_id=np.asarray(["a", "b", "c"]),
        fold=np.asarray([1, 1, 1]),
        fold_split=np.asarray(["train", "val", "test"]),
        channel_names=np.asarray(["log_volatility_level"]),
    )

    dataset = load_rolling_fold_dataset(tmp_path)

    assert dataset.values.shape == (3, 4, 1)
    assert dataset.channel_names == ("log_volatility_level",)
    assert dataset.manifest["_tensor_row"].tolist() == [0, 1, 2]
    assert dataset.valid.tolist() == [True, True, True]
