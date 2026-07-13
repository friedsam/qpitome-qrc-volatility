from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from qpitome_qrc.day5.protocol import attach_cluster_start


def load_script(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_attach_cluster_start_deduplicates_keys_and_uses_earliest_branch() -> None:
    frame = pd.DataFrame({
        "market_key": ["a", "a", "b"],
        "episode_id": [1, 2, 3],
        "branch_date": pd.to_datetime(["2020-03-10", "2020-03-05", "2021-01-02"]),
    })
    clusters = pd.DataFrame({
        "market_key": ["a", "a", "a", "b"],
        "episode_id": [1, 1, 2, 3],
        "cluster_id": ["c1", "c1", "c1", "c2"],
    })

    result = attach_cluster_start(frame, clusters, require_complete=True)

    assert len(result) == len(frame)
    assert result["cluster_id"].tolist() == ["c1", "c1", "c2"]
    assert result["cluster_start"].tolist() == list(
        pd.to_datetime(["2020-03-05", "2020-03-05", "2021-01-02"])
    )


def test_attach_cluster_start_can_reject_incomplete_mapping() -> None:
    frame = pd.DataFrame({
        "market_key": ["a", "b"],
        "episode_id": [1, 2],
        "branch_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
    })
    clusters = pd.DataFrame({
        "market_key": ["a"],
        "episode_id": [1],
        "cluster_id": ["c1"],
    })

    with pytest.raises(ValueError, match="missing cluster ids"):
        attach_cluster_start(frame, clusters, require_complete=True)


def test_fixed_quantum_script_reexports_cluster_attachment() -> None:
    module = load_script(
        "fixed_quantum_cluster_attachment",
        "scripts/modeling/run_cross_market_day5_fixed_quantum_feature.py",
    )

    assert module.attach_cluster_start is attach_cluster_start
