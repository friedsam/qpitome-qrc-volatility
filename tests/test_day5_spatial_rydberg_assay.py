"""Focused tests for the standardized spatial Rydberg assay."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "modeling" / "run_day5_spatial_rydberg_assay_shard.py"
SPEC = importlib.util.spec_from_file_location("day5_spatial_assay", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
assay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assay
SPEC.loader.exec_module(assay)


def test_differential_patterns_are_complementary() -> None:
    train = np.vstack([np.zeros(5), np.ones(5), 2.0 * np.ones(5)])
    rows = np.array([[0.0, 1.0, -1.0, 2.0, -2.0]])
    patterns = assay.differential_patterns(train, rows)
    assert patterns.shape == (1, 10)
    np.testing.assert_allclose(patterns[:, 0::2] + patterns[:, 1::2], 1.0)


def test_split_blocks_constructs_connected_correlators() -> None:
    occ = np.linspace(0.1, 1.0, 10)[None, :]
    raw = []
    expected = []
    for i in range(9):
        for j in range(i + 1, 10):
            raw.append(occ[0, i] * occ[0, j] + 0.025)
            expected.append(0.025)
    features = np.column_stack([occ, np.asarray(raw)[None, :]])
    blocks = assay.split_blocks(features)

    assert blocks["occupations"].shape == (1, 10)
    assert blocks["raw_pairs"].shape == (1, 45)
    assert blocks["connected_pairs"].shape == (1, 45)
    assert blocks["occ_plus_connected"].shape == (1, 55)
    np.testing.assert_allclose(blocks["connected_pairs"], np.asarray(expected)[None, :])


def test_feature_diagnostics_detect_constant_columns() -> None:
    features = np.column_stack([np.arange(20.0), np.ones(20), np.zeros(20)])
    diagnostic = assay.feature_diagnostics(features)
    assert diagnostic["n_features"] == 3
    assert diagnostic["near_constant"] == 2
    assert diagnostic["n95"] == 1


def test_pca_block_respects_requested_rank() -> None:
    rng = np.random.default_rng(8)
    train = rng.normal(size=(30, 12))
    test = rng.normal(size=(2, 12))
    train_pca, test_pca = assay.pca_block(train, test, 5)
    assert train_pca.shape == (30, 5)
    assert test_pca.shape == (2, 5)


def test_frozen_configuration_matches_prior_diagnostic() -> None:
    config = assay.rydberg_config()
    assert config.reservoir.chain_atoms == 10
    assert config.reservoir.chain_spacing_um == 7.5
    assert config.evolution_time_us == 0.55
    assert config.global_omega_rad_us == 6.0
    assert config.global_delta_rad_us == 6.0
    assert config.local_delta_rad_us == 4.0


def test_six_shards_partition_positions_without_overlap() -> None:
    positions = list(range(299))
    shards = [[value for position, value in enumerate(positions) if position % 6 == shard] for shard in range(6)]
    merged = [value for shard in shards for value in shard]
    assert sorted(merged) == positions
    assert max(map(len, shards)) - min(map(len, shards)) <= 1
