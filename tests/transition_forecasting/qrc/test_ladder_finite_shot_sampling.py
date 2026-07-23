from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from transition_forecasting.qrc.frozen_ladder_confirmation_tools import (
    symmetric_ladder_mode_matrix,
)
from transition_forecasting.qrc.ladder_finite_shot_sampling import (
    cache_identity,
    evolve_ladder_probe_probabilities,
    load_probability_cache,
    probabilities_to_symmetric_modes,
    sample_probe_probabilities,
    stable_measurement_seed,
    write_probability_cache,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    feature_names_from_metadata,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    build_temporal_rydberg_ladder_features,
)


def _probabilities(rows: int = 4) -> np.ndarray:
    rng = np.random.default_rng(12)
    values = rng.dirichlet(np.ones(64), size=rows * 3)
    return values.reshape(rows, 3, 64)


def test_probability_modes_reproduce_existing_exact_ladder_modes() -> None:
    windows = np.asarray(
        [
            np.column_stack(
                [np.linspace(-0.8, 0.7, 8), np.linspace(0.1, 0.4, 8)]
            ),
            np.column_stack(
                [np.linspace(0.6, -0.5, 8), np.linspace(0.5, 0.2, 8)]
            ),
        ],
        dtype=float,
    )
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        step_duration_us=0.01,
        probe_fractions=(0.25, 0.5, 1.0),
        max_phase_per_substep=1.0,
    )
    geometry = StaggeredLadderGeometryConfig()
    probabilities, metadata = evolve_ladder_probe_probabilities(
        windows,
        reservoir,
        geometry,
        interaction_scale=1.25,
    )
    probability_modes = probabilities_to_symmetric_modes(probabilities)

    features, feature_metadata = build_temporal_rydberg_ladder_features(
        windows,
        reservoir,
        geometry,
        interaction_scale=1.25,
        condition="ordered",
    )
    feature_names = feature_names_from_metadata(feature_metadata)
    existing_modes, _ = symmetric_ladder_mode_matrix(
        features,
        feature_names,
        tuple(int(value) for value in metadata["probe_steps"]),
    )

    assert probabilities.shape == (2, 3, 64)
    assert np.allclose(probabilities.sum(axis=2), 1.0)
    assert np.allclose(probability_modes, existing_modes, atol=1e-12, rtol=1e-12)


def test_sampling_is_invariant_to_row_order() -> None:
    probabilities = _probabilities()
    sample_ids = np.asarray(["a", "b", "c", "d"])
    probe_steps = (10, 20, 40)
    sampled = sample_probe_probabilities(
        probabilities,
        shots=250,
        base_seed=41,
        fold=3,
        sample_ids=sample_ids,
        probe_steps=probe_steps,
    )

    order = np.asarray([2, 0, 3, 1])
    reordered = sample_probe_probabilities(
        probabilities[order],
        shots=250,
        base_seed=41,
        fold=3,
        sample_ids=sample_ids[order],
        probe_steps=probe_steps,
    )
    inverse = np.argsort(order)

    assert np.array_equal(sampled, reordered[inverse])
    assert np.allclose(sampled.sum(axis=2), 1.0)
    assert np.allclose(sampled * 250.0, np.rint(sampled * 250.0))


def test_probe_and_replicate_seed_streams_are_distinct() -> None:
    common = {
        "fold": 2,
        "sample_id": "sample-7",
    }
    first = stable_measurement_seed(100, probe_step=10, **common)
    second_probe = stable_measurement_seed(100, probe_step=20, **common)
    second_replicate = stable_measurement_seed(101, probe_step=10, **common)

    assert len({first, second_probe, second_replicate}) == 3


def test_probability_cache_rejects_identity_mismatch(tmp_path: Path) -> None:
    probabilities = _probabilities(rows=2)
    modes = probabilities_to_symmetric_modes(probabilities)
    sample_ids = np.asarray(["a", "b"])
    reservoir = TemporalRydbergChainConfig(
        n_atoms=6,
        probe_fractions=(0.25, 0.5, 1.0),
    )
    geometry = StaggeredLadderGeometryConfig()
    identity = cache_identity(
        fold=1,
        sample_ids=sample_ids,
        encoded_windows=np.zeros((2, 40, 2), dtype=float),
        reservoir=reservoir,
        geometry=geometry,
        interaction_scale=1.25,
        probe_steps=(10, 20, 40),
        selection_parameters={
            "candidate_features": CandidateFeatureConfig().to_dict(),
        },
    )
    path = tmp_path / "cache.npz"
    write_probability_cache(
        path,
        probabilities=probabilities,
        exact_modes=modes,
        sample_ids=sample_ids,
        fold_splits=np.asarray(["train", "val"]),
        leads=np.asarray([5, 5]),
        labels=np.asarray([0, 1]),
        episode_ids=np.asarray(["e0", "e1"]),
        origin_dates=np.asarray(["2020-01-01", "2020-01-02"]),
        identity=identity,
    )

    loaded = load_probability_cache(path, expected_identity=identity)
    assert np.array_equal(loaded["probabilities"], probabilities)
    assert np.array_equal(loaded["exact_modes"], modes)

    changed = dict(identity)
    changed["interaction_scale"] = 1.0
    with pytest.raises(ValueError, match="identity does not match"):
        load_probability_cache(path, expected_identity=changed)
