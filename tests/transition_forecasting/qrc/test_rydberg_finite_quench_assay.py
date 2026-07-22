from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.rydberg_finite_quench_tools import (
    FiniteQuenchAssayConfig,
    evolve_finite_quench,
    finite_quench_feature_blocks,
    finite_quench_feature_families,
)
from transition_forecasting.qrc.rydberg_susceptibility_tools import (
    classical_context_features,
    fit_warning_classifier,
    prepare_ladder_history,
    reporter_geometry,
    warning_metrics,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def _reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        omega_mod_fraction=0.60,
        step_duration_us=0.03,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )


def test_finite_quench_config_rejects_unsorted_steps() -> None:
    config = FiniteQuenchAssayConfig(response_probe_steps=(5, 3))
    try:
        config.validate()
    except ValueError as exc:
        assert "unique and sorted" in str(exc)
    else:
        raise AssertionError("invalid response probe steps were accepted")


def test_undisplaced_reporter_control_is_primary_geometry() -> None:
    geometry = StaggeredLadderGeometryConfig(
        defect_site=4,
        defect_dx_um=0.35,
        defect_dy_um=-0.40,
    )
    primary = reporter_geometry(geometry, displaced=False)
    assert primary.defect_site == geometry.defect_site
    assert primary.stagger_fraction == geometry.stagger_fraction
    assert primary.bottom_spacing_scale == geometry.bottom_spacing_scale
    assert primary.defect_dx_um == 0.0
    assert primary.defect_dy_um == 0.0


def test_finite_quench_evolution_and_feature_shapes() -> None:
    rng = np.random.default_rng(7)
    windows = rng.uniform(-0.25, 0.25, size=(3, 4, 2))
    reservoir = _reservoir()
    geometry = reporter_geometry(
        StaggeredLadderGeometryConfig(),
        displaced=False,
    )
    prepared = prepare_ladder_history(
        windows,
        reservoir,
        geometry,
        interaction_scale=1.25,
    )
    snapshots = evolve_finite_quench(
        prepared,
        reservoir,
        delta_offset_rad_us=2.0,
        response_probe_steps=(1, 2),
    )
    assert set(snapshots) == {1, 2}
    assert all(value.shape == (3, 64) for value in snapshots.values())
    assert all(
        np.allclose(
            np.linalg.norm(value, axis=1),
            1.0,
            atol=1e-10,
        )
        for value in snapshots.values()
    )

    blocks, names = finite_quench_feature_blocks(
        prepared,
        snapshots,
        geometry,
        response_probe_steps=(1, 2),
        shots=None,
        seed=11,
    )
    expected = {
        "final_static": 3,
        "quench_modes_raw": 6,
        "quench_modes_delta": 6,
        "reporter_raw": 4,
        "reporter_delta": 4,
        "quench_modes": 12,
        "reporter_quench": 8,
        "finite_quench_all": 20,
        "qrc_all": 23,
    }
    for key, width in expected.items():
        assert blocks[key].shape == (3, width)
        assert len(names[key]) == width
        assert np.isfinite(blocks[key]).all()

    sampled, sampled_names = finite_quench_feature_blocks(
        prepared,
        snapshots,
        geometry,
        response_probe_steps=(1, 2),
        shots=1000,
        seed=11,
    )
    assert sampled_names == names
    assert all(sampled[key].shape == blocks[key].shape for key in blocks)
    assert not np.array_equal(
        sampled["reporter_quench"],
        blocks["reporter_quench"],
    )


def test_finite_quench_families_and_warning_classifier() -> None:
    rng = np.random.default_rng(19)
    samples = 16
    level = rng.normal(size=(samples, 40))
    instability = np.abs(rng.normal(size=(samples, 40)))
    har = rng.normal(size=(samples, 10))
    classical, classical_names = classical_context_features(
        level,
        instability,
        har,
    )
    blocks = {
        "final_static": rng.normal(size=(samples, 3)),
        "quench_modes": rng.normal(size=(samples, 12)),
        "reporter_quench": rng.normal(size=(samples, 8)),
        "finite_quench_all": rng.normal(size=(samples, 20)),
        "qrc_all": rng.normal(size=(samples, 23)),
    }
    names = {
        key: tuple(f"{key}_{index}" for index in range(value.shape[1]))
        for key, value in blocks.items()
    }
    matrices, feature_names = finite_quench_feature_families(
        classical,
        classical_names,
        blocks,
        names,
    )
    assert matrices["classical_plus_reporter_quench"].shape[1] == (
        classical.shape[1] + 8
    )
    assert len(feature_names["classical_plus_all"]) == matrices[
        "classical_plus_all"
    ].shape[1]

    labels = np.asarray([0, 1] * 8)
    signal = labels[:, None] + 0.1 * rng.normal(size=(samples, 2))
    train = np.zeros(samples, dtype=bool)
    train[:10] = True
    validation = ~train
    probability = fit_warning_classifier(
        signal,
        labels,
        train,
        validation,
        classifier_c=0.1,
        seed=3,
    )
    assert np.isnan(probability[train]).all()
    assert np.isfinite(probability[validation]).all()
    payload = warning_metrics(labels, probability, validation)
    assert 0.0 <= payload["average_precision"] <= 1.0
    assert 0.0 <= payload["roc_auc"] <= 1.0
