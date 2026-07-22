from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.rydberg_susceptibility_tools import (
    SusceptibilityAssayConfig,
    classical_context_features,
    feature_families,
    fit_warning_classifier,
    prepare_ladder_history,
    probe_prepared_ladder,
    reporter_geometry,
    susceptibility_feature_blocks,
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


def test_susceptibility_config_rejects_invalid_probe_steps() -> None:
    config = SusceptibilityAssayConfig(response_probe_steps=(3, 1))
    try:
        config.validate()
    except ValueError as exc:
        assert "unique and sorted" in str(exc)
    else:
        raise AssertionError("invalid response probe steps were accepted")


def test_reporter_geometry_removes_only_reporter_displacement() -> None:
    geometry = StaggeredLadderGeometryConfig(
        defect_site=4,
        defect_dx_um=0.35,
        defect_dy_um=-0.40,
    )
    control = reporter_geometry(geometry, displaced=False)
    assert control.defect_site == geometry.defect_site
    assert control.stagger_fraction == geometry.stagger_fraction
    assert control.bottom_spacing_scale == geometry.bottom_spacing_scale
    assert control.defect_dx_um == 0.0
    assert control.defect_dy_um == 0.0
    assert reporter_geometry(geometry, displaced=True) == geometry


def test_preparation_probe_and_feature_shapes() -> None:
    rng = np.random.default_rng(7)
    windows = rng.uniform(-0.25, 0.25, size=(3, 4, 2))
    reservoir = _reservoir()
    geometry = StaggeredLadderGeometryConfig()
    prepared = prepare_ladder_history(
        windows,
        reservoir,
        geometry,
        interaction_scale=1.25,
    )
    assert prepared.final_states.shape == (3, 64)
    assert prepared.history_probe_steps == (2, 4)
    assert np.allclose(
        np.linalg.norm(prepared.final_states, axis=1),
        1.0,
        atol=1e-10,
    )

    branches = probe_prepared_ladder(
        prepared,
        reservoir,
        delta_offset_rad_us=0.8,
        response_probe_steps=(1, 2),
    )
    assert set(branches) == {"minus", "zero", "plus"}
    assert all(set(value) == {1, 2} for value in branches.values())

    blocks, names = susceptibility_feature_blocks(
        prepared,
        branches,
        geometry,
        delta_offset_rad_us=0.8,
        response_probe_steps=(1, 2),
        shots=None,
        seed=11,
    )
    assert blocks["static_modes"].shape == (3, 6)
    assert blocks["chi_modes"].shape == (3, 6)
    assert blocks["kappa_modes"].shape == (3, 6)
    assert blocks["reporter_response"].shape == (3, 8)
    assert blocks["susceptibility_all"].shape == (3, 20)
    assert blocks["qrc_all"].shape == (3, 26)
    assert all(blocks[key].shape[1] == len(names[key]) for key in blocks)
    assert all(np.isfinite(value).all() for value in blocks.values())

    sampled, sampled_names = susceptibility_feature_blocks(
        prepared,
        branches,
        geometry,
        delta_offset_rad_us=0.8,
        response_probe_steps=(1, 2),
        shots=1000,
        seed=11,
    )
    assert sampled_names == names
    assert all(sampled[key].shape == blocks[key].shape for key in blocks)
    assert not np.array_equal(
        sampled["susceptibility_all"],
        blocks["susceptibility_all"],
    )


def test_feature_families_and_warning_classifier() -> None:
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
    qrc_blocks = {
        "static_modes": rng.normal(size=(samples, 9)),
        "chi_modes": rng.normal(size=(samples, 9)),
        "kappa_modes": rng.normal(size=(samples, 9)),
        "reporter_response": rng.normal(size=(samples, 12)),
        "susceptibility_all": rng.normal(size=(samples, 30)),
        "qrc_all": rng.normal(size=(samples, 39)),
    }
    qrc_names = {
        key: tuple(f"{key}_{index}" for index in range(value.shape[1]))
        for key, value in qrc_blocks.items()
    }
    matrices, names = feature_families(
        classical,
        classical_names,
        qrc_blocks,
        qrc_names,
    )
    assert matrices["classical_plus_all"].shape[1] == (
        classical.shape[1] + 39
    )
    assert len(names["classical_plus_all"]) == matrices[
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
