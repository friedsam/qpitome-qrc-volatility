from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.precontrol_ladder_spacing_assay import (
    PrecontrolSpacingAssayConfig,
    geometry_name,
    geometry_variants,
    median_designated_nearest_coupling,
    normalized_interaction_scale,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def test_geometry_variants_change_only_row_spacing() -> None:
    incumbent = StaggeredLadderGeometryConfig()
    config = PrecontrolSpacingAssayConfig()
    variants = geometry_variants(incumbent, config)

    assert set(variants) == {
        geometry_name(8.0),
        geometry_name(9.0),
        geometry_name(10.0),
    }
    for spacing in config.row_spacings_um:
        geometry = variants[geometry_name(spacing)]
        assert np.isclose(geometry.row_spacing_um, spacing)
        assert np.isclose(
            geometry.longitudinal_spacing_um,
            incumbent.longitudinal_spacing_um,
        )
        assert np.isclose(
            geometry.bottom_spacing_scale,
            incumbent.bottom_spacing_scale,
        )
        assert geometry.defect_site == incumbent.defect_site
        assert np.isclose(geometry.defect_dx_um, incumbent.defect_dx_um)
        assert np.isclose(geometry.defect_dy_um, incumbent.defect_dy_um)


def test_normalized_scales_hold_median_designated_coupling() -> None:
    reservoir = TemporalRydbergChainConfig(n_atoms=6)
    incumbent = StaggeredLadderGeometryConfig(row_spacing_um=9.0)
    target = median_designated_nearest_coupling(
        reservoir,
        incumbent,
        interaction_scale=1.25,
    )

    for spacing in (8.0, 9.0, 10.0):
        challenger = StaggeredLadderGeometryConfig(row_spacing_um=spacing)
        scale = normalized_interaction_scale(
            reservoir,
            incumbent,
            challenger,
            incumbent_scale=1.25,
        )
        actual = median_designated_nearest_coupling(
            reservoir,
            challenger,
            interaction_scale=scale,
        )
        assert np.isclose(actual, target, rtol=1e-12, atol=1e-12)

    incumbent_scale = normalized_interaction_scale(
        reservoir,
        incumbent,
        incumbent,
        incumbent_scale=1.25,
    )
    assert np.isclose(incumbent_scale, 1.25)
