from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.ladder_spacing_assay import (
    LadderSpacingAssayConfig,
    geometry_variants,
    probabilities_to_mode_family,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


def test_geometry_variants_are_predeclared_and_distinct() -> None:
    incumbent = StaggeredLadderGeometryConfig()
    variants = geometry_variants(incumbent, LadderSpacingAssayConfig())
    assert tuple(variants) == (
        "incumbent",
        "no_displaced_defect",
        "stronger_bottom_spacing",
        "stronger_cross_row_mixing",
        "weaker_cross_row_mixing",
    )
    assert variants["no_displaced_defect"].defect_dx_um == 0.0
    assert variants["stronger_cross_row_mixing"].row_spacing_um < incumbent.row_spacing_um
    assert variants["weaker_cross_row_mixing"].row_spacing_um > incumbent.row_spacing_um


def test_probability_mode_families_have_expected_width() -> None:
    probabilities = np.zeros((2, 3, 64), dtype=float)
    probabilities[:, :, 0] = 1.0
    symmetric = probabilities_to_mode_family(probabilities, "symmetric_modes")
    compact = probabilities_to_mode_family(probabilities, "compact_modes")
    assert symmetric.shape == (2, 9)
    assert compact.shape == (2, 12)
    assert np.isfinite(symmetric).all()
    assert np.isfinite(compact).all()
