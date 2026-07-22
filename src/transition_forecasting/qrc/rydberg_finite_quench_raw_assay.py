from __future__ import annotations

from pathlib import Path

from transition_forecasting.qrc import rydberg_finite_quench_assay as _base
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
)
from transition_forecasting.qrc.rydberg_finite_quench_tools import (
    FiniteQuenchAssayConfig,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
)


PRIMARY_GEOMETRY = "reporter_undisplaced"
PRIMARY_FAMILY = "classical_plus_reporter_raw"


def run_rydberg_finite_quench_raw_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: FiniteQuenchAssayConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    ladder_geometry: StaggeredLadderGeometryConfig,
    run_id: str | None = None,
) -> Path:
    """Run the finite-quench assay with raw reporter response frozen as primary."""

    if _base.PRIMARY_GEOMETRY != PRIMARY_GEOMETRY:
        raise RuntimeError("unexpected finite-quench primary geometry")
    _base.PRIMARY_FAMILY = PRIMARY_FAMILY
    return _base.run_rydberg_finite_quench_assay(
        fold_dir=fold_dir,
        results_root=results_root,
        assay=assay,
        candidate_features=candidate_features,
        reservoir=reservoir,
        ladder_geometry=ladder_geometry,
        run_id=run_id,
    )
