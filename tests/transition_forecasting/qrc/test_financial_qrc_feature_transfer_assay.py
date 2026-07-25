from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from transition_forecasting.qrc.bivariate_carrier_crossmix_assay import (
    CARRIER_MASKS,
    evolve_carrier_mask_probabilities,
)
from transition_forecasting.qrc.bivariate_crossover_assay import (
    build_crossover_feature_banks,
)
from transition_forecasting.qrc.financial_qrc_feature_transfer_assay import (
    FinancialQRCFeatureTransferConfig,
    _directional_payload,
    _fit_signed_residual_probe,
)
from transition_forecasting.qrc.frozen_ladder_confirmation_tools import (
    symmetric_ladder_mode_matrix,
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


def _reservoir() -> TemporalRydbergChainConfig:
    return TemporalRydbergChainConfig(
        n_atoms=6,
        delta_center_rad_us=6.0,
        delta_span_rad_us=4.0,
        omega_base_rad_us=6.0,
        omega_mod_fraction=0.60,
        step_duration_us=0.01,
        probe_fractions=(0.5, 1.0),
        shots=None,
    )


def test_identity_carrier_reproduces_incumbent_symmetric_modes() -> None:
    rng = np.random.default_rng(20260724)
    windows = rng.uniform(-0.3, 0.3, size=(4, 8, 2))
    geometry = StaggeredLadderGeometryConfig()
    reservoir = _reservoir()

    incumbent_features, incumbent_metadata = build_temporal_rydberg_ladder_features(
        windows,
        reservoir,
        geometry,
        interaction_scale=0.25,
        condition="ordered",
    )
    names = feature_names_from_metadata(incumbent_metadata)
    probes = tuple(int(value) for value in incumbent_metadata["probe_steps"])
    incumbent_modes, _ = symmetric_ladder_mode_matrix(
        incumbent_features,
        names,
        probes,
    )

    probabilities, _ = evolve_carrier_mask_probabilities(
        windows,
        reservoir,
        geometry,
        np.asarray(CARRIER_MASKS["identity"], dtype=float),
        mask_name="identity_test",
        interaction_scale=0.25,
        drive_phase_rad=0.0,
    )
    carrier_modes = build_crossover_feature_banks(probabilities)[
        "six_mode_density_curvature"
    ]
    np.testing.assert_allclose(carrier_modes, incumbent_modes, atol=1e-10, rtol=1e-10)


def test_signed_probe_recovers_known_direction_without_intercept() -> None:
    rng = np.random.default_rng(91)
    rows = 48
    x = rng.normal(size=(rows, 3))
    residual_scalar = 0.7 * x[:, 0] - 0.4 * x[:, 1]
    residuals = np.column_stack([residual_scalar, 0.5 * residual_scalar])
    har = np.zeros_like(residuals)
    y = har + residuals
    valid = np.zeros(rows, dtype=bool)
    valid[:36] = True
    dates = pd.date_range("2020-01-01", periods=rows, freq="D").astype(str).to_numpy()
    config = FinancialQRCFeatureTransferConfig(
        ridge_alphas=(0.001, 0.01, 0.1),
        inner_holdout_fraction=0.25,
    )

    correction, diagnostics, _ = _fit_signed_residual_probe(
        x,
        y=y,
        har=har,
        residuals=residuals,
        residual_train_mask=valid,
        origin_date=dates,
        config=config,
        pc1_only=False,
    )

    assert diagnostics["intercept_max_abs"] == 0.0
    assert np.corrcoef(correction[36:].reshape(-1), residuals[36:].reshape(-1))[0, 1] > 0.95


def test_directional_payload_rejects_universal_positive_uplift() -> None:
    cells = pd.DataFrame(
        {
            "har_residual": [-1.0, -0.5, 0.5, 1.0],
            "qrc_correction": [0.2, 0.2, 0.2, 0.2],
            "y_true": [-1.0, -0.5, 0.5, 1.0],
            "har_prediction": [0.0, 0.0, 0.0, 0.0],
            "augmented_prediction": [0.2, 0.2, 0.2, 0.2],
        }
    )

    payload = _directional_payload(cells)

    assert payload["wrong_up_rate"] == 1.0
    assert payload["residual_sign_gap"] == pytest.approx(0.0)
    assert payload["balanced_sign_accuracy"] == pytest.approx(0.5)


def test_transfer_config_rejects_development_folds_used_for_selection() -> None:
    with pytest.raises(ValueError, match="folds 4-8"):
        FinancialQRCFeatureTransferConfig(folds=(3, 4, 5)).validate()
