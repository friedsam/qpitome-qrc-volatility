from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.transition_forecasting.qrc.run_financial_qrc_input_optimization_assay import (
    DEFAULT_FOLD_DIR,
    DEFAULT_PANEL,
    validate_inputs,
)
from transition_forecasting.qrc.financial_qrc_input_optimization_assay import (
    FinancialQRCInputOptimizationConfig,
    compact_mode_banks,
    fit_fixed_signed_probe,
    signed_market_channels,
)


def test_signed_market_channels_use_bidirectional_stress_orientation() -> None:
    dates = pd.date_range("2020-01-01", periods=4, tz="UTC")
    close = np.asarray([100.0, 98.0, 99.0, 96.0])
    market = pd.DataFrame(
        {
            "open": [100.0, 99.0, 98.5, 98.0],
            "high": [101.0, 100.0, 100.0, 99.0],
            "low": [99.0, 97.0, 98.0, 95.0],
            "close": close,
        },
        index=dates,
    )
    result = signed_market_channels(market, dates[-1], window=3)
    assert result is not None
    stress_return, signed_range = result
    assert stress_return[0] > 0.0
    assert stress_return[1] < 0.0
    assert stress_return[2] > 0.0
    assert signed_range[0] > 0.0
    assert signed_range[1] < 0.0
    assert signed_range[2] > 0.0


def test_compact_mode_banks_have_expected_widths() -> None:
    probabilities = np.zeros((3, 2, 64), dtype=float)
    probabilities[:, :, 0] = 1.0
    banks = compact_mode_banks(probabilities)
    assert banks["six_density_curvature"].shape == (3, 4)
    assert banks["nine_density_curvature_antisymmetric_curvature"].shape == (3, 6)
    assert np.allclose(banks["six_density_curvature"], 0.0)
    assert np.allclose(
        banks["nine_density_curvature_antisymmetric_curvature"], 0.0
    )


def test_fixed_probe_recovers_signed_signal_without_intercept() -> None:
    rng = np.random.default_rng(37)
    features = rng.normal(size=(48, 4))
    signal = 0.8 * features[:, 0] - 0.3 * features[:, 2]
    residuals = np.column_stack([signal, 0.5 * signal])
    fit = np.zeros(48, dtype=bool)
    fit[:36] = True

    correction, diagnostics = fit_fixed_signed_probe(
        features,
        residuals,
        fit,
        ridge_alpha=0.01,
    )

    assert diagnostics["intercept_max_abs"] == 0.0
    assert np.corrcoef(correction[36:].reshape(-1), residuals[36:].reshape(-1))[0, 1] > 0.98


def test_input_optimization_config_is_bounded() -> None:
    with pytest.raises(ValueError, match="restricted to L5"):
        FinancialQRCInputOptimizationConfig(lead=1).validate()
    with pytest.raises(ValueError, match="windows"):
        FinancialQRCInputOptimizationConfig(windows=(4, 40)).validate()
    FinancialQRCInputOptimizationConfig().validate()


def test_runner_defaults_and_input_validation(tmp_path: Path) -> None:
    assert DEFAULT_FOLD_DIR == Path(
        "data/processed/global_transition_dataset_1d/purged_walk_forward_folds"
    )
    assert DEFAULT_PANEL == Path(
        "data/fallback/transition_forecasting/"
        "global_stock_indices_historical_data/all_indices_data.csv"
    )
    fold_dir = tmp_path / "folds"
    fold_dir.mkdir()
    panel = tmp_path / "panel.csv"
    panel.write_text("date,open,high,low,close,ticker\n", encoding="utf-8")
    (fold_dir / "rematched_rolling_manifest.csv").write_text(
        "sample_id\n", encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="rematched_rolling_tensors"):
        validate_inputs(fold_dir, panel)
    (fold_dir / "rematched_rolling_tensors.npz").write_bytes(b"placeholder")
    assert validate_inputs(fold_dir, panel) == (fold_dir, panel)
