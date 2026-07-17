from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.build_volatility_dataset import (
    build_volatility_dataset,
    build_volatility_frame,
)


def _write_inputs(tmp_path: Path, rows: int = 100) -> tuple[Path, Path]:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    base = np.linspace(100.0, 130.0, rows)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.25,
            "volume": np.arange(rows) + 1_000_000,
            "adjusted_close": base + 0.2,
        }
    )
    volatility = pd.DataFrame(
        {
            "date": dates,
            "open": np.linspace(15.0, 20.0, rows),
            "high": np.linspace(16.0, 21.0, rows),
            "low": np.linspace(14.0, 19.0, rows),
            "close": np.linspace(15.5, 20.5, rows),
            "volume": np.zeros(rows),
        }
    )
    market_path = tmp_path / "historic_market.csv"
    volatility_path = tmp_path / "historic_volatility.csv"
    market.to_csv(market_path, index=False)
    volatility.to_csv(volatility_path, index=False)
    return market_path, volatility_path


def test_build_volatility_frame_has_expected_schema(tmp_path: Path) -> None:
    market_path, volatility_path = _write_inputs(tmp_path)
    frame = build_volatility_frame(market_path, volatility_path)

    assert not frame.empty
    assert len(frame.columns) == 41
    assert frame["date"].is_monotonic_increasing
    assert frame[["future_rv_5d", "future_rv_20d"]].notna().all().all()


def test_build_volatility_dataset_uses_requested_output_name(tmp_path: Path) -> None:
    market_path, volatility_path = _write_inputs(tmp_path)
    result = build_volatility_dataset(
        market_data_path=market_path,
        volatility_data_path=volatility_path,
        output_root=tmp_path / "processed",
        output_name="custom_volatility_dataset",
    )

    assert result.output_path == (
        tmp_path
        / "processed"
        / "custom_volatility_dataset"
        / "custom_volatility_dataset.csv"
    )
    assert result.output_path.exists()
    assert result.manifest_path.exists()


def test_invalid_output_name_is_rejected(tmp_path: Path) -> None:
    market_path, volatility_path = _write_inputs(tmp_path)
    with pytest.raises(ValueError, match="output_name"):
        build_volatility_dataset(
            market_data_path=market_path,
            volatility_data_path=volatility_path,
            output_root=tmp_path / "processed",
            output_name="Bad Name",
        )
