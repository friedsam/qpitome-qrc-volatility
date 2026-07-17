from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.build_volatility_dataset import CORE_COLUMNS
from data.validate_volatility_dataset import validate_volatility_dataset


def _valid_frame(rows: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=rows)})
    for index, column in enumerate(CORE_COLUMNS, start=1):
        frame[column] = np.linspace(float(index), float(index + 1), rows)
    return frame


def test_validate_accepts_well_formed_dataset(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    _valid_frame().to_csv(path, index=False)

    result = validate_volatility_dataset(path)

    assert result.rows == 5
    assert result.columns == 1 + len(CORE_COLUMNS)


def test_validate_rejects_duplicate_dates(tmp_path: Path) -> None:
    path = tmp_path / "dataset.csv"
    frame = _valid_frame()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="duplicate dates"):
        validate_volatility_dataset(path)
