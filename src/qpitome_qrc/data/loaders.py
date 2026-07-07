from pathlib import Path

import pandas as pd


DEFAULT_PHASE2_PROCESSED_PATH = Path("data/processed/phase2_spy_vix_volatility.csv")


def load_phase2_volatility_data(
    path: str | Path = DEFAULT_PHASE2_PROCESSED_PATH,
) -> pd.DataFrame:
    """Load the Phase 2 SPY/VIX volatility forecasting dataset.

    Expected file is produced by:
        python scripts/data/prepare_phase2_spy_vix_dataset.py
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed Phase 2 dataset not found: {path}. "
            "Run `python scripts/data/prepare_phase2_spy_vix_dataset.py` first."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = {
        "date",
        "future_rv_5d",
        "future_rv_20d",
        "rv_20d",
        "vix_close",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase 2 dataset missing required columns: {sorted(missing)}")

    return df


# Backward-compatible alias while old notebooks/scripts are being retired.
def load_market_stress_data(path: str | Path = DEFAULT_PHASE2_PROCESSED_PATH) -> pd.DataFrame:
    return load_phase2_volatility_data(path)
