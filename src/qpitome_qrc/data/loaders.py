from pathlib import Path

import pandas as pd


DEFAULT_PROCESSED_PATH = Path("data/processed/market_stress_v0.csv")


def load_market_stress_data(path: str | Path = DEFAULT_PROCESSED_PATH) -> pd.DataFrame:
    """Load processed market-stress dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            "Run `python scripts/prepare_dataset.py` first."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a `date` column.")

    return df