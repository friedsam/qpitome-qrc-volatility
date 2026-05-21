from pathlib import Path

import pandas as pd


RAW_FILES = [
    Path("data/raw/kaggle_market_ohlcv/SPY.csv"),
    Path("data/raw/vix/VIX_History.csv"),
    Path("data/raw/kaggle_market_ohlcv/QQQ_split_adj.csv"),
    Path("data/raw/kaggle_market_ohlcv/SP500.csv"),
]


def inspect_csv(path: Path) -> None:
    print("\n" + "=" * 100)
    print(f"FILE: {path}")
    print(f"exists: {path.exists()}")

    if not path.exists():
        return

    print(f"size_bytes: {path.stat().st_size}")

    df = pd.read_csv(path)
    print(f"shape: {df.shape}")
    print(f"columns: {list(df.columns)}")

    date_col = "date" if "date" in df.columns else "DATE" if "DATE" in df.columns else None
    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        print(f"date_col: {date_col}")
        print(f"date_min: {dates.min()}")
        print(f"date_max: {dates.max()}")
        print(f"bad_dates: {dates.isna().sum()}")
        print(f"duplicate_dates: {dates.duplicated().sum()}")

    print("missing_by_column:")
    print(df.isna().sum())

    print("\nhead:")
    print(df.head(3))

    print("\ntail:")
    print(df.tail(3))


if __name__ == "__main__":
    for file_path in RAW_FILES:
        inspect_csv(file_path)
