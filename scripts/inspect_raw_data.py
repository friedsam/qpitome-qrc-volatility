from pathlib import Path

import pandas as pd


RAW_FILES = [
    Path("data/raw/kaggle_market_ohlcv/SPY.csv"),
    Path("data/raw/vix/VIX_History.csv"),
    Path("data/raw/kaggle_market_ohlcv/SP500.csv"),
    Path("data/raw/kaggle_market_ohlcv/QQQ_raw.csv"),
    Path("data/raw/kaggle_market_ohlcv/QQQ_split_adj.csv"),
    Path("data/raw/kaggle_market_ohlcv/NASDAQ_100.csv"),
]


def inspect_file(path: Path) -> None:
    print("\n" + "=" * 100)
    print(path)

    if not path.exists():
        print("MISSING")
        return

    df = pd.read_csv(path)
    print("shape:", df.shape)
    print("columns:", list(df.columns))

    date_candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "time"}]
    if date_candidates:
        col = date_candidates[0]
        dates = pd.to_datetime(df[col], errors="coerce")
        print("date column:", col)
        print("date min:", dates.min())
        print("date max:", dates.max())
        print("invalid dates:", dates.isna().sum())

    print("\nmissing values:")
    print(df.isna().sum())

    print("\nhead:")
    print(df.head(3))

    print("\ntail:")
    print(df.tail(3))


def main() -> None:
    for path in RAW_FILES:
        inspect_file(path)


if __name__ == "__main__":
    main()
