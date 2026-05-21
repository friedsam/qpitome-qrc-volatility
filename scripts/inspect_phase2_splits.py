from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/processed/phase2_spy_vix_volatility.csv")

TRAIN_END = "2014-12-31"
VAL_END = "2019-12-31"

CALM_Q = 0.50
TURBULENT_Q = 0.80


def assign_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = "test"
    out.loc[out["date"] <= TRAIN_END, "split"] = "train"
    out.loc[(out["date"] > TRAIN_END) & (out["date"] <= VAL_END), "split"] = "val"
    return out


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = assign_split(df)

    train = df[df["split"] == "train"].copy()

    calm_thr = train["rv_20d"].quantile(CALM_Q)
    turbulent_thr = train["future_rv_20d"].quantile(TURBULENT_Q)

    df["calm_now"] = df["rv_20d"] <= calm_thr
    df["future_turbulent_20d"] = df["future_rv_20d"] >= turbulent_thr
    df["transition_event_20d"] = df["calm_now"] & df["future_turbulent_20d"]

    print("DATA")
    print("shape:", df.shape)
    print("date range:", df["date"].min(), "->", df["date"].max())
    print()

    print("TRAIN-ONLY THRESHOLDS")
    print("calm rv_20d q50:", calm_thr)
    print("future turbulent future_rv_20d q80:", turbulent_thr)
    print()

    print("SPLIT COUNTS")
    print(df["split"].value_counts().sort_index())
    print()

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print("=" * 80)
        print(split.upper())
        print("date range:", sub["date"].min(), "->", sub["date"].max())
        print("n:", len(sub))
        print()
        print("targets:")
        print(sub[["future_rv_5d", "future_rv_20d"]].describe())
        print()
        print("transition counts:")
        print(sub["transition_event_20d"].value_counts(dropna=False))
        print("transition rate:", sub["transition_event_20d"].mean())
        print()


if __name__ == "__main__":
    main()
