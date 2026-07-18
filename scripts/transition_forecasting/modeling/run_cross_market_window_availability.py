from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run


def load_panel(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, usecols=["date", "ticker"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise", utc=True).dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    return {
        ticker: group["date"].sort_values().drop_duplicates().to_numpy(dtype="datetime64[ns]")
        for ticker, group in frame.groupby("ticker", sort=True)
    }


def count_available_markets(
    origin: pd.Timestamp,
    panel_dates: dict[str, np.ndarray],
    *,
    required_observations: int = 40,
    max_staleness_days: int = 7,
) -> tuple[int, list[str]]:
    origin64 = np.datetime64(origin.tz_convert("UTC").tz_localize(None).normalize(), "ns")
    available: list[str] = []
    for ticker, dates in panel_dates.items():
        position = int(np.searchsorted(dates, origin64, side="right"))
        if position < required_observations:
            continue
        latest = dates[position - 1]
        staleness = int((origin64 - latest) / np.timedelta64(1, "D"))
        if staleness <= max_staleness_days:
            available.append(ticker)
    return len(available), available


def audit_windows(
    manifest: pd.DataFrame,
    panel_dates: dict[str, np.ndarray],
    *,
    required_observations: int = 40,
    max_staleness_days: int = 7,
) -> pd.DataFrame:
    frame = manifest.copy()
    frame["origin_date"] = pd.to_datetime(frame["origin_date"], errors="raise", utc=True)
    rows = []
    total = len(frame)
    for index, row in frame.iterrows():
        count, tickers = count_available_markets(
            row["origin_date"],
            panel_dates,
            required_observations=required_observations,
            max_staleness_days=max_staleness_days,
        )
        target = str(row["market_group"])
        rows.append({
            "sample_id": row["sample_id"],
            "origin_date": row["origin_date"],
            "year": int(row["origin_date"].year),
            "market_group": target,
            "available_markets": count,
            "target_exact_ticker_match": target in panel_dates,
            "target_available": target in tickers,
            "available_tickers": "|".join(tickers),
        })
        if (index + 1) % 500 == 0 or index + 1 == total:
            print(
                f"samples={index + 1:,}/{total:,} latest_year={row['origin_date'].year} "
                f"available_markets={count}",
                flush=True,
            )
    return pd.DataFrame(rows)


def summarize_by_year(audit: pd.DataFrame) -> pd.DataFrame:
    return audit.groupby("year", as_index=False).agg(
        samples=("sample_id", "count"),
        median_available_markets=("available_markets", "median"),
        min_available_markets=("available_markets", "min"),
        max_available_markets=("available_markets", "max"),
        target_exact_match_rate=("target_exact_ticker_match", "mean"),
        target_available_rate=("target_available", "mean"),
    )


def summarize_by_fold(audit: pd.DataFrame, rolling_manifest: pd.DataFrame) -> pd.DataFrame:
    lookup = rolling_manifest[["sample_id", "fold", "fold_split"]].drop_duplicates()
    merged = audit.merge(lookup, on="sample_id", how="inner")
    return merged.groupby(["fold", "fold_split"], as_index=False).agg(
        samples=("sample_id", "count"),
        median_available_markets=("available_markets", "median"),
        min_available_markets=("available_markets", "min"),
        max_available_markets=("available_markets", "max"),
        target_available_rate=("target_available", "mean"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dynamic cross-market availability for every Stage D input window.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/raw/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv"),
    )
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    parser.add_argument("--required-observations", type=int, default=40)
    parser.add_argument("--max-staleness-days", type=int, default=7)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/cross_market_window_availability"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    panel_dates = load_panel(args.panel)
    manifest = pd.read_csv(args.sample_manifest)
    rolling = pd.read_csv(args.rolling_manifest)

    audit = audit_windows(
        manifest,
        panel_dates,
        required_observations=args.required_observations,
        max_staleness_days=args.max_staleness_days,
    )
    by_year = summarize_by_year(audit)
    by_fold = summarize_by_fold(audit, rolling)

    audit.to_csv(run_dir / "window_availability_by_sample.csv", index=False)
    by_year.to_csv(run_dir / "window_availability_by_year.csv", index=False)
    by_fold.to_csv(run_dir / "window_availability_by_fold.csv", index=False)

    exact_manifest_markets = sorted(set(manifest["market_group"].astype(str)) & set(panel_dates))
    unmatched_manifest_markets = sorted(set(manifest["market_group"].astype(str)) - set(panel_dates))
    payload = {
        "test_evaluated": False,
        "panel_markets": len(panel_dates),
        "manifest_markets": int(manifest["market_group"].nunique()),
        "exact_manifest_market_matches": exact_manifest_markets,
        "unmatched_manifest_markets": unmatched_manifest_markets,
        "median_available_markets": float(audit["available_markets"].median()),
        "min_available_markets": int(audit["available_markets"].min()),
        "max_available_markets": int(audit["available_markets"].max()),
        "target_exact_match_rate": float(audit["target_exact_ticker_match"].mean()),
        "target_available_rate": float(audit["target_available"].mean()),
    }
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("\nSUMMARY", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    print("\nBY FOLD", flush=True)
    print(by_fold.to_string(index=False), flush=True)
    print("\nBY YEAR", flush=True)
    print(by_year.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
