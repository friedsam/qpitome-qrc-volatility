from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run

DATE_CANDIDATES = ("date", "datetime", "timestamp", "origin_date", "event_onset")
MARKET_CANDIDATES = ("market", "market_group", "symbol", "ticker", "asset", "index")
VALUE_CANDIDATES = ("close", "adj close", "adj_close", "value", "price", "volatility", "log_volatility")


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    return next((lookup[name] for name in candidates if name in lookup), None)


def audit_csv(path: Path, *, chunksize: int = 100_000) -> tuple[pd.DataFrame, dict[str, object]]:
    header = pd.read_csv(path, nrows=0)
    columns = list(header.columns)
    date_col = _pick(columns, DATE_CANDIDATES)
    market_col = _pick(columns, MARKET_CANDIDATES)
    value_col = _pick(columns, VALUE_CANDIDATES)
    if date_col is None or market_col is None:
        raise ValueError(f"could not identify date/market columns in {path}; columns={columns}")

    numeric_candidates = [column for column in columns if column not in {date_col, market_col}]
    market_counts: Counter[str] = Counter()
    market_dates: dict[str, set[pd.Timestamp]] = defaultdict(set)
    all_dates: set[pd.Timestamp] = set()
    duplicate_pairs = 0
    seen_pairs: set[tuple[str, pd.Timestamp]] = set()
    total_rows = 0
    numeric_nonmissing: Counter[str] = Counter()

    usecols = list(dict.fromkeys([date_col, market_col, *numeric_candidates]))
    for chunk_number, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize), start=1):
        dates = pd.to_datetime(chunk[date_col], errors="coerce", utc=True).dt.normalize()
        markets = chunk[market_col].astype("string")
        valid = dates.notna() & markets.notna()
        dates = dates[valid]
        markets = markets[valid]
        total_rows += int(valid.sum())

        for market, date in zip(markets.astype(str), dates, strict=False):
            key = (market, date)
            duplicate_pairs += int(key in seen_pairs)
            seen_pairs.add(key)
            market_counts[market] += 1
            market_dates[market].add(date)
            all_dates.add(date)

        for column in numeric_candidates:
            numeric_nonmissing[column] += int(pd.to_numeric(chunk.loc[valid, column], errors="coerce").notna().sum())

        print(
            f"chunk={chunk_number} rows={total_rows:,} markets={len(market_counts)} dates={len(all_dates):,}",
            flush=True,
        )

    if not all_dates:
        raise ValueError(f"no valid dated observations found in {path}")

    date_count = len(all_dates)
    rows = []
    for market in sorted(market_counts):
        dates = market_dates[market]
        rows.append({
            "market": market,
            "rows": market_counts[market],
            "dates": len(dates),
            "coverage_of_union": len(dates) / date_count,
            "min_date": min(dates),
            "max_date": max(dates),
        })
    per_market = pd.DataFrame(rows).sort_values(["dates", "market"], ascending=[False, True])

    value_rank = sorted(
        ((column, count / max(total_rows, 1)) for column, count in numeric_nonmissing.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if value_col is None and value_rank:
        value_col = value_rank[0][0]

    summary = {
        "path": str(path),
        "columns": columns,
        "date_column": date_col,
        "market_column": market_col,
        "suggested_value_column": value_col,
        "numeric_column_nonmissing_fraction": {name: fraction for name, fraction in value_rank},
        "rows": total_rows,
        "markets": len(market_counts),
        "union_dates": date_count,
        "min_date": str(min(all_dates)),
        "max_date": str(max(all_dates)),
        "duplicate_market_dates": duplicate_pairs,
        "markets_with_90pct_union_coverage": int((per_market["coverage_of_union"] >= 0.9).sum()),
        "markets_with_75pct_union_coverage": int((per_market["coverage_of_union"] >= 0.75).sum()),
    }
    return per_market, summary


def sample_overlap(per_market: pd.DataFrame, sample_manifest: Path) -> dict[str, object]:
    manifest = pd.read_csv(sample_manifest)
    date_col = _pick(list(manifest.columns), ("origin_date", "event_onset", "date"))
    market_col = _pick(list(manifest.columns), ("market_group", "market", "symbol", "ticker"))
    if date_col is None or market_col is None:
        return {"sample_manifest": str(sample_manifest), "usable": False}
    return {
        "sample_manifest": str(sample_manifest),
        "usable": True,
        "sample_date_column": date_col,
        "sample_market_column": market_col,
        "sample_rows": int(len(manifest)),
        "sample_markets": int(manifest[market_col].nunique()),
        "sample_min_date": str(pd.to_datetime(manifest[date_col], utc=True).min()),
        "sample_max_date": str(pd.to_datetime(manifest[date_col], utc=True).max()),
        "panel_markets": int(len(per_market)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream the complete raw market panel and quantify cross-market feasibility.")
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("data/raw/transition_forecasting/global_stock_indices_historical_data/all_indices_data.csv"),
    )
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/cross_market_panel_audit"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    per_market, summary = audit_csv(args.panel, chunksize=args.chunksize)
    overlap = sample_overlap(per_market, args.sample_manifest)
    per_market.to_csv(run_dir / "panel_coverage_by_market.csv", index=False)
    payload = {**summary, "sample_overlap": overlap}
    (run_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")

    print("\nPANEL SUMMARY", flush=True)
    print(json.dumps(payload, indent=2), flush=True)
    print("\nMARKET COVERAGE", flush=True)
    print(per_market.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
