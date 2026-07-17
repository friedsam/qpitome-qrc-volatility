from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.transition_events import detect_onsets, event_anatomy, log_parkinson

TRAIN_CUTOFF = pd.Timestamp("2016-01-01")
CLUSTER_DAYS = 7
MIN_TRAIN_DAYS = 250

EXCLUDED_INDICES = {
    "all_indices_data": "consolidated_duplicate_panel",
    "^VIX_data": "volatility_index_not_equity_index",
}

MARKET_GROUPS = {
    "000001.SS_data": "china_mainland",
    "399001.SZ_data": "china_mainland",
    "^AORD_data": "australia",
    "^AXJO_data": "australia",
    "^DJI_data": "united_states",
    "^GSPC_data": "united_states",
    "^IXIC_data": "united_states",
    "^NYA_data": "united_states",
    "^RUT_data": "united_states",
    "^XAX_data": "united_states",
    "^FCHI_data": "france",
    "^FTSE_data": "united_kingdom",
    "^GDAXI_data": "germany",
    "^GSPTSE_data": "canada",
    "^HSI_data": "hong_kong",
    "^IPSA_data": "chile",
    "^JKSE_data": "indonesia",
    "^KLSE_data": "malaysia",
    "^KS11_data": "south_korea",
    "^MERV_data": "argentina",
    "^MXX_data": "mexico",
    "^N100_data": "europe_regional",
    "^N225_data": "japan",
    "^NZ50_data": "new_zealand",
    "^STI_data": "singapore",
    "^STOXX50E_data": "eurozone_regional",
    "^TA125.TA_data": "israel",
    "^TWII_data": "taiwan",
    "^BFX_data": "belgium",
    "^BSESN_data": "india",
    "^BVSP_data": "brazil",
}

REPRESENTATIVE_PRIORITY = {
    "united_states": ["^GSPC_data", "^NYA_data", "^DJI_data", "^IXIC_data", "^RUT_data", "^XAX_data"],
    "australia": ["^AORD_data", "^AXJO_data"],
    "china_mainland": ["000001.SS_data", "399001.SZ_data"],
}


def _load_generic_ohlc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    normalized = {column.strip().lower(): column for column in frame.columns}
    required = {"date", "high", "low"}
    if not required.issubset(normalized):
        raise ValueError(f"{path}: missing date/high/low columns")
    result = pd.DataFrame(
        {
            "Date": pd.to_datetime(frame[normalized["date"]], errors="coerce"),
            "High": pd.to_numeric(frame[normalized["high"]], errors="coerce"),
            "Low": pd.to_numeric(frame[normalized["low"]], errors="coerce"),
        }
    ).dropna()
    result = result[(result.High > 0) & (result.Low > 0) & (result.High >= result.Low)]
    return result.drop_duplicates("Date", keep="last").set_index("Date").sort_index()


def _cluster_dates(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Cluster unique onset dates in fixed windows anchored at each cluster start."""
    if catalogue.empty:
        result = catalogue.copy()
        result["episode_id"] = pd.Series(dtype=str)
        return result
    dates = sorted(pd.to_datetime(catalogue["onset_date"]).unique())
    clusters: list[list[pd.Timestamp]] = []
    for raw_date in dates:
        date = pd.Timestamp(raw_date)
        if not clusters or (date - clusters[-1][0]).days > CLUSTER_DAYS:
            clusters.append([date])
        else:
            clusters[-1].append(date)
    mapping = {
        date: f"GE{number:03d}"
        for number, cluster in enumerate(clusters, 1)
        for date in cluster
    }
    result = catalogue.copy()
    result["episode_id"] = pd.to_datetime(result["onset_date"]).map(mapping)
    return result


def _representative_rank(index_name: str, market_group: str) -> int:
    priority = REPRESENTATIVE_PRIORITY.get(market_group, [index_name])
    try:
        return priority.index(index_name)
    except ValueError:
        return len(priority)


def _effective_start_map(range_quality_path: Path) -> dict[str, pd.Timestamp]:
    quality = pd.read_csv(range_quality_path)
    required = {"index", "recommended_effective_start"}
    if not required.issubset(quality.columns):
        raise ValueError(f"{range_quality_path}: missing columns {sorted(required - set(quality.columns))}")
    starts = pd.to_datetime(quality["recommended_effective_start"], errors="coerce")
    if starts.isna().any():
        bad = quality.loc[starts.isna(), "index"].astype(str).tolist()
        raise ValueError(f"{range_quality_path}: missing effective starts for {bad}")
    return dict(zip(quality["index"].astype(str), starts))


def build_global_transition_catalogue(
    data_root: Path,
    inventory_path: Path,
    range_quality_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    del data_root
    inventory = pd.read_csv(inventory_path)
    eligible = inventory[inventory["eligible"].astype(bool)].copy()
    effective_starts = _effective_start_map(range_quality_path)
    raw_events: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    applied_starts: dict[str, str] = {}

    for _, row in eligible.iterrows():
        index_name = str(row["index"])
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = Path.cwd() / path
        if index_name in EXCLUDED_INDICES:
            skipped.append({"index": index_name, "reason": EXCLUDED_INDICES[index_name]})
            continue
        if index_name not in effective_starts:
            skipped.append({"index": index_name, "reason": "missing_range_quality_effective_start"})
            continue

        effective_start = effective_starts[index_name]
        market_group = MARKET_GROUPS.get(index_name, index_name)
        frame = _load_generic_ohlc(path)
        frame = frame[frame.index >= effective_start]
        series = log_parkinson(frame)
        applied_starts[index_name] = str(effective_start.date())
        training = series[series.index < TRAIN_CUTOFF]
        if len(training) < MIN_TRAIN_DAYS:
            skipped.append({"index": index_name, "reason": "insufficient_pre_2016_training_history_after_range_filter"})
            continue
        threshold = float(training.quantile(0.80))
        for onset in detect_onsets(series, threshold):
            onset_date = pd.Timestamp(series.index[onset])
            if onset < 40:
                continue
            raw_events.append(
                {
                    "index": index_name,
                    "market_group": market_group,
                    "onset_date": onset_date,
                    "effective_start": effective_start,
                    "threshold": threshold,
                    "train_days": int(len(training)),
                    "history_start": str(series.index.min().date()),
                    "history_end": str(series.index.max().date()),
                    **event_anatomy(series, onset, threshold),
                }
            )

    raw = pd.DataFrame(raw_events)
    if raw.empty:
        clustered = raw.copy()
        representative = raw.copy()
    else:
        raw = raw.sort_values(["onset_date", "market_group", "index"]).reset_index(drop=True)
        clustered = _cluster_dates(raw)
        clustered["representative_rank"] = [
            _representative_rank(str(index), str(group))
            for index, group in zip(clustered["index"], clustered["market_group"])
        ]
        representative = (
            clustered.sort_values(["episode_id", "market_group", "representative_rank", "onset_date"])
            .drop_duplicates(["episode_id", "market_group"], keep="first")
            .reset_index(drop=True)
        )

    episodes = (
        representative.groupby("episode_id", as_index=False)
        .agg(
            start_date=("onset_date", "min"),
            end_date=("onset_date", "max"),
            n_markets=("market_group", "nunique"),
            n_index_events=("index", "size"),
        )
        if not representative.empty
        else pd.DataFrame(columns=["episode_id", "start_date", "end_date", "n_markets", "n_index_events"])
    )

    summary = {
        "eligible_inventory_files": int(len(eligible)),
        "range_quality_file": str(range_quality_path),
        "effective_starts_applied": applied_starts,
        "excluded_or_skipped_indices": skipped,
        "raw_index_events": int(len(raw)),
        "global_temporal_clusters": int(clustered["episode_id"].nunique()) if not clustered.empty else 0,
        "representative_market_events": int(len(representative)),
        "representative_global_clusters": int(episodes["episode_id"].nunique()) if not episodes.empty else 0,
        "markets_retained": int(representative["market_group"].nunique()) if not representative.empty else 0,
        "events_by_index": raw.groupby("index").size().sort_values(ascending=False).to_dict() if not raw.empty else {},
        "events_by_market": representative.groupby("market_group").size().sort_values(ascending=False).to_dict() if not representative.empty else {},
        "events_by_decade": (
            representative.assign(decade=(pd.to_datetime(representative["onset_date"]).dt.year // 10) * 10)
            .groupby("decade")
            .size()
            .to_dict()
            if not representative.empty
            else {}
        ),
    }
    return raw, clustered, representative, {"summary": summary, "episodes": episodes.to_dict(orient="records")}


def write_global_transition_outputs(
    data_root: Path,
    inventory_path: Path,
    range_quality_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    raw, clustered, representative, report = build_global_transition_catalogue(
        data_root,
        inventory_path,
        range_quality_path,
    )
    raw.to_csv(run_dir / "raw_transition_catalogue.csv", index=False)
    clustered.to_csv(run_dir / "clustered_transition_catalogue.csv", index=False)
    representative.to_csv(run_dir / "representative_transition_catalogue.csv", index=False)
    pd.DataFrame(report["episodes"]).to_csv(run_dir / "global_episode_catalogue.csv", index=False)
    (run_dir / "transition_count_summary.json").write_text(
        json.dumps(report["summary"], indent=2, default=str) + "\n"
    )
    return report["summary"]
