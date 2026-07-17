from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _market_years_by_decade(inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    eligible = inventory[inventory["eligible"].astype(bool)].copy()
    eligible = eligible[~eligible["index"].isin(["all_indices_data", "^VIX_data"])]
    for _, row in eligible.iterrows():
        start = pd.Timestamp(row["start"])
        end = pd.Timestamp(row["end"])
        for decade in range((start.year // 10) * 10, (end.year // 10) * 10 + 1, 10):
            left = max(start, pd.Timestamp(f"{decade}-01-01"))
            right = min(end, pd.Timestamp(f"{decade + 9}-12-31"))
            if right >= left:
                years = (right - left).days / 365.2425
                rows.append({"decade": decade, "market_years": years})
    return pd.DataFrame(rows).groupby("decade", as_index=False)["market_years"].sum()


def validate_global_catalogue(
    inventory_path: Path,
    raw_catalogue_path: Path,
    representative_catalogue_path: Path,
    episode_catalogue_path: Path,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(inventory_path)
    raw = pd.read_csv(raw_catalogue_path, parse_dates=["onset_date"])
    representative = pd.read_csv(representative_catalogue_path, parse_dates=["onset_date"])
    episodes = pd.read_csv(episode_catalogue_path, parse_dates=["start_date", "end_date"])

    episodes = episodes.copy()
    episodes["span_days"] = (episodes["end_date"] - episodes["start_date"]).dt.days
    episodes["is_singleton_market"] = episodes["n_markets"] == 1
    episodes["is_chain_cluster"] = episodes["span_days"] > 7

    cluster_distribution = (
        episodes.groupby(["n_markets", "span_days"], as_index=False)
        .size()
        .rename(columns={"size": "episodes"})
    )

    events_by_decade = (
        representative.assign(decade=(representative["onset_date"].dt.year // 10) * 10)
        .groupby("decade", as_index=False)
        .size()
        .rename(columns={"size": "events"})
    )
    exposure = _market_years_by_decade(inventory)
    decade_rates = events_by_decade.merge(exposure, on="decade", how="left")
    decade_rates["events_per_100_market_years"] = (
        100.0 * decade_rates["events"] / decade_rates["market_years"]
    )

    summary: dict[str, object] = {
        "raw_index_events": int(len(raw)),
        "representative_market_events": int(len(representative)),
        "global_episodes": int(len(episodes)),
        "singleton_market_episodes": int(episodes["is_singleton_market"].sum()),
        "multi_market_episodes": int((~episodes["is_singleton_market"]).sum()),
        "singleton_fraction": float(episodes["is_singleton_market"].mean()),
        "maximum_cluster_span_days": int(episodes["span_days"].max()),
        "chain_cluster_count": int(episodes["is_chain_cluster"].sum()),
        "median_markets_per_episode": float(episodes["n_markets"].median()),
        "maximum_markets_per_episode": int(episodes["n_markets"].max()),
        "episodes_with_at_least_3_markets": int((episodes["n_markets"] >= 3).sum()),
        "episodes_with_at_least_5_markets": int((episodes["n_markets"] >= 5).sum()),
        "market_count": int(representative["market_group"].nunique()),
    }
    return summary, cluster_distribution, decade_rates


def write_validation_outputs(
    inventory_path: Path,
    raw_catalogue_path: Path,
    representative_catalogue_path: Path,
    episode_catalogue_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    summary, cluster_distribution, decade_rates = validate_global_catalogue(
        inventory_path,
        raw_catalogue_path,
        representative_catalogue_path,
        episode_catalogue_path,
    )
    cluster_distribution.to_csv(run_dir / "cluster_size_span_distribution.csv", index=False)
    decade_rates.to_csv(run_dir / "events_per_100_market_years.csv", index=False)
    (run_dir / "global_catalogue_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
