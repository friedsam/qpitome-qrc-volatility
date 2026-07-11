"""Build the exploratory day-0..5 path panel for the locked day-5 risk set.

This helper is intentionally limited to the experiment branch. It reconstructs
five causal post-entry path steps for Nikkei 225, FTSE 100, Russell 2000, and
SPY, then merges them onto the existing day-5 landmark frame.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
)
from qpitome_qrc.regimes.branch_transition_path import (
    TRANSITION_PATH_COLUMNS,
    add_transition_path_channels,
)
from qpitome_qrc.regimes.market_portability import audit_market_portability

FOLLOWUP = 120
LANDMARK = 5
INPUTS = {
    "nikkei_225": Path("data/raw/portability/nikkei_225_fred_raw.csv"),
    "ftse_100": Path("data/raw/portability/ftse_100_raw.csv"),
    "russell_2000": Path("data/raw/portability/russell_2000_raw.csv"),
}
DEFAULT_OUT = Path("scratch/path_panel_day0_5.csv")


def complete(episodes: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    return episodes[
        (n_rows - episodes["branch_idx"].astype(int) - 1) >= FOLLOWUP
    ].copy()


def path_rows(
    market_key: str,
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    daily = add_transition_path_channels(daily)
    events = events[
        events["resolved_within_followup"]
        & events["event_type"].isin(["recovery", "relapse"])
    ].copy()
    episodes = episodes.merge(
        events[
            [
                "episode_id",
                "event_day",
                "event_type",
                "upper_barrier",
                "lower_barrier",
            ]
        ],
        on="episode_id",
    )

    prices = daily["spy_adj_close"].to_numpy(float)
    rows = []
    for ep in episodes.itertuples(index=False):
        branch_idx = int(ep.branch_idx)
        if int(ep.event_day) <= LANDMARK or branch_idx + LANDMARK >= len(daily):
            continue

        record = {
            "market_key": market_key,
            "episode_id": int(ep.episode_id),
        }
        branch_price = prices[branch_idx]
        for day_offset in range(1, LANDMARK + 1):
            record[f"r_d{day_offset}"] = (
                prices[branch_idx + day_offset] / branch_price - 1.0
            )
            day = daily.iloc[branch_idx + day_offset]
            for channel in TRANSITION_PATH_COLUMNS:
                record[f"{channel}_d{day_offset}"] = float(day[channel])
        rows.append(record)

    return pd.DataFrame(rows)


def main(out: Path = DEFAULT_OUT) -> None:
    frames = []

    for market_key, path in INPUTS.items():
        daily, episodes, events, _ = audit_market_portability(path, market_key)
        episodes = complete(episodes, len(daily))
        events = events[events["episode_id"].isin(episodes["episode_id"])].copy()
        frame = path_rows(market_key, daily, episodes, events)
        frames.append(frame)
        print(market_key, len(frame))

    daily = (
        pd.read_csv(
            "data/processed/phase3_spy_vix_volatility_extended.csv",
            parse_dates=["date"],
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    episodes = pd.read_csv(
        "results/regimes/branching_extractor_audit_v1/mid_grid_episodes.csv",
        parse_dates=["branch_date"],
    )
    episodes = complete(episodes, len(daily))
    events = build_first_passage_episode_table(
        daily,
        episodes,
        IntermediateProbeConfig(max_followup=FOLLOWUP),
    )
    spy_frame = path_rows("spy", daily, episodes, events)
    frames.append(spy_frame)
    print("spy", len(spy_frame))

    panel = pd.concat(frames, ignore_index=True)
    landmark = pd.read_csv(
        "results/baselines/cross_market_day5_direction_v1/day5_landmark_frame.csv"
    )
    merged = landmark.merge(
        panel,
        on=["market_key", "episode_id"],
        how="inner",
    )

    print("landmark frame:", len(landmark), "merged:", len(merged))
    error = np.abs(
        merged["r_d5"] - merged["current_return_5d_from_branch"]
    ).max()
    print("max |r_d5 - current_return_5d_from_branch| =", error)

    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print("saved:", out)


if __name__ == "__main__":
    main()
