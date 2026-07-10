"""Cross-market portability utilities for the fixed branching-state detector.

The portability question is deliberately narrow: does the unchanged causal detector
identify the same qualitative transient escape object in other broad equity markets?
No market-specific detector retuning is allowed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_intermediate_dynamics import (
    IntermediateProbeConfig,
    build_first_passage_episode_table,
)
from qpitome_qrc.regimes.branching_state import (
    BranchStateConfig,
    build_branch_episodes,
    detect_branch_state,
)


@dataclass(frozen=True)
class MarketSpec:
    """Raw-file schema and provenance identifier for one portability market."""

    market: str
    source: str
    date_column: str
    close_column: str
    date_format: str | None = None


DEFAULT_MARKETS: dict[str, MarketSpec] = {
    "nikkei_225": MarketSpec(
        market="Nikkei 225",
        source="FRED:NIKKEI225",
        date_column="observation_date",
        close_column="NIKKEI225",
    ),
    "ftse_100": MarketSpec(
        market="FTSE 100",
        source="WSJ:UKX",
        date_column="Date",
        close_column="Close",
        date_format="%m/%d/%y",
    ),
    "russell_2000": MarketSpec(
        market="Russell 2000",
        source="WSJ:RUT",
        date_column="Date",
        close_column="Close",
        date_format="%m/%d/%y",
    ),
}


def _normalized_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out


def load_market_close(path: Path, spec: MarketSpec) -> pd.DataFrame:
    """Load one raw market file into an ascending, unique daily close series."""
    raw = _normalized_columns(pd.read_csv(path))
    required = {spec.date_column, spec.close_column}
    missing = required - set(raw.columns)
    if missing:
        raise KeyError(f"Missing columns for {spec.market}: {sorted(missing)}")

    dates = pd.to_datetime(
        raw[spec.date_column],
        format=spec.date_format,
        errors="coerce",
    )
    closes = pd.to_numeric(raw[spec.close_column], errors="coerce")
    out = pd.DataFrame({"date": dates, "close": closes})
    out = (
        out.dropna(subset=["date", "close"])
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="last")
        .reset_index(drop=True)
    )
    if out.empty:
        raise ValueError(f"No valid observations for {spec.market}")
    if (out["close"] <= 0).any():
        raise ValueError(f"Nonpositive close found for {spec.market}")
    return out


def build_detector_input(close: pd.DataFrame) -> pd.DataFrame:
    """Map a generic close series to the unchanged detector input contract."""
    required = {"date", "close"}
    missing = required - set(close.columns)
    if missing:
        raise KeyError(f"Missing close-series columns: {sorted(missing)}")

    out = close.copy().sort_values("date").reset_index(drop=True)
    out["spy_adj_close"] = out["close"].astype(float)
    out["spy_log_return"] = np.log(out["spy_adj_close"]).diff()
    out["rv_5d"] = out["spy_log_return"].rolling(5).std(ddof=1) * np.sqrt(252.0)
    out["rv_20d"] = out["spy_log_return"].rolling(20).std(ddof=1) * np.sqrt(252.0)
    return out


def summarize_portability(
    market_key: str,
    spec: MarketSpec,
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    max_followup: int = 120,
) -> pd.DataFrame:
    """Return one-row detector-portability summary for a market."""
    if episodes.empty:
        return pd.DataFrame(
            [
                {
                    "market_key": market_key,
                    "market": spec.market,
                    "source": spec.source,
                    "first_date": daily["date"].min(),
                    "last_date": daily["date"].max(),
                    "n_daily_rows": len(daily),
                    "n_branch_episodes": 0,
                    "n_complete_120d": 0,
                    "n_recovery": 0,
                    "n_relapse": 0,
                    "n_unresolved_120d": 0,
                    "fraction_resolved_120d": np.nan,
                    "median_residence_days": np.nan,
                }
            ]
        )

    n_rows = len(daily)
    available_followup = n_rows - episodes["branch_idx"].astype(int) - 1
    complete_ids = set(
        episodes.loc[available_followup >= max_followup, "episode_id"].astype(int)
    )
    complete_events = events[events["episode_id"].astype(int).isin(complete_ids)].copy()
    resolved = complete_events[complete_events["resolved_within_followup"]].copy()
    counts = resolved["event_type"].value_counts()
    n_complete = len(complete_events)
    n_resolved = len(resolved)

    return pd.DataFrame(
        [
            {
                "market_key": market_key,
                "market": spec.market,
                "source": spec.source,
                "first_date": daily["date"].min(),
                "last_date": daily["date"].max(),
                "n_daily_rows": len(daily),
                "n_branch_episodes": len(episodes),
                "n_complete_120d": n_complete,
                "n_recovery": int(counts.get("recovery", 0)),
                "n_relapse": int(counts.get("relapse", 0)),
                "n_unresolved_120d": int(n_complete - n_resolved),
                "fraction_resolved_120d": (
                    float(n_resolved / n_complete) if n_complete else np.nan
                ),
                "median_residence_days": (
                    float(resolved["event_day"].median()) if n_resolved else np.nan
                ),
            }
        ]
    )


def audit_market_portability(
    path: Path,
    market_key: str,
    state_config: BranchStateConfig | None = None,
    max_followup: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one market and run the unchanged detector plus fixed-barrier escape audit."""
    if market_key not in DEFAULT_MARKETS:
        raise KeyError(f"Unknown market key: {market_key}")
    spec = DEFAULT_MARKETS[market_key]
    close = load_market_close(path, spec)
    daily = build_detector_input(close)
    detected = detect_branch_state(daily, state_config or BranchStateConfig())
    episodes = build_branch_episodes(detected, state_config or BranchStateConfig())
    events = build_first_passage_episode_table(
        detected,
        episodes,
        IntermediateProbeConfig(max_followup=max_followup),
    )
    summary = summarize_portability(
        market_key,
        spec,
        detected,
        episodes,
        events,
        max_followup=max_followup,
    )
    return detected, episodes, events, summary
