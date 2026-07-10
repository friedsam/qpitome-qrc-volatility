"""Audit the unchanged branching-state detector across broad equity markets.

This is a narrow portability test, not a repetition of the full SPY discovery program.
For each market:
1. load the manually acquired raw daily close series;
2. apply the exact existing causal detector without retuning;
3. apply the exact existing volatility-scaled fixed barriers;
4. report episode counts, first-passage direction, residence, and unresolved fraction.

The audit does not claim that pooled market episodes are independent. Calendar-crisis
clustering is a separate downstream dependence audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from qpitome_qrc.regimes.branching_state import BranchStateConfig
from qpitome_qrc.regimes.market_portability import (
    DEFAULT_MARKETS,
    audit_market_portability,
)


DEFAULT_INPUTS = {
    "nikkei_225": Path("data/raw/portability/nikkei_225_fred_raw.csv"),
    "ftse_100": Path("data/raw/portability/ftse_100_raw.csv"),
    "russell_2000": Path("data/raw/portability/russell_2000_raw.csv"),
}
DEFAULT_OUTPUT = Path("results/diagnostics/cross_market_branch_portability_v1")

SOURCE_URLS = {
    "nikkei_225": "https://fred.stlouisfed.org/series/NIKKEI225",
    "ftse_100": "https://www.wsj.com/market-data/quotes/index/UK/UKX/historical-prices",
    "russell_2000": "https://www.wsj.com/market-data/quotes/index/RUT/historical-prices",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nikkei", type=Path, default=DEFAULT_INPUTS["nikkei_225"])
    parser.add_argument("--ftse", type=Path, default=DEFAULT_INPUTS["ftse_100"])
    parser.add_argument("--russell", type=Path, default=DEFAULT_INPUTS["russell_2000"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = {
        "nikkei_225": args.nikkei,
        "ftse_100": args.ftse,
        "russell_2000": args.russell,
    }
    for path in inputs.values():
        if not path.exists():
            raise FileNotFoundError(path)
    args.output.mkdir(parents=True, exist_ok=True)

    summaries: list[pd.DataFrame] = []
    config = BranchStateConfig()

    for market_key, path in inputs.items():
        detected, episodes, events, summary = audit_market_portability(
            path,
            market_key,
            state_config=config,
            max_followup=120,
        )
        market_output = args.output / market_key
        market_output.mkdir(parents=True, exist_ok=True)
        episodes.to_csv(market_output / "branch_episodes.csv", index=False)
        events.to_csv(market_output / "first_passage_events_120d.csv", index=False)
        summary.to_csv(market_output / "portability_summary.csv", index=False)
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(args.output / "market_portability_summary.csv", index=False)

    manifest = {
        "question": "Does the unchanged causal branching-state detector identify the same qualitative transient escape object in other broad equity markets?",
        "detector_retuning": False,
        "state_config": config.__dict__,
        "fixed_barrier_followup_days": 120,
        "market_inputs": {
            market_key: {
                "market": DEFAULT_MARKETS[market_key].market,
                "source": DEFAULT_MARKETS[market_key].source,
                "source_url": SOURCE_URLS[market_key],
                "raw_path": str(path),
            }
            for market_key, path in inputs.items()
        },
        "interpretation_limit": "Repeated market episodes are not assumed independent across synchronized global crises. Calendar-crisis clustering is evaluated separately.",
        "admission_logic": "A market is scientifically portable only if the unchanged dimensionless detector finds recurrent finite-residence episodes that mostly resolve through the same fixed-barrier first-passage structure without market-specific parameter surgery.",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Cross-market branching-state portability audit")
    print(combined.to_string(index=False))
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
