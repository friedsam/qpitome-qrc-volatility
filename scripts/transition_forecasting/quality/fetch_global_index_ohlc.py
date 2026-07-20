from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.data.acquisition import acquire_source

DEFAULT_DESTINATION = Path("data/raw/transition_forecasting/global_stock_indices_historical_data")
DEFAULT_FALLBACK = Path("data/fallback/transition_forecasting/global_stock_indices_historical_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquire and verify the global stock-index raw OHLC snapshot, using the repository fallback when needed."
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--fallback", type=Path, default=DEFAULT_FALLBACK)
    parser.add_argument("--source-mode", choices=("auto", "live", "fallback"), default="auto")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination only after a new candidate has passed source validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = acquire_source(
        args.destination,
        args.fallback,
        source_mode=args.source_mode,
        force=args.force,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
