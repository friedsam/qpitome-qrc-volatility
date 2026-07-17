from __future__ import annotations

import argparse
from pathlib import Path

from data.transition_events import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Build persistent-volatility transition events and matched controls.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.run_dir, args.seed)


if __name__ == "__main__":
    main()
