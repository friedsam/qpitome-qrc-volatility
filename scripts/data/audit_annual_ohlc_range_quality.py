from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.ohlc_range_quality import write_annual_range_quality
from experiments.runs import begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit annual nonzero OHLC range coverage for an index history.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/transition_forecasting/global_stock_indices_historical_data/"
            "individual_indices_data/^GSPC_data.csv"
        ),
    )
    parser.add_argument("--minimum-nonzero-fraction", type=float, default=0.95)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/quality/annual_ohlc_range_quality"),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = write_annual_range_quality(
        args.input,
        run_dir,
        minimum_nonzero_fraction=args.minimum_nonzero_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
