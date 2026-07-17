from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data.early_ohlc_quality import write_early_ohlc_audit
from experiments.runs import begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit repeated OHLC values in the earliest index history.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/raw/transition_forecasting/global_stock_indices_historical_data/"
            "individual_indices_data/^GSPC_data.csv"
        ),
    )
    parser.add_argument("--early-end", type=str, default="1950-01-01")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/audit_early_ohlc_quality"),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = write_early_ohlc_audit(args.input, run_dir, pd.Timestamp(args.early_end))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
