from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.global_transition_catalogue import write_global_transition_outputs
from experiments.runs import begin_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an expanded global persistent-volatility transition catalogue."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/transition_forecasting/global_stock_indices_historical_data"),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "results/transition_forecasting/audit_global_index_ohlc/"
            "global_index_audit_001/global_index_ohlc_inventory.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/build_global_transition_catalogue"),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = write_global_transition_outputs(args.data_dir, args.inventory, run_dir)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
