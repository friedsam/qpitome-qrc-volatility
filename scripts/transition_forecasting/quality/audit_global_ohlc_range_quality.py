from __future__ import annotations

import argparse
import json
from pathlib import Path

from transition_forecasting.quality.global_range_quality import write_global_range_quality
from experiments.runs import begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit effective OHLC range-history starts across all eligible indices.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "results/transition_forecasting/quality/global_index_ohlc_audit/"
            "global_index_audit_001/global_index_ohlc_inventory.csv"
        ),
    )
    parser.add_argument("--minimum-nonzero-fraction", type=float, default=0.95)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/quality/global_ohlc_range_quality"),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = write_global_range_quality(
        args.inventory,
        run_dir,
        minimum_nonzero_fraction=args.minimum_nonzero_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
