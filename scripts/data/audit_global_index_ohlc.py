from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.global_index_ohlc_audit import audit_directory
from experiments.runs import begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a directory of global-index daily OHLC CSV files.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/quality/global_index_ohlc_audit"),
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--min-years", type=float, default=15.0)
    parser.add_argument("--min-valid-fraction", type=float, default=0.98)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    inventory, summary = audit_directory(
        args.data_dir,
        min_years=args.min_years,
        min_valid_fraction=args.min_valid_fraction,
    )
    inventory.to_csv(run_dir / "global_index_ohlc_inventory.csv", index=False)
    (run_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
