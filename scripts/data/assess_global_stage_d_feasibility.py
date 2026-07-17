from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.global_stage_d_feasibility import write_stage_d_feasibility
from experiments.runs import begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess Stage D positive-sample feasibility for the global catalogue.")
    parser.add_argument(
        "--catalogue-dir",
        type=Path,
        default=Path(
            "results/transition_forecasting/catalogue/global_transition_catalogue/"
            "global_transition_catalogue_003"
        ),
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "results/transition_forecasting/quality/global_index_ohlc_audit/"
            "global_index_audit_001/global_index_ohlc_inventory.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_d_feasibility"),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = write_stage_d_feasibility(
        args.catalogue_dir / "representative_transition_catalogue.csv",
        args.inventory,
        run_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
