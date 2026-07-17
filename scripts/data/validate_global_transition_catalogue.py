from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.global_transition_validation import write_validation_outputs
from experiments.runs import begin_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the expanded global transition catalogue before Stage D.")
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(
            "results/transition_forecasting/quality/global_index_ohlc_audit/"
            "global_index_audit_001/global_index_ohlc_inventory.csv"
        ),
    )
    parser.add_argument(
        "--catalogue-dir",
        type=Path,
        default=Path(
            "results/transition_forecasting/catalogue/global_transition_catalogue/"
            "global_transition_catalogue_001"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/catalogue/global_transition_validation"),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = write_validation_outputs(
        args.inventory,
        args.catalogue_dir / "raw_transition_catalogue.csv",
        args.catalogue_dir / "representative_transition_catalogue.csv",
        args.catalogue_dir / "global_episode_catalogue.csv",
        run_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
