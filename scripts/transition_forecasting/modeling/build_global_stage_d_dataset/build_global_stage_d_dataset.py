from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runs import begin_run
from transition_forecasting.modeling.global_stage_d_dataset import write_global_stage_d_dataset
from transition_forecasting.modeling.stage_d_candidate_pool import write_global_candidate_pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the global Stage D matched dataset and full control candidate pool."
    )
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
        default=Path(
            "results/transition_forecasting/modeling/build_global_stage_d_dataset"
        ),
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument(
        "--skip-candidate-pool",
        action="store_true",
        help="Build only the legacy matched Stage D artifact.",
    )
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    catalogue_path = args.catalogue_dir / "representative_transition_catalogue.csv"

    stage_d_summary = write_global_stage_d_dataset(
        catalogue_path,
        args.inventory,
        run_dir,
    )
    candidate_summary = None
    if not args.skip_candidate_pool:
        candidate_summary = write_global_candidate_pool(
            catalogue_path,
            args.inventory,
            run_dir,
        )

    payload = {
        "stage_d": stage_d_summary,
        "candidate_pool": candidate_summary,
        "candidate_pool_built": candidate_summary is not None,
    }
    (run_dir / "build_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
