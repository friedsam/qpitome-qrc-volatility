from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runs import begin_run
from transition_forecasting.modeling.chronological_rematched_dataset import (
    write_rematched_rolling_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strict fold-local rematched Stage E rolling datasets."
    )
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--test-fraction", type=float, default=0.17)
    parser.add_argument("--embargo-days", type=int, default=10)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/transition_forecasting/modeling/build_stage_e_chronological_dataset"
        ),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    summary = write_rematched_rolling_dataset(
        args.stage_d_run,
        run_dir,
        n_folds=args.n_folds,
        test_fraction=args.test_fraction,
        embargo_days=args.embargo_days,
        controls_per_positive=args.controls_per_positive,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
