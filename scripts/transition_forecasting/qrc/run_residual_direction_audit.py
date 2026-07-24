from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.residual_direction_audit import (
    ResidualDirectionAuditConfig,
    run_residual_direction_audit,
)


DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_residual_direction_audit"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic residual-correction sanity cases and audit the sign "
            "behavior of saved QRC validation predictions."
        )
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dataset", default="historical")
    parser.add_argument("--later-folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--split-horizon", type=int, default=4)
    parser.add_argument("--lambda-min", type=float, default=-2.0)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--synthetic-seed", type=int, default=20260723)
    parser.add_argument("--synthetic-rows", type=int, default=600)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResidualDirectionAuditConfig(
        dataset=args.dataset,
        later_folds=tuple(args.later_folds),
        lead=args.lead,
        split_horizon=args.split_horizon,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_step=args.lambda_step,
        synthetic_seed=args.synthetic_seed,
        synthetic_rows=args.synthetic_rows,
    )
    run_dir = run_residual_direction_audit(
        predictions_path=args.predictions,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
