from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.l5_template_increment_assay import (
    L5TemplateIncrementConfig,
    run_l5_template_increment_assay,
)


DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_l5_template_increment_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an intercept-only L5 crisis template against centered "
            "no-intercept QRC increments with independently signed early/late lambdas."
        )
    )
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--comparison-run", type=Path, required=True)
    parser.add_argument("--dataset-name", default="historical")
    parser.add_argument(
        "--selection-seeds",
        type=int,
        nargs="+",
        default=[20260721, 20260722, 20260723],
    )
    parser.add_argument(
        "--alpha-grid",
        type=float,
        nargs="+",
        default=[1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--signed-lambda-grid",
        type=float,
        nargs="+",
        default=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = L5TemplateIncrementConfig(
        selection_seeds=tuple(args.selection_seeds),
        alpha_grid=tuple(args.alpha_grid),
        signed_lambda_grid=tuple(args.signed_lambda_grid),
    )
    run_dir = run_l5_template_increment_assay(
        fold_dir=args.fold_dir,
        comparison_run=args.comparison_run,
        results_root=args.out_root,
        dataset_name=args.dataset_name,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
