from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.bivariate_all_orders_readout_assay import (
    BivariateAllOrdersReadoutConfig,
    run_bivariate_all_orders_readout_assay,
)

DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_bivariate_all_orders_readout_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse a completed bilinear mixing run and fit joint readouts over the "
            "palindrome and all four ordered D/X observable banks without rerunning "
            "the quantum simulation."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Completed bivariate_bilinear_mixing run directory or ZIP archive.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=None,
        help="Override the source run's paired-permutation count.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_bivariate_all_orders_readout_assay(
        source_run=args.source_run,
        results_root=args.out_root,
        config=BivariateAllOrdersReadoutConfig(permutations=args.permutations),
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
