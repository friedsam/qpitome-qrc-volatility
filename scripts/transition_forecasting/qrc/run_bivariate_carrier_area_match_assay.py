from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.bivariate_carrier_area_match_assay import (
    BivariateCarrierAreaMatchConfig,
    run_bivariate_carrier_area_match_assay,
)

DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_bivariate_carrier_area_match_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the completed pure-slot and full-strength carrier maps with a "
            "first-order area-matched carrier map that halves the global field strengths "
            "while preserving total duration and interaction evolution."
        )
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        required=True,
        help="Completed bivariate_carrier_crossmix run directory or ZIP archive.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional subset of seeds already present in the source run.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        help="Optional paired-permutation count; defaults to the source run value.",
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BivariateCarrierAreaMatchConfig(
        seeds=tuple(args.seeds) if args.seeds is not None else None,
        permutations=args.permutations,
    )
    run_dir = run_bivariate_carrier_area_match_assay(
        source_run=args.source_run,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
