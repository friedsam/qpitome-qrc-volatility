from __future__ import annotations

import argparse
from pathlib import Path

from experiments.runs import begin_run
from transition_forecasting.modeling.cross_market_parameter_screen import (
    CONFIGS,
    REFINEMENT_CONFIGS,
    run_screen,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact parameter screen for the cross-market ESN.")
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    parser.add_argument("--tensor-run", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--profile", choices=["initial", "refinement"], default="initial")
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_cross_market_parameter_screen"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    available = CONFIGS if args.profile == "initial" else REFINEMENT_CONFIGS
    max_configs = len(available) if args.max_configs is None else args.max_configs
    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    run_screen(
        stage_d_run=args.stage_d_run,
        rolling_manifest=args.rolling_manifest,
        tensor_run=args.tensor_run,
        run_dir=run_dir,
        seeds=tuple(args.seeds),
        max_configs=max_configs,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
