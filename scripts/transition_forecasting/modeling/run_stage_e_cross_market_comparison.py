from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runs import begin_run
from transition_forecasting.modeling.cross_market_comparison import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Rolling validation comparison of univariate and compact cross-market ridge/ESN models.")
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    parser.add_argument("--tensor-run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17])
    parser.add_argument("--ridge-alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--esn-alphas", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_cross_market_comparison"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    summary = run_comparison(
        sample_manifest=args.sample_manifest,
        rolling_manifest=args.rolling_manifest,
        tensor_run=args.tensor_run,
        run_dir=run_dir,
        seeds=args.seeds,
        ridge_alphas=args.ridge_alphas,
        esn_alphas=args.esn_alphas,
    )
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
