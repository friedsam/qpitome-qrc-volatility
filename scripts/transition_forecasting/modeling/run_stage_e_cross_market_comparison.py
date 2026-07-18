from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runs import begin_run
from transition_forecasting.modeling.cross_market_comparison import run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare cross-market ESN features against established Stage E baselines on a common rolling intersection."
    )
    parser.add_argument("--stage-d-run", type=Path, required=True)
    parser.add_argument("--rolling-manifest", type=Path, required=True)
    parser.add_argument("--tensor-run", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_cross_market_integrated"),
    )
    parser.add_argument("--run-id")
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, vars(args), run_id=args.run_id)
    summary = run_comparison(
        stage_d_run=args.stage_d_run,
        rolling_manifest=args.rolling_manifest,
        tensor_run=args.tensor_run,
        run_dir=run_dir,
        seeds=args.seeds,
        alphas=args.alphas,
    )
    print("\n" + json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
