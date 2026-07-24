from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.residual_head_qlike_dominance import (
    QlikeDominanceConfig,
    run_residual_head_qlike_dominance,
)


DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_residual_head_qlike_dominance"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose whether pooled exponential QLIKE selects a positive residual-head "
            "lambda because a small underprediction tail dominates wrong-sign calm cells."
        )
    )
    parser.add_argument("--root-cause-run", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lambda-cap", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = QlikeDominanceConfig(lambda_cap=float(args.lambda_cap))
    run_dir = run_residual_head_qlike_dominance(
        root_cause_run=args.root_cause_run,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
