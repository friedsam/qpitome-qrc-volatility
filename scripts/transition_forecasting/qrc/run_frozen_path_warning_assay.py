from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc import frozen_path_warning_assay as warning_assay
from transition_forecasting.qrc.forecast_warning_bootstrap import (
    cluster_bootstrap_ap,
    paired_cluster_bootstrap_ap,
)
from transition_forecasting.qrc.forecast_warning_tools import (
    FrozenPathWarningConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive L5 transition-warning scores from already frozen ten-day HAR, "
            "chain and Rydberg-ladder forecasts without rerunning the reservoir."
        )
    )
    parser.add_argument("--development-run-dir", type=Path, required=True)
    parser.add_argument("--confirmation-run-dir", type=Path, required=True)
    parser.add_argument("--lead", type=int, default=5)
    parser.add_argument("--baseline-horizon", type=int, default=1)
    parser.add_argument("--onset-horizon", type=int, default=5)
    parser.add_argument("--forecast-horizons", type=int, default=10)
    parser.add_argument(
        "--development-folds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--confirmation-folds",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8],
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_frozen_path_warning_assay"
        ),
    )
    parser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FrozenPathWarningConfig(
        lead=args.lead,
        baseline_horizon=args.baseline_horizon,
        onset_horizon=args.onset_horizon,
        forecast_horizons=args.forecast_horizons,
        development_folds=tuple(args.development_folds),
        confirmation_folds=tuple(args.confirmation_folds),
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    warning_assay.cluster_bootstrap_ap = cluster_bootstrap_ap
    warning_assay.paired_cluster_bootstrap_ap = paired_cluster_bootstrap_ap
    run_dir = warning_assay.run_frozen_path_warning_assay(
        development_run_dir=args.development_run_dir,
        confirmation_run_dir=args.confirmation_run_dir,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
