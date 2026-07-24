from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.residual_head_root_cause_archive import (
    run_residual_head_root_cause_from_archive,
)
from transition_forecasting.qrc.residual_head_root_cause_assay import (
    ResidualHeadRootCauseConfig,
)


DEFAULT_RESULTS_ROOT = Path(
    "results/transition_forecasting/qrc/run_residual_head_root_cause_assay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the signed residual-head root-cause assay directly from the immutable "
            "ladder_readout_upgrade ZIP or unzipped result directory."
        )
    )
    parser.add_argument("--feature-archive-source", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--later-folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--out-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResidualHeadRootCauseConfig(
        folds=tuple(args.folds),
        later_folds=tuple(args.later_folds),
    )
    run_dir = run_residual_head_root_cause_from_archive(
        feature_archive_source=args.feature_archive_source,
        results_root=args.out_root,
        config=config,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
