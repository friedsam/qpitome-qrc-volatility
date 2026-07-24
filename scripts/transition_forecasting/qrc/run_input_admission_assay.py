from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.input_admission_assay import (
    InputAdmissionConfig,
    run_input_admission_assay,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test downside-return, intraday-range, and overnight-return inputs "
            "on the frozen handoff rows."
        )
    )
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path(
            "data/fallback/transition_forecasting/"
            "global_stock_indices_historical_data/all_indices_data.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 5])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_input_admission_assay(
        evaluation_manifest=args.evaluation_manifest,
        panel_path=args.panel,
        output_dir=args.out_dir,
        config=InputAdmissionConfig(
            folds=tuple(args.folds),
            leads=tuple(args.leads),
        ),
    )
    print(f"WROTE {output}")


if __name__ == "__main__":
    main()
