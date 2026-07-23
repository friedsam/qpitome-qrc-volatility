from __future__ import annotations

import argparse
from pathlib import Path

from transition_forecasting.qrc.ladder_spacing_broad_har_reanalysis import (
    run_spacing_broad_har_reanalysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-score existing ladder distance caches against the established all-row HAR baseline."
    )
    parser.add_argument(
        "--spacing-run",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/run_ladder_spacing_assay/"
            "ladder_spacing_freeze_001"
        ),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/transition_forecasting/qrc/"
            "run_ladder_spacing_broad_har_reanalysis"
        ),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()
    run_dir = run_spacing_broad_har_reanalysis(
        spacing_run=args.spacing_run,
        results_root=args.results_root,
        run_id=args.run_id,
    )
    print(f"WROTE {run_dir}")


if __name__ == "__main__":
    main()
