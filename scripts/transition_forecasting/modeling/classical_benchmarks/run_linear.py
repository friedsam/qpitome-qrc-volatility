from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from transition_forecasting.modeling.classical_benchmarks.linear import run_linear_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persistence, canonical HAR, and sequence-ridge transition benchmarks.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/classical_benchmarks/linear"),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--selection-folds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--confirmation-folds", type=int, nargs="+", default=[7, 8])
    parser.add_argument(
        "--sequence-alphas", type=float, nargs="+",
        default=[10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
    )
    args = parser.parse_args()
    run_dir, summary = run_linear_benchmark(
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        run_id=args.run_id,
        selection_folds=tuple(args.selection_folds),
        confirmation_folds=tuple(args.confirmation_folds),
        sequence_alphas=tuple(args.sequence_alphas),
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
