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

from transition_forecasting.modeling.classical_benchmarks.garch import run_garch_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the causal Student-t GARCH(1,1) transition benchmark.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_forecasting/modeling/classical_benchmarks/garch"))
    parser.add_argument("--run-id")
    parser.add_argument("--selection-folds", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--confirmation-folds", type=int, nargs="+", default=[7, 8])
    parser.add_argument("--history", type=int, default=2500)
    parser.add_argument("--minimum-history", type=int, default=250)
    parser.add_argument("--backend", choices=["auto", "arch", "scipy"], default="auto")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args()
    run_dir, summary = run_garch_benchmark(
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        run_id=args.run_id,
        selection_folds=tuple(args.selection_folds),
        confirmation_folds=tuple(args.confirmation_folds),
        history=args.history,
        minimum_history=args.minimum_history,
        backend=args.backend,
        max_workers=args.max_workers,
        limit_rows=args.limit_rows,
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
