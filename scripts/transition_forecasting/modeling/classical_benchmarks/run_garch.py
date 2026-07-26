from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from transition_forecasting.modeling.classical_benchmarks.garch import run_garch_benchmark
from transition_forecasting.modeling.classical_benchmarks.spec import load_frozen_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/classical_benchmarks/garch/run"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args()
    spec = load_frozen_spec(args.spec)
    garch = spec["garch"]
    run_dir, summary = run_garch_benchmark(
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        run_id=args.run_id,
        selection_folds=tuple(spec["selection_folds"]),
        confirmation_folds=tuple(spec["confirmation_folds"]),
        history=int(garch["history"]),
        minimum_history=int(garch["minimum_history"]),
        backend=str(garch["backend"]),
        max_workers=args.max_workers,
        limit_rows=args.limit_rows,
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
