from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from transition_forecasting.modeling.classical_benchmarks.linear import run_linear_benchmark
from transition_forecasting.modeling.classical_benchmarks.spec import load_frozen_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/classical_benchmarks/linear/run"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    spec = load_frozen_spec(args.spec)
    run_dir, summary = run_linear_benchmark(
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        run_id=args.run_id,
        sequence_alpha=float(spec["linear"]["sequence_ridge_alpha"]),
        har_alpha=float(spec["linear"]["har_alpha"]),
        selection_folds=tuple(spec["selection_folds"]),
        confirmation_folds=tuple(spec["confirmation_folds"]),
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
