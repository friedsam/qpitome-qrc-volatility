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

from transition_forecasting.modeling.classical_benchmarks.esn_tuning import (
    run_esn_tuning_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the direct ESN on folds 4-6 with transition-first guardrails, then confirm on folds 7-8."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/classical_benchmarks/esn_tuning"),
    )
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_dir, summary = run_esn_tuning_benchmark(
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        run_id=args.run_id,
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
