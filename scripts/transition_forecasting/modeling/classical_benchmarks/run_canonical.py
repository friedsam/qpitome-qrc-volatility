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

from transition_forecasting.modeling.classical_benchmarks.canonical import run_canonical_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine completed classical runs on their exact common validation rows.")
    parser.add_argument("--linear-run", type=Path, required=True)
    parser.add_argument("--garch-run", type=Path, required=True)
    parser.add_argument("--esn-run", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/transition_forecasting/modeling/classical_benchmarks/canonical"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_dir, summary = run_canonical_comparison(
        linear_run=args.linear_run,
        garch_run=args.garch_run,
        esn_run=args.esn_run,
        results_root=args.results_root,
        run_id=args.run_id,
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
