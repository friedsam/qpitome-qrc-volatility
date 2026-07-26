from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
for candidate in (REPO_ROOT, REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from transition_forecasting.modeling.classical_benchmarks.esn_frozen import run_frozen_esn_benchmark
from transition_forecasting.modeling.classical_benchmarks.spec import load_frozen_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/classical_benchmarks/esn/run"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--spec", type=Path)
    args = parser.parse_args()
    spec = load_frozen_spec(args.spec)
    esn = spec["esn"]
    config = {
        "config_id": esn["config_id"],
        "n": int(esn["n"]),
        "connectivity": float(esn["connectivity"]),
        "sr": float(esn["spectral_radius"]),
        "inp": float(esn["input_scale"]),
        "leak": float(esn["leak"]),
    }
    run_dir, summary = run_frozen_esn_benchmark(
        dataset_root=args.dataset_root,
        results_root=args.results_root,
        run_id=args.run_id,
        config=config,
        alpha=float(esn["alpha"]),
        seeds=tuple(int(seed) for seed in esn["seeds"]),
        selection_folds=tuple(spec["selection_folds"]),
        confirmation_folds=tuple(spec["confirmation_folds"]),
    )
    print(json.dumps({"run_dir": str(run_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
