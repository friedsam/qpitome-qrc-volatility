from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transition_forecasting.modeling.classical_benchmarks.common import load_rematched_dataset
from transition_forecasting.modeling.classical_benchmarks.esn import select_esn_spec


def run_selection(
    *,
    dataset_root: Path,
    folds: tuple[int, ...],
    configs: tuple[dict[str, object], ...],
    seeds: tuple[int, ...],
    alphas: tuple[float, ...],
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_rematched_dataset(dataset_root)
    config, alpha, tuning, seed_metrics, summary = select_esn_spec(
        dataset,
        folds=folds,
        configs=configs,
        seeds=seeds,
        alphas=alphas,
    )
    tuning.to_csv(output_dir / "esn_tuning_by_fold.csv", index=False)
    seed_metrics.to_csv(output_dir / "esn_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "esn_tuning_summary.csv", index=False)
    selected = {"selected_config": config, "selected_alpha": float(alpha)}
    (output_dir / "selected_spec.json").write_text(
        json.dumps(selected, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "runtime.json").write_text(
        json.dumps({"wall_seconds": time.perf_counter() - started}, indent=2) + "\n",
        encoding="utf-8",
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Select one fixed direct-ESN specification on development folds.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    parser.add_argument("--configs-json", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selected = run_selection(
        dataset_root=args.dataset_root,
        folds=tuple(args.folds),
        configs=tuple(json.loads(args.configs_json)),
        seeds=tuple(args.seeds),
        alphas=tuple(args.alphas),
        output_dir=args.output_dir,
    )
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
