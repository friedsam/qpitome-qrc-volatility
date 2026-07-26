from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transition_forecasting.modeling.classical_benchmarks.common import load_rematched_dataset
from transition_forecasting.modeling.classical_benchmarks.esn import predict_fold


def run_fold_prediction(
    *,
    dataset_root: Path,
    fold: int,
    config: dict[str, object],
    alpha: float,
    seeds: tuple[int, ...],
    output: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    dataset = load_rematched_dataset(dataset_root)
    frame = predict_fold(
        dataset,
        fold=fold,
        config=config,
        alpha=alpha,
        seeds=seeds,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, compression="gzip")
    runtime = {"wall_seconds": time.perf_counter() - started, "fold": int(fold)}
    output.with_suffix(output.suffix + ".runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    return {"output": str(output), "prediction_rows": int(len(frame)), **runtime}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one frozen-fold direct-ESN and shuffled-control prediction file.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_fold_prediction(
        dataset_root=args.dataset_root,
        fold=args.fold,
        config=json.loads(args.config_json),
        alpha=args.alpha,
        seeds=tuple(args.seeds),
        output=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
