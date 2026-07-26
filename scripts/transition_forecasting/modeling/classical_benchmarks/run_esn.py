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

from experiments.runs import begin_run
from transition_forecasting.modeling.classical_benchmarks.esn import (
    DEFAULT_ALPHAS,
    DEFAULT_CONFIGS,
    DEFAULT_CONFIRMATION_FOLDS,
    DEFAULT_SEEDS,
    DEFAULT_SELECTION_FOLDS,
)
from transition_forecasting.modeling.classical_benchmarks.esn_finalize import finalize_esn_run
from transition_forecasting.modeling.classical_benchmarks.esn_fold_worker import run_fold_prediction
from transition_forecasting.modeling.classical_benchmarks.esn_selection_worker import run_selection


def _selected_spec(run_dir: Path) -> tuple[dict[str, object], float]:
    path = run_dir / "selection" / "selected_spec.json"
    selected = json.loads(path.read_text(encoding="utf-8"))
    return selected["selected_config"], float(selected["selected_alpha"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the direct ESN benchmark as resumable selection, per-fold prediction, and finalization phases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="Select one fixed ESN configuration and alpha on development folds.")
    select.add_argument("--dataset-root", type=Path, required=True)
    select.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/classical_benchmarks/esn"),
    )
    select.add_argument("--run-id")
    select.add_argument("--selection-folds", type=int, nargs="+", default=list(DEFAULT_SELECTION_FOLDS))
    select.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    select.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    select.add_argument("--configs-json", default=json.dumps(list(DEFAULT_CONFIGS), sort_keys=True))
    predict = subparsers.add_parser("predict-fold", help="Generate one fold using the frozen selected ESN specification.")
    predict.add_argument("--dataset-root", type=Path, required=True)
    predict.add_argument("--run-dir", type=Path, required=True)
    predict.add_argument("--fold", type=int, required=True)
    predict.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    finalize = subparsers.add_parser("finalize", help="Combine completed fold predictions into benchmark tables.")
    finalize.add_argument("--dataset-root", type=Path, required=True)
    finalize.add_argument("--run-dir", type=Path, required=True)
    finalize.add_argument("--selection-folds", type=int, nargs="+", default=list(DEFAULT_SELECTION_FOLDS))
    finalize.add_argument("--confirmation-folds", type=int, nargs="+", default=list(DEFAULT_CONFIRMATION_FOLDS))
    finalize.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.command == "select":
        configs = tuple(json.loads(args.configs_json))
        params = {
            "benchmark": "direct_esn",
            "dataset_root": args.dataset_root,
            "selection_folds": tuple(args.selection_folds),
            "confirmation_folds": DEFAULT_CONFIRMATION_FOLDS,
            "seeds": tuple(args.seeds),
            "alphas": tuple(args.alphas),
            "configs": configs,
            "representation": "level_diff_time",
            "pooling": "final_mean_std",
            "washout": 10,
            "target": "direct future log-volatility path",
            "test_evaluated": False,
        }
        run_dir = begin_run(args.results_root, params, run_id=args.run_id)
        selected = run_selection(
            dataset_root=args.dataset_root,
            folds=tuple(args.selection_folds),
            configs=configs,
            seeds=tuple(args.seeds),
            alphas=tuple(args.alphas),
            output_dir=run_dir / "selection",
        )
        result = {"run_dir": str(run_dir), **selected}
    elif args.command == "predict-fold":
        config, alpha = _selected_spec(args.run_dir)
        result = run_fold_prediction(
            dataset_root=args.dataset_root,
            fold=args.fold,
            config=config,
            alpha=alpha,
            seeds=tuple(args.seeds),
            output=args.run_dir / "fold_predictions" / f"fold_{args.fold}.csv.gz",
        )
    else:
        result = finalize_esn_run(
            run_dir=args.run_dir,
            dataset_root=args.dataset_root,
            selection_folds=tuple(args.selection_folds),
            confirmation_folds=tuple(args.confirmation_folds),
            seeds=tuple(args.seeds),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
