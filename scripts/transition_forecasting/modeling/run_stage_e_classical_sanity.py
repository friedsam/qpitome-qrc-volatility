from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    DEFAULT_ALPHAS,
    load_stage_d_run,
    run_classical_sanity_ladder,
    validate_split_integrity,
)


def _latest_stage_d_run(root: Path) -> Path:
    candidates = sorted(
        path for path in root.iterdir()
        if path.is_dir()
        and (path / "sample_manifest.csv").exists()
        and (path / "sequence_tensors.npz").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"no complete Stage D run found under {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage E persistence, HAR, and linear sequence sanity baselines."
    )
    parser.add_argument(
        "--stage-d-root",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_d_dataset"),
    )
    parser.add_argument("--stage-d-run", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/stage_e_classical_sanity"),
    )
    parser.add_argument("--run-id", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )
    args = parser.parse_args()

    stage_d_run = args.stage_d_run or _latest_stage_d_run(args.stage_d_root)
    resolved = vars(args).copy()
    resolved["stage_d_run"] = stage_d_run
    run_dir = begin_run(args.out_dir, resolved, run_id=args.run_id)

    data = load_stage_d_run(stage_d_run)
    split_integrity = validate_split_integrity(data.manifest)
    metrics, predictions, tuning, summary = run_classical_sanity_ladder(
        data,
        alphas=tuple(args.alphas),
        seed=args.seed,
    )

    metrics.to_csv(run_dir / "validation_metrics.csv", index=False)
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    tuning.to_csv(run_dir / "ridge_tuning.csv", index=False)
    (run_dir / "model_selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "split_integrity.json").write_text(json.dumps(split_integrity, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
