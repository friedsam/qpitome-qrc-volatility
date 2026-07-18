from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import (
    DEFAULT_ALPHAS,
    StageEData,
    TARGET_COLUMNS,
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


def _load_stage_d_run(run_dir: Path) -> StageEData:
    manifest = pd.read_csv(run_dir / "sample_manifest.csv")
    # Stage D currently stores sample_id with object dtype. These artifacts are
    # generated locally by this project, so pickle-enabled loading is required
    # for backward compatibility until the writer is migrated to Unicode dtype.
    with np.load(run_dir / "sequence_tensors.npz", allow_pickle=True) as tensors:
        sequences = np.asarray(tensors["X"], dtype=float)
        tensor_ids = tensors["sample_id"].astype(str)
    manifest_ids = manifest["sample_id"].astype(str).to_numpy()
    if sequences.ndim != 3 or sequences.shape[1:] != (40, 1):
        raise ValueError(f"expected sequence tensor shape (n, 40, 1), got {sequences.shape}")
    if len(manifest) != len(sequences) or not np.array_equal(manifest_ids, tensor_ids):
        raise ValueError("sample manifest and tensor sample IDs are not aligned")
    required = {"sample_id", "label", "lead", "episode_id", "split", *TARGET_COLUMNS}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"Stage D manifest is missing columns: {sorted(missing)}")
    return StageEData(manifest=manifest, sequences=sequences)


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

    data = _load_stage_d_run(stage_d_run)
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
