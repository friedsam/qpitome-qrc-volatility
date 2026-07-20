from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.runs import begin_run
from transition_forecasting.modeling.global_stage_d_dataset import write_global_stage_d_dataset


def rewrite_portable_npz(path: Path) -> None:
    """Rewrite trusted local output so string metadata does not require pickle."""
    with np.load(path, allow_pickle=True) as archive:
        payload = {
            "X": np.asarray(archive["X"], dtype=float),
            "sample_id": np.asarray(archive["sample_id"], dtype=str),
            "label": np.asarray(archive["label"], dtype=int),
            "lead": np.asarray(archive["lead"], dtype=int),
            "episode_id": np.asarray(archive["episode_id"], dtype=str),
            "split": np.asarray(archive["split"], dtype=str),
        }
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the global Stage D matched transition dataset.")
    parser.add_argument("--representative-catalogue", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/transition_forecasting/modeling/global_stage_d_dataset"),
    )
    parser.add_argument("--run-id", type=str)
    args = parser.parse_args()

    run_dir = begin_run(args.out_dir, args, run_id=args.run_id)
    summary = write_global_stage_d_dataset(
        args.representative_catalogue,
        args.inventory,
        run_dir,
    )
    rewrite_portable_npz(run_dir / "sequence_tensors.npz")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
