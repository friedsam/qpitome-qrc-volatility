#!/usr/bin/env python3
"""Prepare hardware-friendly Task B inputs for a temporal Rydberg reservoir.

The full 13x8 causal path tensor is too wide to map directly onto one global
analog control without introducing arbitrary learned compression. This script
therefore creates three fixed, label-independent encodings:

1. return_only: signed weekly return sequence;
2. return_uncertainty: signed return plus HMM transition uncertainty;
3. return_volatility: signed return plus 13-week realized volatility.

Each channel is smoothly bounded to [-1, 1] with tanh(z / scale). The encodings
are intended for time multiplexing: one market week becomes one or two short
control segments. No target labels are used in the transformation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENCODINGS = {
    "return_only": ["return_pct"],
    "return_uncertainty": ["return_pct", "long_regime_uncertainty"],
    "return_volatility": ["return_pct", "rolling_vol_13w"],
}


def smooth_bound(values: np.ndarray, divisor: float) -> np.ndarray:
    return np.tanh(values / divisor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pathdir",
        type=Path,
        default=Path("/tmp/branch_destination_paths"),
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("/tmp/branch_destination_rydberg_inputs"),
    )
    parser.add_argument(
        "--bound-divisor",
        type=float,
        default=2.0,
        help="Apply tanh(z / divisor) to fixed pre-split standardized channels.",
    )
    args = parser.parse_args()

    paths = np.load(args.pathdir / "paths.npy")
    metadata = pd.read_csv(args.pathdir / "episode_metadata.csv")
    standardization = pd.read_csv(args.pathdir / "standardization.csv")
    channels = standardization["channel"].tolist()
    channel_to_index = {channel: index for index, channel in enumerate(channels)}

    missing = sorted({channel for group in ENCODINGS.values() for channel in group}.difference(channels))
    if missing:
        raise ValueError(f"Required channels absent from path tensor: {missing}")
    if len(paths) != len(metadata):
        raise ValueError("paths.npy and episode_metadata.csv length mismatch")

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_shape": list(paths.shape),
        "n_episodes": int(paths.shape[0]),
        "path_weeks": int(paths.shape[1]),
        "bound_transform": f"tanh(pre_split_standardized_value / {args.bound_divisor})",
        "label_independent": True,
        "intended_use": "time-multiplexed analog Rydberg controls with one segment per channel per week",
        "encodings": {},
    }

    summary_rows = []
    for name, selected_channels in ENCODINGS.items():
        indices = [channel_to_index[channel] for channel in selected_channels]
        tensor = smooth_bound(paths[:, :, indices], args.bound_divisor)
        np.save(args.outdir / f"{name}.npy", tensor)
        manifest["encodings"][name] = {
            "channels": selected_channels,
            "shape": list(tensor.shape),
            "segments_per_episode": int(tensor.shape[1] * tensor.shape[2]),
        }
        for channel_index, channel in enumerate(selected_channels):
            values = tensor[:, :, channel_index]
            summary_rows.append({
                "encoding": name,
                "channel": channel,
                "minimum": float(values.min()),
                "q01": float(np.quantile(values, 0.01)),
                "median": float(np.median(values)),
                "q99": float(np.quantile(values, 0.99)),
                "maximum": float(values.max()),
                "fraction_abs_above_0_95": float(np.mean(np.abs(values) > 0.95)),
            })

    metadata.to_csv(args.outdir / "episode_metadata.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(args.outdir / "encoding_summary.csv", index=False)
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Task B hardware-friendly Rydberg input encodings")
    print(json.dumps(manifest, indent=2))
    print("\nBounded-control diagnostics")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPromotion rule")
    print("Start with return_only and return_uncertainty. Keep return_volatility only as a challenge-aligned ablation.")


if __name__ == "__main__":
    main()
