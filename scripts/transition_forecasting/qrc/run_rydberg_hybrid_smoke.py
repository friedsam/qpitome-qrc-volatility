from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.qrc.rydberg_dense import RydbergDenseConfig, evolve_sequence


def _safe_array(bundle: np.lib.npyio.NpzFile, name: str) -> np.ndarray | None:
    try:
        array = np.asarray(bundle[name])
    except ValueError as exc:
        if "Object arrays cannot be loaded" in str(exc):
            return None
        raise
    if array.dtype == object:
        return None
    return array


def _choose_sequence_array(bundle: np.lib.npyio.NpzFile, key: str | None) -> tuple[str, np.ndarray]:
    if key is not None:
        array = _safe_array(bundle, key)
        if array is None:
            raise ValueError(f"{key} is an object array and cannot be used as a numeric sequence tensor")
        if array.ndim < 2:
            raise ValueError(f"{key} must be at least 2-D; got {array.shape}")
        return key, array

    candidates: list[tuple[str, np.ndarray]] = []
    for name in bundle.files:
        array = _safe_array(bundle, name)
        if array is None:
            continue
        if array.ndim in (2, 3) and array.shape[0] > 1 and array.shape[1] >= 5:
            candidates.append((name, array))

    if not candidates:
        inventory = {}
        for name in bundle.files:
            array = _safe_array(bundle, name)
            inventory[name] = None if array is None else list(array.shape)
        raise ValueError(f"No numeric sequence-like NPZ array found. Arrays: {inventory}")

    candidates.sort(key=lambda item: (item[1].ndim != 3, -item[1].shape[1]))
    return candidates[0]


def _to_sequences(array: np.ndarray, channel: int) -> np.ndarray:
    if array.ndim == 2:
        return np.asarray(array, dtype=float)
    if channel < 0 or channel >= array.shape[2]:
        raise ValueError(f"channel {channel} outside array shape {array.shape}")
    return np.asarray(array[:, :, channel], dtype=float)


def _standardize_rows(sequences: np.ndarray) -> np.ndarray:
    center = np.nanmean(sequences, axis=1, keepdims=True)
    scale = np.nanstd(sequences, axis=1, keepdims=True)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    result = (sequences - center) / scale
    if not np.isfinite(result).all():
        raise ValueError("non-finite standardized sequence values")
    return np.clip(result, -4.0, 4.0)


def _local_pattern(sequence: np.ndarray, n_atoms: int) -> np.ndarray:
    context = np.array(
        [
            sequence[-1],
            sequence.mean(),
            sequence[-min(5, len(sequence)):].mean(),
            sequence.std(),
        ],
        dtype=float,
    )
    pattern = np.resize(context, n_atoms)
    norm = max(float(np.max(np.abs(pattern))), 1.0)
    return pattern / norm


def main() -> None:
    parser = argparse.ArgumentParser(description="First hybrid spatial-temporal Rydberg QRC mechanism assay.")
    parser.add_argument("--tensor", type=Path, required=True)
    parser.add_argument("--array-key", default=None)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--out-root", type=Path, default=Path("results/transition_forecasting/qrc/rydberg_hybrid_smoke"))
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=False)

    # This repository-generated NPZ contains metadata stored as object arrays.
    # Loading with pickle enabled is acceptable only for this trusted local artifact.
    with np.load(args.tensor, allow_pickle=True) as bundle:
        array_key, raw = _choose_sequence_array(bundle, args.array_key)
        inventory = {}
        for name in bundle.files:
            array = np.asarray(bundle[name])
            inventory[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "object_array": bool(array.dtype == object),
            }

    sequences = _to_sequences(raw, args.channel)
    if sequences.shape[1] < args.sequence_length:
        raise ValueError(f"requested {args.sequence_length} steps but tensor has {sequences.shape[1]}")
    sequences = _standardize_rows(sequences[:, -args.sequence_length:])
    n_samples = min(args.max_samples, len(sequences))
    sequences = sequences[:n_samples]

    config = RydbergDenseConfig()
    probes = tuple(sorted(set((max(1, args.sequence_length // 2), args.sequence_length))))
    rng = np.random.default_rng(args.seed)
    records: list[dict[str, object]] = []

    for sample_index, sequence in enumerate(sequences):
        local = _local_pattern(sequence, config.n_atoms)
        shuffled = sequence[rng.permutation(len(sequence))]
        reversed_sequence = sequence[::-1]
        conditions = {
            "ordered": (sequence, True, False),
            "shuffled": (shuffled, True, False),
            "reversed": (reversed_sequence, True, False),
            "reset": (sequence, True, True),
            "interaction_off": (sequence, False, False),
        }
        for condition, (values, interactions, reset_each_step) in conditions.items():
            features = evolve_sequence(
                values,
                local,
                config,
                probe_steps=probes,
                interactions=interactions,
                reset_each_step=reset_each_step,
            )
            for feature_index, value in enumerate(features):
                records.append(
                    {
                        "sample_index": sample_index,
                        "condition": condition,
                        "feature_index": feature_index,
                        "value": float(value),
                    }
                )

    frame = pd.DataFrame(records)
    wide = frame.pivot(index=["sample_index", "feature_index"], columns="condition", values="value").reset_index()
    for condition in ("shuffled", "reversed", "reset", "interaction_off"):
        wide[f"abs_ordered_minus_{condition}"] = np.abs(wide["ordered"] - wide[condition])

    summary = {
        "tensor": str(args.tensor),
        "npz_inventory": inventory,
        "selected_array_key": array_key,
        "selected_array_shape": list(raw.shape),
        "channel": args.channel,
        "samples": n_samples,
        "sequence_length": args.sequence_length,
        "probe_steps": list(probes),
        "config": config.__dict__,
        "mean_absolute_differences": {
            condition: float(wide[f"abs_ordered_minus_{condition}"].mean())
            for condition in ("shuffled", "reversed", "reset", "interaction_off")
        },
        "purpose": "mechanism smoke test only; no forecasting claim",
        "test_rows_used": 0,
    }
    frame.to_csv(outdir / "features_long.csv.gz", index=False)
    wide.to_csv(outdir / "paired_feature_differences.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"WROTE {outdir}")


if __name__ == "__main__":
    main()
