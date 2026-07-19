from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.qrc.rydberg_dense import RydbergDenseConfig, evolve_sequence


def load_sequences(path: Path, channel: int, sequence_length: int, max_samples: int) -> tuple[np.ndarray, list[str], str]:
    with np.load(path, allow_pickle=True) as z:
        x = np.asarray(z["X"], dtype=float)
        valid = np.asarray(z["valid"], dtype=bool)
        sample_id = np.asarray(z["sample_id"], dtype=object)
        channel_names = [str(v) for v in np.asarray(z["channel_names"])]
    seq = x[:, -sequence_length:, channel]
    eligible = valid & np.isfinite(seq).all(axis=1)
    idx = np.flatnonzero(eligible)[:max_samples]
    seq = seq[idx]
    center = seq.mean(axis=1, keepdims=True)
    scale = seq.std(axis=1, keepdims=True)
    scale[scale < 1e-8] = 1.0
    seq = np.clip((seq - center) / scale, -4.0, 4.0)
    return seq, [str(sample_id[i]) for i in idx], channel_names[channel]


def local_pattern(sequence: np.ndarray, n_atoms: int) -> np.ndarray:
    values = np.array([sequence[-1], sequence.mean(), sequence[-5:].mean(), sequence.std()], dtype=float)
    pattern = np.resize(values, n_atoms)
    return pattern / max(float(np.max(np.abs(pattern))), 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tensor", type=Path, required=True)
    p.add_argument("--channel", type=int, default=0)
    p.add_argument("--sequence-length", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260719)
    p.add_argument("--out-root", type=Path, default=Path("results/transition_forecasting/qrc/rydberg_regime_sweep"))
    args = p.parse_args()

    sequences, sample_ids, channel_name = load_sequences(args.tensor, args.channel, args.sequence_length, args.max_samples)
    probes = (args.sequence_length // 2, args.sequence_length)
    rng = np.random.default_rng(args.seed)
    base = RydbergDenseConfig()
    rows: list[dict[str, object]] = []

    for input_scale in (0.15, 0.35, 0.60):
        for interaction in (0.25, 0.75, 1.50):
            for step_duration in (0.10, 0.20, 0.35):
                cfg = replace(base, input_scale=input_scale, nearest_interaction=interaction, step_duration=step_duration)
                ordered_all, shuffled_all, reversed_all = [], [], []
                for sequence in sequences:
                    local = local_pattern(sequence, cfg.n_atoms)
                    ordered = evolve_sequence(sequence, local, cfg, probe_steps=probes)
                    shuffled = evolve_sequence(sequence[rng.permutation(len(sequence))], local, cfg, probe_steps=probes)
                    reversed_features = evolve_sequence(sequence[::-1], local, cfg, probe_steps=probes)
                    ordered_all.append(ordered)
                    shuffled_all.append(shuffled)
                    reversed_all.append(reversed_features)
                ordered_matrix = np.vstack(ordered_all)
                shuffled_matrix = np.vstack(shuffled_all)
                reversed_matrix = np.vstack(reversed_all)
                scale = float(np.std(ordered_matrix)) + 1e-12
                order_gap = float(np.mean(np.abs(ordered_matrix - shuffled_matrix)))
                reverse_gap = float(np.mean(np.abs(ordered_matrix - reversed_matrix)))
                rows.append({
                    "input_scale": input_scale,
                    "nearest_interaction": interaction,
                    "step_duration": step_duration,
                    "ordered_feature_std": float(np.std(ordered_matrix)),
                    "ordered_vs_shuffled_mae": order_gap,
                    "ordered_vs_reversed_mae": reverse_gap,
                    "normalized_shuffled_gap": order_gap / scale,
                    "normalized_reversed_gap": reverse_gap / scale,
                    "score": (order_gap + reverse_gap) / (2.0 * scale),
                })

    frame = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(outdir / "regime_sweep.csv", index=False)
    summary = {
        "tensor": str(args.tensor),
        "channel": args.channel,
        "channel_name": channel_name,
        "samples": len(sequences),
        "sample_ids": sample_ids,
        "sequence_length": args.sequence_length,
        "probe_steps": list(probes),
        "best": frame.iloc[0].to_dict(),
        "purpose": "select an order-sensitive, non-saturated dense-simulator regime",
        "test_rows_used": 0,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(frame.head(10).to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"WROTE {outdir}")


if __name__ == "__main__":
    main()
