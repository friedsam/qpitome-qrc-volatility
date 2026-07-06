#!/usr/bin/env python3
"""
Rydberg/neutral-atom TFIM-like reservoir response diagnostic.

Purpose
-------
Characterize the input-response shape of a small AHS Rydberg reservoir before
feeding market/regime labels. This is a simulator-only diagnostic. It asks:

    Does a fixed neutral-atom geometry plus input-modulated detuning/Rabi drive
    produce nonlinear, blockade-shaped, nontrivial occupation features?

It is not a forecasting/regime model yet.

Example
-------
python scripts/rydberg_tfim_response.py \
  --n-atoms 6 --shots 100 --n-inputs 41 --encoding detuning \
  --out artifacts/rydberg_tfim_response.csv

Zoom the previously observed transition region:
python scripts/rydberg_tfim_response.py \
  --n-atoms 6 --shots 300 --n-inputs 31 --encoding detuning \
  --x-min -0.30 --x-max 0.15 \
  --out artifacts/rydberg_tfim_response_zoom.csv

Outputs include the artificial scan coordinate x and the physical control
coordinate delta_end_over_omega. Use delta_end_over_omega for interpretation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
from braket.ahs.atom_arrangement import AtomArrangement
from braket.ahs.driving_field import DrivingField
from braket.devices import LocalSimulator
from braket.timings.time_series import TimeSeries


@dataclass
class ResponseConfig:
    n_atoms: int
    spacing_um: float
    shots: int
    n_inputs: int
    x_min: float
    x_max: float
    encoding: str
    omega_base_rad_s: float
    omega_scale: float
    detuning_start_multiple: float
    detuning_end_base_multiple: float
    detuning_end_scale_multiple: float
    time_max_s: float
    time_ramp_s: float


def make_chain_register(n_atoms: int, spacing_um: float) -> AtomArrangement:
    spacing_m = spacing_um * 1e-6
    x0 = -0.5 * (n_atoms - 1) * spacing_m
    reg = AtomArrangement()
    for i in range(n_atoms):
        reg.add([x0 + i * spacing_m, 0.0])
    return reg


def control_values(x: float, cfg: ResponseConfig) -> tuple[float, float]:
    """Return (omega_max, delta_end_over_omega_base) for scan coordinate x."""
    omega_max = cfg.omega_base_rad_s
    delta_end_over_omega = cfg.detuning_end_base_multiple

    if cfg.encoding == "detuning":
        delta_end_over_omega = cfg.detuning_end_base_multiple + cfg.detuning_end_scale_multiple * x
    elif cfg.encoding == "rabi":
        omega_max = cfg.omega_base_rad_s * (1.0 + cfg.omega_scale * x)
        if omega_max <= 0:
            raise ValueError("omega_max became non-positive; reduce --omega-scale or scan range")
    else:
        raise ValueError(f"Unknown encoding: {cfg.encoding}")

    return omega_max, delta_end_over_omega


def make_program(x: float, cfg: ResponseConfig) -> AnalogHamiltonianSimulation:
    reg = make_chain_register(cfg.n_atoms, cfg.spacing_um)
    omega_max, delta_end_over_omega = control_values(x, cfg)

    t0 = 0.0
    tr = cfg.time_ramp_s
    tm = cfg.time_max_s
    omega = TimeSeries().put(t0, 0.0).put(tr, omega_max).put(tm - tr, omega_max).put(tm, 0.0)
    phase = TimeSeries().put(t0, 0.0).put(tm, 0.0)
    detuning = TimeSeries().put(
        t0, cfg.detuning_start_multiple * cfg.omega_base_rad_s
    ).put(
        tm, delta_end_over_omega * cfg.omega_base_rad_s
    )

    drive = DrivingField(amplitude=omega, phase=phase, detuning=detuning)
    return AnalogHamiltonianSimulation(register=reg, hamiltonian=drive)


def measurement_to_bits(measurement: Any) -> str:
    # Use 1 for Rydberg/up, 0 for ground/down, e for empty/lost sites.
    chars: list[str] = []
    for pre, post in zip(measurement.pre_sequence, measurement.post_sequence, strict=True):
        if not pre:
            chars.append("e")
        elif post:
            chars.append("1")
        else:
            chars.append("0")
    return "".join(chars)


def state_label(bits: str | None) -> str | None:
    # Prefix prevents pandas from reading bitstrings as integers and dropping leading zeros.
    return None if bits is None else f"b{bits}"


def bitstring_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    probs = np.array([v / total for v in counts.values()], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def summarize_counts(
    x: float,
    counts: Counter[str],
    n_atoms: int,
    shots_requested: int,
    omega_max_rad_s: float,
    delta_end_over_omega: float,
) -> dict[str, Any]:
    shots = sum(counts.values())
    valid_counts = {k: v for k, v in counts.items() if "e" not in k}
    valid_shots = sum(valid_counts.values())
    empty_frac = 1.0 - valid_shots / shots if shots else float("nan")

    occ_rows = []
    for bits, c in valid_counts.items():
        occ_rows.extend([[int(ch) for ch in bits]] * c)

    if occ_rows:
        occ = np.array(occ_rows, dtype=float)
        excitation_count = occ.sum(axis=1)
        mean_exc = float(excitation_count.mean())
        var_exc = float(excitation_count.var(ddof=0))
        site_means = occ.mean(axis=0)
        nearest_pair = np.array([occ[:, i] * occ[:, i + 1] for i in range(n_atoms - 1)]).T
        next_nearest_pair = (
            np.array([occ[:, i] * occ[:, i + 2] for i in range(n_atoms - 2)]).T
            if n_atoms >= 3
            else np.empty((len(occ), 0))
        )
        nn_pair_mean = float(nearest_pair.mean()) if nearest_pair.size else float("nan")
        nnn_pair_mean = float(next_nearest_pair.mean()) if next_nearest_pair.size else float("nan")
        blockade_contrast = nnn_pair_mean - nn_pair_mean
        even_occ = float(occ[:, ::2].sum(axis=1).mean())
        odd_occ = float(occ[:, 1::2].sum(axis=1).mean())
        staggered = even_occ - odd_occ
    else:
        mean_exc = var_exc = nn_pair_mean = nnn_pair_mean = blockade_contrast = staggered = float("nan")
        site_means = np.full(n_atoms, np.nan)

    top = counts.most_common(5)
    top_bits = [top[i][0] if len(top) > i else None for i in range(5)]
    top_freq = [top[i][1] / shots if len(top) > i and shots else float("nan") for i in range(5)]

    row: dict[str, Any] = {
        "x": x,
        "delta_end_over_omega": delta_end_over_omega,
        "omega_max_rad_s": omega_max_rad_s,
        "shots_requested": shots_requested,
        "shots_returned": shots,
        "valid_shots": valid_shots,
        "empty_site_fraction": empty_frac,
        "n_unique_states": len(counts),
        "entropy_bits": bitstring_entropy(counts),
        "mean_excitation": mean_exc,
        "var_excitation": var_exc,
        "nearest_pair_11_mean": nn_pair_mean,
        "next_nearest_pair_11_mean": nnn_pair_mean,
        "blockade_contrast_nnn_minus_nn": blockade_contrast,
        "staggered_even_minus_odd": staggered,
        "top1_state": state_label(top_bits[0]),
        "top1_freq": top_freq[0],
        "top2_state": state_label(top_bits[1]),
        "top2_freq": top_freq[1],
        "top3_state": state_label(top_bits[2]),
        "top3_freq": top_freq[2],
        "top4_state": state_label(top_bits[3]),
        "top4_freq": top_freq[3],
        "top5_state": state_label(top_bits[4]),
        "top5_freq": top_freq[4],
    }
    for i, val in enumerate(site_means):
        row[f"site_{i}_excitation"] = float(val)
    return row


def plot_response(df: pd.DataFrame, out_prefix: Path) -> None:
    x_col = "delta_end_over_omega" if "delta_end_over_omega" in df.columns else "x"
    x_label = "final detuning / omega_base" if x_col == "delta_end_over_omega" else "input x"
    plots = [
        ("mean_excitation", "Mean Rydberg excitations", "mean_excitation"),
        ("entropy_bits", "Bitstring entropy", "entropy_bits"),
        ("nearest_pair_11_mean", "Nearest-neighbor co-excitation", "nearest_pair_11_mean"),
        ("blockade_contrast_nnn_minus_nn", "Blockade contrast", "blockade_contrast"),
        ("top1_freq", "Dominant pattern frequency", "top1_freq"),
    ]
    for col, title, suffix in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df[x_col], df[col], marker="o")
        ax.set_xlabel(x_label)
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_prefix.with_name(f"{out_prefix.name}_{suffix}.png"), dpi=160)
        plt.close(fig)


def run(cfg: ResponseConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    sim = LocalSimulator("braket_ahs")
    xs = np.linspace(cfg.x_min, cfg.x_max, cfg.n_inputs)
    rows = []
    for idx, x in enumerate(xs, start=1):
        omega_max, delta_end_over_omega = control_values(float(x), cfg)
        print(
            f"[{idx:03d}/{len(xs):03d}] x={x:+.3f} "
            f"delta_end/omega={delta_end_over_omega:+.3f}",
            flush=True,
        )
        program = make_program(float(x), cfg)
        result = sim.run(program, shots=cfg.shots).result()
        counts = Counter(measurement_to_bits(m) for m in result.measurements)
        rows.append(
            summarize_counts(
                float(x),
                counts,
                cfg.n_atoms,
                cfg.shots,
                omega_max,
                delta_end_over_omega,
            )
        )
    df = pd.DataFrame(rows)

    metrics = {}
    dx = np.diff(df["delta_end_over_omega"].to_numpy(dtype=float))
    for col in [
        "mean_excitation",
        "entropy_bits",
        "nearest_pair_11_mean",
        "blockade_contrast_nnn_minus_nn",
        "top1_freq",
    ]:
        y = df[col].to_numpy(dtype=float)
        dy = np.diff(y)
        finite = np.isfinite(dy) & np.isfinite(dx) & (dx != 0)
        metrics[f"max_abs_slope_vs_delta_{col}"] = (
            float(np.max(np.abs(dy[finite] / dx[finite]))) if finite.any() else None
        )

    summary = {
        "config": asdict(cfg),
        "n_rows": len(df),
        "physical_scan": {
            "delta_end_over_omega_min": float(df["delta_end_over_omega"].min()),
            "delta_end_over_omega_max": float(df["delta_end_over_omega"].max()),
            "x_min": cfg.x_min,
            "x_max": cfg.x_max,
        },
        "metrics": metrics,
        "interpretation_notes": [
            "Use delta_end_over_omega, not x, for physical interpretation.",
            "Flat curves imply weak input sensitivity.",
            "Top1 frequency near 1 across all scan points implies deterministic collapse.",
            "A localized high slope in mean excitation/entropy/pair features suggests a threshold region worth testing with market-state inputs.",
            "Positive blockade_contrast_nnn_minus_nn means next-nearest co-excitation exceeds nearest-neighbor co-excitation; this is a blockade signature, not proof of entanglement.",
        ],
    }
    return df, summary


def main() -> int:
    p = argparse.ArgumentParser(description="Rydberg TFIM-like AHS response diagnostic")
    p.add_argument("--n-atoms", type=int, default=6)
    p.add_argument("--spacing-um", type=float, default=6.0)
    p.add_argument("--shots", type=int, default=100)
    p.add_argument("--n-inputs", type=int, default=41)
    p.add_argument("--x-min", type=float, default=-1.0)
    p.add_argument("--x-max", type=float, default=1.0)
    p.add_argument("--encoding", choices=["detuning", "rabi"], default="detuning")
    p.add_argument("--omega-base-rad-s", type=float, default=6.3e6)
    p.add_argument("--omega-scale", type=float, default=0.5)
    p.add_argument("--detuning-start-multiple", type=float, default=-4.0)
    p.add_argument("--detuning-end-base-multiple", type=float, default=1.0)
    p.add_argument("--detuning-end-scale-multiple", type=float, default=5.0)
    p.add_argument("--time-max-s", type=float, default=3.0e-6)
    p.add_argument("--time-ramp-s", type=float, default=0.3e-6)
    p.add_argument("--out", type=Path, default=Path("artifacts/rydberg_tfim_response.csv"))
    args = p.parse_args()

    if args.n_atoms < 2:
        raise SystemExit("--n-atoms must be >= 2")
    if args.shots <= 0:
        raise SystemExit("--shots must be > 0")
    if args.n_inputs < 3:
        raise SystemExit("--n-inputs must be >= 3")
    if args.x_min >= args.x_max:
        raise SystemExit("--x-min must be < --x-max")

    cfg = ResponseConfig(
        n_atoms=args.n_atoms,
        spacing_um=args.spacing_um,
        shots=args.shots,
        n_inputs=args.n_inputs,
        x_min=args.x_min,
        x_max=args.x_max,
        encoding=args.encoding,
        omega_base_rad_s=args.omega_base_rad_s,
        omega_scale=args.omega_scale,
        detuning_start_multiple=args.detuning_start_multiple,
        detuning_end_base_multiple=args.detuning_end_base_multiple,
        detuning_end_scale_multiple=args.detuning_end_scale_multiple,
        time_max_s=args.time_max_s,
        time_ramp_s=args.time_ramp_s,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df, summary = run(cfg)
    df.to_csv(args.out, index=False)
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_response(df, args.out.with_suffix(""))

    print(f"\nWrote {args.out}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
