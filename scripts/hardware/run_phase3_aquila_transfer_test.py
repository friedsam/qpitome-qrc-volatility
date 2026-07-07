#!/usr/bin/env python3
"""Run the two-window Phase 3 Aquila hardware-transfer test.

Default behavior is dry-run only: reconstruct the exact Fold-5 market windows,
build the 8-atom Aquila AHS programs, and write a reproducible manifest. No
hardware task is submitted unless --hardware is given together with the exact
confirmation token.

This is a final-state transfer test, not a full hardware recreation of the
288-feature production reservoir readout. The production reservoir collects
features after every anchor; reproducing that trajectory on hardware would
require one truncated task per anchor. Here, one task per market window measures
only the final 36 observables (8 occupations + 28 pair correlations).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qpitome_qrc.evaluation.walkforward import (
    make_purged_walkforward_folds,
    slice_fold_frames,
)
from qpitome_qrc.qrc.rydberg_reservoir import (
    AquilaConstraints,
    RydbergQRCConfig,
    _drive_from_window,
    atom_positions,
    build_rydberg_feature_matrix,
    make_level_rate_sequence_splits,
    select_rydberg_anchor_indices,
)

QBRAID_AQUILA_DEVICE_ID = "aws:quera:qpu:aquila"
TARGET = "future_rv_20d"
CONFIRM_TOKEN = "SUBMIT_2_AQUILA_TASKS"


def make_folds(n: int, *, n_folds: int, min_train: int, val_size: int, purge: int) -> list[dict]:
    """Backward-compatible wrapper around the shared Phase 3 protocol."""
    return make_purged_walkforward_folds(
        n,
        n_folds=n_folds,
        min_train=min_train,
        val_size=val_size,
        purge=purge,
    )


def config_from_spec(spec: dict[str, Any]) -> RydbergQRCConfig:
    c = spec["hardware_candidate"]
    return RydbergQRCConfig(
        geometry=c["geometry"],
        n_atoms_slow=int(c["n_atoms_slow"]),
        n_atoms_fast=int(c["n_atoms_fast"]),
        spacing_slow_um=float(c["spacing_slow_um"]),
        spacing_fast_um=float(c["spacing_fast_um"]),
        row_gap_um=float(c["row_gap_um"]),
        lookback_days=int(c["lookback_days"]),
        anchor_count=int(c["anchor_count"]),
        anchor_policy=c["anchor_policy"],
        reverse_anchors=bool(c["reverse_anchors"]),
        total_time_us=float(c["total_time_us"]),
        delta_center_rad_us=float(c["delta_center_rad_us"]),
        delta_span_rad_us=float(c["delta_span_rad_us"]),
        omega_base_rad_us=float(c["omega_base_rad_us"]),
        omega_mod_frac=float(c["omega_mod_frac"]),
        encoding=c["encoding"],
        collect_anchor_features=False,
        shots=None,
    )


def reconstruct_selected_windows(
    data_path: Path,
    specs: list[dict[str, Any]],
    config: RydbergQRCConfig,
    *,
    level_col: str,
    rate_col: str,
) -> dict[str, np.ndarray]:
    df = pd.read_csv(data_path).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    fold = make_folds(len(df), n_folds=5, min_train=2500, val_size=504, purge=60)[-1]
    split_frames = slice_fold_frames(df, fold)
    seq = make_level_rate_sequence_splits(
        split_frames,
        level_col=level_col,
        rate_col=rate_col,
        target_column=TARGET,
        lookback_days=config.lookback_days,
    )

    X_test, _, dates = seq["test"]
    date_strings = pd.to_datetime(dates).dt.strftime("%Y-%m-%d").to_numpy()
    out: dict[str, np.ndarray] = {}
    for spec in specs:
        date = spec["date"]
        matches = np.flatnonzero(date_strings == date)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one Fold-5 test window for {date}; found {len(matches)}")
        out[date] = X_test[matches[0]]
    return out


def _append_point(series: Any, t_us: float, value_rad_us: float) -> None:
    series.put(float(t_us) * 1e-6, float(value_rad_us) * 1e6)


def build_hardware_waveform(
    window: np.ndarray,
    config: RydbergQRCConfig,
    hw: AquilaConstraints,
) -> tuple[Any, dict[str, Any]]:
    """Build an Aquila AHS program with finite-slew ramps between plateaus."""
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries

    if config.encoding != "plateau":
        raise ValueError("This transfer wrapper currently supports the frozen plateau candidate only")

    anchor_idx = select_rydberg_anchor_indices(config)
    deltas, omegas = _drive_from_window(window[None, :, :], anchor_idx, config, hw)
    delta = deltas[0]
    omega = omegas[0]
    k_count = len(anchor_idx)
    hold_us = config.total_time_us / k_count

    register = AtomArrangement()
    for x_um, y_um in atom_positions(config):
        register.add([float(x_um) * 1e-6, float(y_um) * 1e-6])

    amp = TimeSeries()
    det = TimeSeries()
    phase = TimeSeries()
    timeline: list[dict[str, float | int | str]] = []
    t_us = 0.0

    _append_point(amp, t_us, 0.0)
    _append_point(det, t_us, float(delta[0]))
    _append_point(phase, t_us, 0.0)

    initial_ramp = max(float(omega[0]) / hw.omega_slew_rad_us2, 0.001)
    t_us += initial_ramp
    _append_point(amp, t_us, float(omega[0]))
    _append_point(det, t_us, float(delta[0]))
    timeline.append({"kind": "initial_ramp", "end_us": t_us, "duration_us": initial_ramp})

    for k in range(k_count):
        start_hold = t_us
        t_us += hold_us
        _append_point(amp, t_us, float(omega[k]))
        _append_point(det, t_us, float(delta[k]))
        timeline.append(
            {
                "kind": "hold",
                "anchor": k,
                "start_us": start_hold,
                "end_us": t_us,
                "duration_us": hold_us,
                "omega_rad_us": float(omega[k]),
                "delta_rad_us": float(delta[k]),
            }
        )

        if k < k_count - 1:
            ramp_us = max(
                abs(float(omega[k + 1] - omega[k])) / hw.omega_slew_rad_us2,
                abs(float(delta[k + 1] - delta[k])) / hw.delta_slew_rad_us2,
                0.001,
            )
            t_us += ramp_us
            _append_point(amp, t_us, float(omega[k + 1]))
            _append_point(det, t_us, float(delta[k + 1]))
            timeline.append(
                {
                    "kind": "inter_anchor_ramp",
                    "after_anchor": k,
                    "end_us": t_us,
                    "duration_us": ramp_us,
                }
            )

    final_ramp = max(float(omega[-1]) / hw.omega_slew_rad_us2, 0.001)
    t_us += final_ramp
    _append_point(amp, t_us, 0.0)
    _append_point(det, t_us, float(delta[-1]))
    _append_point(phase, t_us, 0.0)
    timeline.append({"kind": "final_ramp", "end_us": t_us, "duration_us": final_ramp})

    drive = DrivingField(amplitude=amp, phase=phase, detuning=det)
    program = AnalogHamiltonianSimulation(register=register, hamiltonian=drive)
    manifest = {
        "anchor_indices": anchor_idx.tolist(),
        "anchor_level_scaled": window[anchor_idx, 0].tolist(),
        "anchor_rate_scaled": window[anchor_idx, 1].tolist(),
        "delta_rad_us": delta.tolist(),
        "omega_rad_us": omega.tolist(),
        "plateau_hold_us_each": hold_us,
        "simulated_plateau_time_us": config.total_time_us,
        "hardware_waveform_total_time_us": t_us,
        "timeline": timeline,
    }
    return program, manifest


def measurement_to_state(measurement: Any) -> str:
    """Map Braket AHS pre/post site occupancy to e=empty, u=Rydberg, d=ground."""
    state = []
    for pre, post in zip(measurement.pre_sequence, measurement.post_sequence, strict=True):
        if not pre:
            state.append("e")
        elif not post:
            state.append("u")
        else:
            state.append("d")
    return "".join(state)


def summarize_braket_result(result: Any, n_atoms: int) -> dict[str, Any]:
    counts = Counter(measurement_to_state(m) for m in result.measurements)
    return summarize_state_counts(counts, n_atoms)


def _normalize_qbraid_state(raw_state: Any, n_atoms: int) -> str | None:
    """Normalize common qBraid AHS count-key formats to d/u/e strings."""
    text = str(raw_state).strip().lower()
    compact = "".join(ch for ch in text if ch.isalnum())

    if len(compact) == n_atoms and set(compact) <= {"0", "1"}:
        return "".join("u" if ch == "1" else "d" for ch in compact)
    if len(compact) == n_atoms and set(compact) <= {"g", "r", "e"}:
        return "".join({"g": "d", "r": "u", "e": "e"}[ch] for ch in compact)
    if len(compact) == n_atoms and set(compact) <= {"d", "u", "e"}:
        return compact
    return None


def summarize_qbraid_counts(raw_counts: Any, n_atoms: int) -> dict[str, Any]:
    counts_dict = dict(raw_counts)
    normalized: Counter[str] = Counter()
    unparsed: dict[str, int] = {}
    for raw_state, count in counts_dict.items():
        state = _normalize_qbraid_state(raw_state, n_atoms)
        if state is None:
            unparsed[str(raw_state)] = int(count)
        else:
            normalized[state] += int(count)

    summary = summarize_state_counts(normalized, n_atoms)
    summary["raw_measurement_counts"] = {str(k): int(v) for k, v in counts_dict.items()}
    summary["unparsed_measurement_counts"] = unparsed
    summary["binary_mapping_note"] = "For qBraid binary AHS count keys, 1 is treated as Rydberg/up and 0 as ground/down."
    return summary


def summarize_state_counts(counts: Counter[str], n_atoms: int) -> dict[str, Any]:
    shots = sum(counts.values())
    valid_counts = {state: count for state, count in counts.items() if "e" not in state}
    valid_shots = sum(valid_counts.values())

    pair_index = [(i, j) for i in range(n_atoms - 1) for j in range(i + 1, n_atoms)]
    occ = np.zeros(n_atoms, dtype=float)
    pairs = np.zeros(len(pair_index), dtype=float)
    if valid_shots:
        for state, count in valid_counts.items():
            bits = np.array([1.0 if ch == "u" else 0.0 for ch in state])
            occ += count * bits
            pairs += count * np.array([bits[i] * bits[j] for i, j in pair_index])
        occ /= valid_shots
        pairs /= valid_shots

    return {
        "shots_parsed": shots,
        "fully_loaded_shots": valid_shots,
        "fully_loaded_fraction": valid_shots / shots if shots else None,
        "n_unique_states": len(counts),
        "top_counts": counts.most_common(30),
        "postselected_final_features": np.concatenate([occ, pairs]).tolist() if valid_shots else None,
        "occupations": occ.tolist() if valid_shots else None,
        "pair_index": pair_index,
        "pair_correlations": pairs.tolist() if valid_shots else None,
    }


def exact_final_features(window: np.ndarray, config: RydbergQRCConfig) -> list[float]:
    features = build_rydberg_feature_matrix(window[None, :, :], config, verbose=False)
    return features[0].tolist()


def finite_shot_final_features(
    window: np.ndarray,
    config: RydbergQRCConfig,
    *,
    shots: int,
    seed: int,
) -> list[float]:
    noisy_cfg = RydbergQRCConfig(**{**asdict(config), "shots": shots, "shot_seed": seed})
    features = build_rydberg_feature_matrix(window[None, :, :], noisy_cfg, verbose=False)
    return features[0].tolist()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/processed/phase2_spy_vix_volatility.csv"))
    p.add_argument(
        "--specs",
        type=Path,
        default=Path("scratch/aquila_mvp/aquila_run_specs_validation2.json"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("scratch/aquila_transfer_test"))
    p.add_argument("--level-col", default="vix_rv_spread")
    p.add_argument("--rate-col", default="rv_accel_log_5_20")
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--shot-seed", type=int, default=7)
    p.add_argument("--device-check", action="store_true")
    p.add_argument("--local-sim", action="store_true")
    p.add_argument("--hardware", action="store_true")
    p.add_argument("--confirm", default="")
    args = p.parse_args()

    if args.shots <= 0 or args.shots > 1000:
        raise SystemExit("shots must be in 1..1000")
    if args.hardware and args.confirm != CONFIRM_TOKEN:
        raise SystemExit(f"Hardware submission blocked. Re-run with --confirm {CONFIRM_TOKEN}")

    specs = json.loads(args.specs.read_text(encoding="utf-8"))
    if len(specs) != 2:
        raise SystemExit(f"Expected exactly two validation specs; found {len(specs)}")

    config = config_from_spec(specs[0])
    for other in specs[1:]:
        if config_from_spec(other) != config:
            raise SystemExit("The two run specs do not share an identical hardware candidate")

    windows = reconstruct_selected_windows(
        args.data,
        specs,
        config,
        level_col=args.level_col,
        rate_col=args.rate_col,
    )

    hw = AquilaConstraints()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "two-window final-state Aquila transfer test; not a benchmark",
        "important_scope_note": (
            "One Aquila task per window measures only the final 36 observables. "
            "The production 288-feature reservoir collects after all 8 anchors and would require 8 truncated tasks per window on hardware."
        ),
        "qbraid_device_id": QBRAID_AQUILA_DEVICE_ID,
        "shots": args.shots,
        "config": asdict(config),
        "windows": [],
    }

    device = None
    if args.device_check or args.hardware:
        from qbraid.runtime import QbraidProvider

        provider = QbraidProvider()
        device = provider.get_device(QBRAID_AQUILA_DEVICE_ID)
        report["device"] = {
            "status": str(device.status()),
            "experiment_type": str(device.profile.experiment_type),
            "program_spec": str(device.profile.program_spec),
        }

    for spec in specs:
        date = spec["date"]
        window = windows[date]
        program, waveform = build_hardware_waveform(window, config, hw)
        item: dict[str, Any] = {
            "date": date,
            "selection_bucket": spec["selection_bucket"],
            "labels": {"q90": spec["q90_label"], "q95": spec["q95_label"]},
            "simulator_scores": spec["simulator_scores"],
            "waveform": waveform,
            "exact_final_features": exact_final_features(window, config),
            "bitstring_1000shot_final_features": finite_shot_final_features(
                window, config, shots=args.shots, seed=args.shot_seed
            ),
            "ahs_ir_preview": str(program.to_ir())[:12000],
        }

        if args.local_sim:
            from braket.devices import LocalSimulator

            sim = LocalSimulator("braket_ahs")
            result = sim.run(program, shots=args.shots).result()
            item["local_ahs_result"] = summarize_braket_result(result, len(atom_positions(config)))

        if args.hardware:
            assert device is not None
            job = device.run(
                program,
                shots=args.shots,
                tags={"project": "qpitome-phase3", "date": date, "test": "aquila-transfer"},
            )
            item["hardware_job_id"] = str(job.id)
            item["hardware_job_status_initial"] = str(job.status())
            checkpoint = args.out_dir / "aquila_transfer_report.json"
            report["windows"].append(item)
            checkpoint.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
            print(f"Submitted {date}: job {job.id}")

            result = job.result()
            raw_counts = result.data.measurement_counts
            item["hardware_job_status_final"] = str(job.status())
            item["hardware_result"] = summarize_qbraid_counts(raw_counts, len(atom_positions(config)))
        else:
            report["windows"].append(item)

        checkpoint = args.out_dir / "aquila_transfer_report.json"
        checkpoint.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
        print(f"Prepared {date}: waveform {waveform['hardware_waveform_total_time_us']:.6f} us")

    out = args.out_dir / "aquila_transfer_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"Wrote {out}")
    if not args.hardware:
        print("No hardware tasks submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
