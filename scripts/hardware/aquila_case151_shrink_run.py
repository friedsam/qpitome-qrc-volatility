#!/usr/bin/env python3
"""
Prepare, monitor, and collect the three-probe Aquila observable
attenuation study for case 151.

This uses the hardware-native schedule already accepted by Aquila:
- 6 atoms on a 3 x 2 rectangular register
- one simultaneous two-channel control target per lag
- probe endpoints at 10, 20, and 40 steps
- 1.00, 1.95, and 3.85 microseconds total duration

Modes
-----
Prepare simulator reference only:
  python scripts/hardware/aquila_case151_shrink_run.py prepare

Check job status:
  python scripts/hardware/aquila_case151_shrink_run.py status

Collect completed jobs and create the attenuation graph:
  python scripts/hardware/aquila_case151_shrink_run.py collect
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import itertools
import json
from pathlib import Path
import runpy
from typing import Any

import numpy as np


PROBES = (10, 20, 40)
POSITIONS_UM = np.asarray(
    [
        [-8.2,  4.4],
        [ 0.0,  4.4],
        [ 8.2,  4.4],
        [-8.2, -4.4],
        [ 0.0, -4.4],
        [ 8.2, -4.4],
    ],
    dtype=float,
)

DEFAULT_CASE = Path(
    "results/transition_forecasting/hardware/"
    "aquila_case151/freeze_001/case151_freeze.npz"
)
DEFAULT_PREFLIGHT_SCRIPT = Path(
    "scripts/hardware/aquila_case151_support.py"
)
DEFAULT_OUTDIR = Path(
    "results/transition_forecasting/hardware/"
    "aquila_case151/shrink_001"
)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return jsonable(value.to_dict())
        except Exception:
            pass
    return repr(value)


def load_support(preflight_script: Path):
    if not preflight_script.is_file():
        raise FileNotFoundError(f"missing support script: {preflight_script}")
    return runpy.run_path(str(preflight_script))


def validate_geometry(positions_um: np.ndarray) -> None:
    for i, j in itertools.combinations(range(len(positions_um)), 2):
        dx = abs(float(positions_um[i, 0] - positions_um[j, 0]))
        dy = abs(float(positions_um[i, 1] - positions_um[j, 1]))
        radial = float(np.linalg.norm(positions_um[i] - positions_um[j]))
        if radial < 4.0 - 1e-12:
            raise RuntimeError(f"sites {i},{j}: radial separation {radial} um < 4 um")
        if 0.0 < dx < 4.0 - 1e-12:
            raise RuntimeError(f"sites {i},{j}: x separation {dx} um violates grid rule")
        if 0.0 < dy < 4.0 - 1e-12:
            raise RuntimeError(f"sites {i},{j}: y separation {dy} um violates grid rule")


def build_native_series(
    encoded: np.ndarray,
    probe_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the exact hardware-native schedule already accepted at probe 40.

    Time grid:
      t=0: amplitude zero, detuning held at first encoded value
      t=50 ns: first encoded target
      subsequent encoded targets every 95 ns
      final 95 ns ramp to amplitude zero
    """
    if probe_steps not in PROBES:
        raise ValueError(f"probe_steps must be one of {PROBES}")

    times_ns = [0]
    omega = [0.0]
    delta_targets: list[float] = []

    for step in range(probe_steps):
        omega_value = 6.0e6 * (1.0 + 0.6 * float(encoded[step, 1]))
        delta_value = 6.0e6 + 4.0e6 * float(encoded[step, 0])

        omega_value = float(int(round(omega_value / 400.0)) * 400)
        delta_value = float(round(delta_value / 0.2) * 0.2)

        times_ns.append(50 + 95 * step)
        omega.append(omega_value)
        delta_targets.append(delta_value)

    delta = [delta_targets[0], *delta_targets]

    times_ns.append(50 + 95 * probe_steps)
    omega.append(0.0)
    delta.append(delta[-1])

    times_s = np.asarray(times_ns, dtype=float) * 1e-9
    omega = np.asarray(omega, dtype=float)
    delta = np.asarray(delta, dtype=float)
    phase = np.zeros(len(times_s), dtype=float)

    if np.any(np.diff(times_s) < 50e-9 - 1e-15):
        raise RuntimeError("time spacing fell below 50 ns")
    if times_s[-1] > 4.0e-6 + 1e-15:
        raise RuntimeError("program duration exceeds 4 microseconds")

    return times_s, omega, delta, phase


def build_program(
    positions_um: np.ndarray,
    times_s: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
):
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries

    register = AtomArrangement()
    for x_um, y_um in positions_um:
        register.add(
            (
                Decimal(str(x_um)) * Decimal("1e-6"),
                Decimal(str(y_um)) * Decimal("1e-6"),
            )
        )

    omega_ts = TimeSeries()
    delta_ts = TimeSeries()
    phase_ts = TimeSeries()

    for time_s, value in zip(times_s, omega):
        omega_ts.put(Decimal(str(time_s)), Decimal(str(value)))
    for time_s, value in zip(times_s, delta):
        delta_ts.put(Decimal(str(time_s)), Decimal(str(value)))

    phase_ts.put(Decimal("0"), Decimal("0"))
    phase_ts.put(Decimal(str(times_s[-1])), Decimal("0"))

    return AnalogHamiltonianSimulation(
        register=register,
        hamiltonian=DrivingField(
            amplitude=omega_ts,
            phase=phase_ts,
            detuning=delta_ts,
        ),
    )


def observable_names() -> tuple[str, ...]:
    occupations = tuple(f"n_{site}" for site in range(6))
    pairs = tuple(
        f"n_{left}_n_{right}"
        for left, right in itertools.combinations(range(6), 2)
    )
    return occupations + pairs


def simulator_reference(
    support: dict[str, Any],
    case: dict[str, np.ndarray],
    outdir: Path,
) -> dict[int, dict[str, np.ndarray]]:
    names = observable_names()
    bits = support["occupation_bits"](6)
    pairs = tuple(itertools.combinations(range(6), 2))
    pair_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in pairs],
        axis=1,
    )

    reference: dict[int, dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    for probe in PROBES:
        times, omega, delta, _ = build_native_series(case["encoded_sequence"], probe)
        probability = support["simulate_linear_program"](
            POSITIONS_UM,
            times,
            omega,
            delta,
            dt_max_us=0.001,
        )
        occupations = probability @ bits
        pair_expectations = probability @ pair_bits
        values = np.concatenate([occupations, pair_expectations])

        reference[probe] = {
            "times_s": times,
            "omega_rad_s": omega,
            "delta_rad_s": delta,
            "probability": probability,
            "observables": values,
        }

        for index, (name, value) in enumerate(zip(names, values)):
            rows.append(
                {
                    "probe_steps": probe,
                    "observable_index": index,
                    "observable": name,
                    "kind": "occupation" if index < 6 else "pair",
                    "simulator_expectation": float(value),
                }
            )

    import csv

    with (outdir / "simulator_reference_observables.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        outdir / "simulator_reference_probabilities.npz",
        **{
            f"probe_{probe}_probability": data["probability"]
            for probe, data in reference.items()
        },
        **{
            f"probe_{probe}_times_s": data["times_s"]
            for probe, data in reference.items()
        },
        **{
            f"probe_{probe}_omega_rad_s": data["omega_rad_s"]
            for probe, data in reference.items()
        },
        **{
            f"probe_{probe}_delta_rad_s": data["delta_rad_s"]
            for probe, data in reference.items()
        },
    )

    summary = {
        str(probe): {
            "total_time_us": float(data["times_s"][-1] * 1e6),
            "occupation_min": float(data["observables"][:6].min()),
            "occupation_max": float(data["observables"][:6].max()),
            "pair_min": float(data["observables"][6:].min()),
            "pair_max": float(data["observables"][6:].max()),
        }
        for probe, data in reference.items()
    }
    (outdir / "simulator_reference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("SIMULATOR REFERENCE")
    for probe in PROBES:
        item = summary[str(probe)]
        print(
            f"probe={probe:2d} time={item['total_time_us']:.2f} us "
            f"occupations=[{item['occupation_min']:.4f}, "
            f"{item['occupation_max']:.4f}] "
            f"pairs=[{item['pair_min']:.4f}, {item['pair_max']:.4f}]"
        )
    print("Simulator signal is non-saturated and suitable for comparison.")

    return reference


def load_job_rows(outdir: Path) -> list[dict[str, Any]]:
    path = outdir / "submitted_jobs.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing submission record: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("submitted_jobs.json must contain exactly three jobs")
    return rows


def connect_job(job_id: str):
    from qbraid.runtime.native import QbraidProvider, QbraidJob

    device = QbraidProvider().get_device("aws:quera:qpu:aquila")
    return QbraidJob(job_id=job_id, device=device)


def print_status(outdir: Path) -> None:
    rows = load_job_rows(outdir)
    all_completed = True
    for row in rows:
        job = connect_job(row["job_id"])
        status = job.status()
        print(
            f"probe={int(row['probe_steps']):2d} "
            f"job={row['job_id']} status={status}"
        )
        if getattr(status, "name", str(status)) != "COMPLETED":
            all_completed = False
            if getattr(status, "name", str(status)) in {"FAILED", "CANCELLED"}:
                print("  metadata:", jsonable(job.metadata()))
    print("ALL_COMPLETED:", all_completed)


def measurement_arrays(measurements: list[Any]) -> tuple[np.ndarray, dict[str, int]]:
    valid: list[np.ndarray] = []
    counts = {
        "total": len(measurements),
        "successful": 0,
        "full_pre_sequence": 0,
        "used": 0,
    }

    for measurement in measurements:
        success = bool(getattr(measurement, "success", False))
        if not success:
            continue
        counts["successful"] += 1

        pre = np.asarray(getattr(measurement, "pre_sequence", []), dtype=int)
        post = np.asarray(getattr(measurement, "post_sequence", []), dtype=int)
        if pre.shape != (6,) or post.shape != (6,):
            continue
        if not np.all(pre == 1):
            continue
        counts["full_pre_sequence"] += 1

        valid.append(1 - post)
        counts["used"] += 1

    if not valid:
        raise RuntimeError("no usable measurements after successful/full-pre filtering")

    return np.stack(valid, axis=0).astype(float), counts


def fit_through_origin(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.dot(x, x))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def collect_results(
    support: dict[str, Any],
    case: dict[str, np.ndarray],
    outdir: Path,
) -> None:
    reference = simulator_reference(support, case, outdir)
    job_rows = load_job_rows(outdir)
    job_by_probe = {int(row["probe_steps"]): row for row in job_rows}

    names = observable_names()
    pairs = tuple(itertools.combinations(range(6), 2))
    result_rows: list[dict[str, Any]] = []
    raw_archive: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for probe in PROBES:
        row = job_by_probe[probe]
        job = connect_job(row["job_id"])
        status = job.status()
        if getattr(status, "name", str(status)) != "COMPLETED":
            raise RuntimeError(f"probe {probe} is not completed: {status}")

        result = job.result()
        measurements = result.data.measurements or []
        rydberg, shot_counts = measurement_arrays(measurements)

        occupations = rydberg.mean(axis=0)
        pair_values = np.asarray(
            [
                np.mean(rydberg[:, left] * rydberg[:, right])
                for left, right in pairs
            ],
            dtype=float,
        )
        hardware = np.concatenate([occupations, pair_values])
        simulator = reference[probe]["observables"]

        slope = fit_through_origin(simulator, hardware)
        correlation = (
            float(np.corrcoef(simulator, hardware)[0, 1])
            if np.std(simulator) > 0 and np.std(hardware) > 0
            else float("nan")
        )
        rmse_value = float(np.sqrt(np.mean((hardware - simulator) ** 2)))
        mae_value = float(np.mean(np.abs(hardware - simulator)))

        metadata = job.metadata()
        summary = {
            "probe_steps": probe,
            "job_id": row["job_id"],
            "requested_shots": int(row["shots"]),
            "total_measurements": shot_counts["total"],
            "used_measurements": shot_counts["used"],
            "used_fraction": shot_counts["used"] / max(1, shot_counts["total"]),
            "attenuation_through_origin": slope,
            "pearson_correlation": correlation,
            "rmse": rmse_value,
            "mae": mae_value,
            "reported_cost": metadata.get("cost"),
        }
        summary_rows.append(summary)

        for index, name in enumerate(names):
            result_rows.append(
                {
                    "probe_steps": probe,
                    "observable_index": index,
                    "observable": name,
                    "kind": "occupation" if index < 6 else "pair",
                    "simulator_expectation": float(simulator[index]),
                    "hardware_expectation": float(hardware[index]),
                    "difference": float(hardware[index] - simulator[index]),
                    "absolute_difference": float(abs(hardware[index] - simulator[index])),
                }
            )

        raw_archive[f"probe_{probe}_rydberg_bits"] = rydberg
        raw_archive[f"probe_{probe}_simulator_observables"] = simulator
        raw_archive[f"probe_{probe}_hardware_observables"] = hardware

        raw_json = {
            "probe_steps": probe,
            "job_id": row["job_id"],
            "shot_filter_counts": shot_counts,
            "job_metadata": jsonable(metadata),
            "measurements": [
                {
                    "success": bool(getattr(m, "success", False)),
                    "pre_sequence": jsonable(getattr(m, "pre_sequence", None)),
                    "post_sequence": jsonable(getattr(m, "post_sequence", None)),
                }
                for m in measurements
            ],
        }
        (outdir / f"probe_{probe}_raw_result.json").write_text(
            json.dumps(raw_json, indent=2) + "\n",
            encoding="utf-8",
        )

    import csv

    with (outdir / "hardware_vs_simulator_observables.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)

    with (outdir / "shrink_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    np.savez_compressed(outdir / "hardware_measurements.npz", **raw_archive)

    all_simulator = np.asarray(
        [row["simulator_expectation"] for row in result_rows], dtype=float
    )
    all_hardware = np.asarray(
        [row["hardware_expectation"] for row in result_rows], dtype=float
    )
    pooled_slope = fit_through_origin(all_simulator, all_hardware)
    pooled_corr = float(np.corrcoef(all_simulator, all_hardware)[0, 1])
    pooled_rmse = float(
        np.sqrt(np.mean((all_hardware - all_simulator) ** 2))
    )

    pooled = {
        "attenuation_through_origin": pooled_slope,
        "pearson_correlation": pooled_corr,
        "rmse": pooled_rmse,
        "n_observables": len(result_rows),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (outdir / "pooled_shrink_metrics.json").write_text(
        json.dumps(pooled, indent=2) + "\n",
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for probe in PROBES:
        subset = [row for row in result_rows if row["probe_steps"] == probe]
        x = [row["simulator_expectation"] for row in subset]
        y = [row["hardware_expectation"] for row in subset]
        ax.scatter(x, y, label=f"{probe}-step probe", alpha=0.8)

    upper = max(float(all_simulator.max()), float(all_hardware.max())) * 1.08
    grid = np.linspace(0.0, upper, 200)
    ax.plot(grid, grid, linestyle="--", linewidth=1.2, label="Ideal agreement")
    ax.plot(
        grid,
        pooled_slope * grid,
        linewidth=1.5,
        label=f"Through-origin fit: {pooled_slope:.3f}",
    )
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel("Exact simulator expectation")
    ax.set_ylabel("Aquila expectation")
    ax.set_title("Aquila observable attenuation: case 151")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "aquila_observable_shrink.png", dpi=220)
    fig.savefig(outdir / "aquila_observable_shrink.pdf")
    plt.close(fig)

    print("COLLECTION COMPLETE")
    for summary in summary_rows:
        print(
            f"probe={summary['probe_steps']:2d} "
            f"used={summary['used_measurements']}/"
            f"{summary['total_measurements']} "
            f"attenuation={summary['attenuation_through_origin']:.4f} "
            f"corr={summary['pearson_correlation']:.4f} "
            f"rmse={summary['rmse']:.4f}"
        )
    print(
        f"POOLED attenuation={pooled_slope:.4f} "
        f"corr={pooled_corr:.4f} rmse={pooled_rmse:.4f}"
    )
    print("Graph:", outdir / "aquila_observable_shrink.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "status", "collect"))
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument(
        "--preflight-script",
        type=Path,
        default=DEFAULT_PREFLIGHT_SCRIPT,
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    validate_geometry(POSITIONS_UM)

    support = load_support(args.preflight_script)
    case = support["load_case"](args.case)

    if args.mode == "prepare":
        simulator_reference(support, case, args.outdir)
    elif args.mode == "status":
        print_status(args.outdir)
    elif args.mode == "collect":
        collect_results(support, case, args.outdir)


if __name__ == "__main__":
    main()
