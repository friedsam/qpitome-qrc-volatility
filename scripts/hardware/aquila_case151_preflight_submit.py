#!/usr/bin/env python3
"""
Preflight and optionally submit the frozen P_GE151_^MERV_data_L5 case to Aquila.

Default behavior is DRY RUN:
  1. Load the independently reproduced frozen case.
  2. Connect to Aquila and save live metadata/capabilities.
  3. Build three destructive-probe programs (10, 20, 40 lags).
  4. Try Braket discretization for requested stretch factors.
  5. Re-simulate the hardware-translated piecewise-linear waveform with the
     frozen six-atom exact model and evaluate the frozen readout.
  6. Refuse submission unless an explicitly selected factor passes both device
     discretization and the simulator-space survival gate.

Submission is explicit:
  python aquila_case151_preflight_submit.py --case case151_freeze.npz \
      --stretch 1.0 --shots 1000 --submit

No local detuning is used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np


PROBES = (10, 20, 40)
SCHEDULE = (("A", 0.25), ("B", 0.50), ("A", 0.25))


def qlike(y: np.ndarray, p: np.ndarray) -> float:
    e = np.asarray(y, float) - np.asarray(p, float)
    return float(np.mean(np.exp(2.0 * e) - 2.0 * e - 1.0))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def load_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as b:
        return {name: np.asarray(b[name]) for name in b.files}


def branch_targets(encoded: np.ndarray, probe_steps: int) -> tuple[np.ndarray, np.ndarray]:
    targets: list[tuple[float, float]] = []
    durations: list[float] = []
    for step in range(probe_steps):
        for branch, fraction in SCHEDULE:
            if branch == "A":
                detuning_channel, amplitude_channel = 0, 1
            else:
                detuning_channel, amplitude_channel = 1, 0
            omega_rad_us = 6.0 * (1.0 + 0.6 * encoded[step, amplitude_channel])
            delta_rad_us = 6.0 + 4.0 * encoded[step, detuning_channel]
            targets.append((omega_rad_us * 1e6, delta_rad_us * 1e6))
            durations.append(0.02 * fraction * 1e-6)
    return np.asarray(targets, float), np.asarray(durations, float)


def build_boundary_linear_series(
    encoded: np.ndarray,
    probe_steps: int,
    stretch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Hardware translation with one linearly interpolated control target per A/B/A segment.

    This is not silently treated as identical to the piecewise-constant research model.
    The exact six-atom simulator below evaluates this translated waveform before submit.
    """
    targets, nominal_durations = branch_targets(encoded, probe_steps)
    durations = nominal_durations * float(stretch)
    times = [0.0]
    omega = [0.0]
    delta = [float(targets[0, 1])]
    t = 0.0
    for (target_omega, target_delta), duration in zip(targets, durations):
        t += float(duration)
        times.append(t)
        omega.append(float(target_omega))
        delta.append(float(target_delta))
    # Hardware amplitude must end at zero. Keep final detuning fixed during ramp-down.
    t += float(0.005e-6 * stretch)
    times.append(t)
    omega.append(0.0)
    delta.append(delta[-1])
    return (
        np.asarray(times, float),
        np.asarray(omega, float),
        np.asarray(delta, float),
        np.zeros(len(times), float),
    )


def build_braket_program(
    positions_um: np.ndarray,
    times: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    phase: np.ndarray,
):
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries

    register = AtomArrangement()
    for x_um, y_um in positions_um:
        register.add((float(x_um) * 1e-6, float(y_um) * 1e-6))

    omega_ts = TimeSeries()
    delta_ts = TimeSeries()
    phase_ts = TimeSeries()
    for t, value in zip(times, omega):
        omega_ts.put(float(t), float(value))
    for t, value in zip(times, delta):
        delta_ts.put(float(t), float(value))
    # Phase is constant and only needs endpoints.
    phase_ts.put(float(times[0]), float(phase[0]))
    phase_ts.put(float(times[-1]), float(phase[-1]))

    drive = DrivingField(amplitude=omega_ts, phase=phase_ts, detuning=delta_ts)
    return AnalogHamiltonianSimulation(register=register, hamiltonian=drive)


def find_aws_device(wrapper: Any) -> Any | None:
    """Find the wrapped Braket AwsDevice without relying on one qBraid SDK version."""
    candidates = [
        wrapper,
        getattr(wrapper, "_device", None),
        getattr(wrapper, "_aws_device", None),
        getattr(wrapper, "device", None),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if hasattr(candidate, "properties") and hasattr(candidate, "arn"):
            return candidate
    return None


def connect_aquila():
    errors: list[str] = []
    try:
        from qbraid.runtime import QbraidProvider
        provider = QbraidProvider()
        for identifier in (
            "aws:quera:qpu:aquila",
            "arn:aws:braket:us-east-1::device/qpu/quera/Aquila",
        ):
            try:
                device = provider.get_device(identifier)
                return device
            except Exception as exc:
                errors.append(f"QbraidProvider {identifier}: {exc!r}")
    except Exception as exc:
        errors.append(f"QbraidProvider import/init: {exc!r}")

    try:
        from qbraid.runtime import BraketProvider
        provider = BraketProvider()
        device = provider.get_device(
            "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
        )
        return device
    except Exception as exc:
        errors.append(f"BraketProvider: {exc!r}")

    raise RuntimeError("Could not connect to Aquila:\n" + "\n".join(errors))


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "dict"):
        try:
            return jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        try:
            return jsonable(value.model_dump())
        except Exception:
            pass
    return repr(value)


# ---- Independent exact simulator for the translated waveform ----

def occupation_bits(n_atoms: int = 6) -> np.ndarray:
    states = np.arange(2**n_atoms)
    return np.stack(
        [((states >> (n_atoms - 1 - site)) & 1) for site in range(n_atoms)],
        axis=1,
    ).astype(float)


def interaction_data(positions_um: np.ndarray):
    bits = occupation_bits(6)
    displacement = positions_um[:, None, :] - positions_um[None, :, :]
    distance = np.sqrt(np.sum(displacement**2, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        coupling = 5.42e6 / distance**6
    np.fill_diagonal(coupling, 0.0)
    energy = np.einsum("si,ij,sj->s", bits, np.triu(coupling, 1), bits)
    return bits, bits.sum(axis=1), energy


def apply_global_rxy(state: np.ndarray, angle: float, phase: float) -> np.ndarray:
    result = state[None, :]
    cosine = np.asarray([np.cos(angle / 2.0)])
    sine = np.asarray([np.sin(angle / 2.0)])
    plus = np.asarray([np.exp(1j * phase)])
    minus = np.conjugate(plus)
    for qubit in range(6):
        left = 2**qubit
        right = 2 ** (5 - qubit)
        view = result.reshape(-1, left, 2, right)
        zero = view[:, :, 0, :].copy()
        one = view[:, :, 1, :].copy()
        view[:, :, 0, :] = (
            cosine[:, None, None] * zero
            - 1j * sine[:, None, None] * plus[:, None, None] * one
        )
        view[:, :, 1, :] = (
            -1j * sine[:, None, None] * minus[:, None, None] * zero
            + cosine[:, None, None] * one
        )
        result = view.reshape(-1, 64)
    return result[0]


def evolve_constant(
    state: np.ndarray,
    omega_rad_us: float,
    delta_rad_us: float,
    duration_us: float,
    total_occupation: np.ndarray,
    interaction_energy: np.ndarray,
) -> np.ndarray:
    scale = max(abs(omega_rad_us), abs(delta_rad_us), float(np.max(np.abs(interaction_energy))), 1e-12)
    substeps = int(np.clip(np.ceil(scale * duration_us / 0.25), 1, 512))
    dt = duration_us / substeps
    diagonal = interaction_energy - delta_rad_us * total_occupation
    half = np.exp(-0.5j * dt * diagonal)
    full = half * half
    result = state * half
    for index in range(substeps):
        result = apply_global_rxy(result, omega_rad_us * dt, 0.0)
        result *= half if index == substeps - 1 else full
    return result / np.linalg.norm(result)


def simulate_linear_program(
    positions_um: np.ndarray,
    times_s: np.ndarray,
    omega_rad_s: np.ndarray,
    delta_rad_s: np.ndarray,
    dt_max_us: float = 0.001,
) -> np.ndarray:
    _, total_occupation, interaction_energy = interaction_data(positions_um)
    state = np.zeros(64, complex)
    state[0] = 1.0
    for left in range(len(times_s) - 1):
        duration_us = (times_s[left + 1] - times_s[left]) * 1e6
        subintervals = max(1, int(np.ceil(duration_us / dt_max_us)))
        dt_us = duration_us / subintervals
        for sub in range(subintervals):
            fraction = (sub + 0.5) / subintervals
            omega = (
                omega_rad_s[left]
                + fraction * (omega_rad_s[left + 1] - omega_rad_s[left])
            ) / 1e6
            delta = (
                delta_rad_s[left]
                + fraction * (delta_rad_s[left + 1] - delta_rad_s[left])
            ) / 1e6
            state = evolve_constant(
                state,
                float(omega),
                float(delta),
                float(dt_us),
                total_occupation,
                interaction_energy,
            )
    probability = np.abs(state) ** 2
    return probability / probability.sum()


def raw_features(probabilities: list[np.ndarray]) -> np.ndarray:
    bits = occupation_bits(6)
    pairs = tuple(itertools.combinations(range(6), 2))
    pair_bits = np.stack([bits[:, i] * bits[:, j] for i, j in pairs], axis=1)
    blocks = []
    for probability in probabilities:
        blocks.append(probability @ bits)
        blocks.append(probability @ pair_bits)
    return np.concatenate(blocks)


def frozen_prediction(case: dict[str, np.ndarray], feature: np.ndarray):
    design = (
        feature - case["feature_scaler_mean"]
    ) / case["feature_scaler_scale"]
    raw = design @ case["readout_coefficients"].T
    correction = float(case["selected_lambda"][0]) * raw
    prediction = case["har_prediction"] + correction
    return prediction, correction


@dataclass
class Preflight:
    stretch: float
    programs: dict[int, Any]
    discretized: dict[int, Any]
    failures: dict[int, str]
    prediction: np.ndarray
    correction: np.ndarray
    late_delta_qlike: float
    late_delta_rmse: float
    total_times_us: dict[int, float]


def preflight_factor(
    case: dict[str, np.ndarray],
    stretch: float,
    aws_device: Any | None,
) -> Preflight:
    programs = {}
    discretized = {}
    failures = {}
    probabilities = []
    total_times = {}
    for probe in PROBES:
        times, omega, delta, phase = build_boundary_linear_series(
            case["encoded_sequence"], probe, stretch
        )
        total_times[probe] = float(times[-1] * 1e6)
        program = build_braket_program(
            case["interaction_matched_positions_um"], times, omega, delta, phase
        )
        programs[probe] = program
        if aws_device is not None:
            try:
                discretized[probe] = program.discretize(aws_device)
            except Exception as exc:
                failures[probe] = repr(exc)
        probabilities.append(
            simulate_linear_program(
                case["interaction_matched_positions_um"], times, omega, delta
            )
        )
    feature = raw_features(probabilities)
    prediction, correction = frozen_prediction(case, feature)
    y = case["realized_future"]
    har = case["har_prediction"]
    late = slice(4, 10)
    return Preflight(
        stretch=float(stretch),
        programs=programs,
        discretized=discretized,
        failures=failures,
        prediction=prediction,
        correction=correction,
        late_delta_qlike=qlike(y[late], prediction[late]) - qlike(y[late], har[late]),
        late_delta_rmse=rmse(y[late], prediction[late]) - rmse(y[late], har[late]),
        total_times_us=total_times,
    )


def save_preflight(outdir: Path, checks: list[Preflight]) -> None:
    rows = []
    for check in checks:
        rows.append(
            {
                "stretch": check.stretch,
                "all_three_discretized": len(check.discretized) == 3,
                "late_delta_qlike": check.late_delta_qlike,
                "late_delta_rmse": check.late_delta_rmse,
                "probe10_time_us": check.total_times_us[10],
                "probe20_time_us": check.total_times_us[20],
                "probe40_time_us": check.total_times_us[40],
                "failures": json.dumps(check.failures, sort_keys=True),
            }
        )
    import csv
    with (outdir / "preflight_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, default=Path("case151_freeze.npz"))
    parser.add_argument("--outdir", type=Path, default=Path("aquila_case151_run"))
    parser.add_argument(
        "--try-stretches",
        type=float,
        nargs="+",
        default=[1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0],
    )
    parser.add_argument("--stretch", type=float)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if args.shots < 1 or args.shots > 1000:
        raise ValueError("Aquila shots must lie in [1, 1000]")
    args.outdir.mkdir(parents=True, exist_ok=True)
    case = load_case(args.case)

    device = connect_aquila()
    print("DEVICE:", device)
    try:
        print("STATUS:", device.status())
    except Exception as exc:
        print("STATUS unavailable:", repr(exc))
    try:
        metadata = device.metadata()
        print("METADATA:", metadata)
        (args.outdir / "qbraid_device_metadata.json").write_text(
            json.dumps(jsonable(metadata), indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print("METADATA unavailable:", repr(exc))

    aws_device = find_aws_device(device)
    if aws_device is not None:
        properties = jsonable(aws_device.properties)
        (args.outdir / "aquila_live_properties.json").write_text(
            json.dumps(properties, indent=2) + "\n",
            encoding="utf-8",
        )
        print("Found wrapped AwsDevice:", aws_device.arn)
    else:
        print(
            "WARNING: wrapped AwsDevice was not exposed. Device discretization "
            "cannot be checked without submitting. Submission remains disabled."
        )

    factors = [args.stretch] if args.stretch is not None else args.try_stretches
    checks = [preflight_factor(case, float(value), aws_device) for value in factors]
    save_preflight(args.outdir, checks)

    print("\nPREFLIGHT")
    for check in checks:
        print(
            f"stretch={check.stretch:g} "
            f"discretized={len(check.discretized)}/3 "
            f"H5-H10 dQLIKE={check.late_delta_qlike:+.6f} "
            f"dRMSE={check.late_delta_rmse:+.6f} "
            f"times_us={check.total_times_us}"
        )
        if check.failures:
            for probe, failure in check.failures.items():
                print(f"  probe {probe}: {failure}")

    if not args.submit:
        print("\nDRY RUN ONLY. No quantum tasks were submitted.")
        print("Review preflight_summary.csv, then rerun with --stretch X --submit.")
        return

    if args.stretch is None:
        raise ValueError("--submit requires an explicit --stretch")
    selected = checks[0]
    if aws_device is None or len(selected.discretized) != 3:
        raise RuntimeError("Refusing submission: all three programs were not discretized")
    if selected.late_delta_qlike >= 0.0:
        raise RuntimeError(
            "Refusing submission: hardware-translated exact simulator loses the "
            "H5-H10 QLIKE improvement"
        )

    job_rows = []
    for probe in PROBES:
        program = selected.discretized[probe]
        job = device.run(program, shots=args.shots)
        job_id = getattr(job, "id", None)
        if callable(job_id):
            job_id = job_id()
        row = {
            "probe_steps": probe,
            "shots": args.shots,
            "stretch": args.stretch,
            "job_id": str(job_id),
            "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        job_rows.append(row)
        print("SUBMITTED:", row)

    (args.outdir / "submitted_jobs.json").write_text(
        json.dumps(job_rows, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
