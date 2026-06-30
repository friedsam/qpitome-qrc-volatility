#!/usr/bin/env python3
"""
Credit-efficient qBraid / Amazon Braket / QuEra Aquila readiness probe.

Default mode is local AHS simulation only. Hardware submission requires --hardware.
Purpose: verify environment, credentials, device properties, AHS discretization,
minimal run/retrieval, and result format without touching the volatility model.

Typical use on qBraid Lab:
    python scripts/qbraid_ahs_readiness_probe.py --check-qbraid --simulate --shots 50
    python scripts/qbraid_ahs_readiness_probe.py --check-qbraid --device-check
    python scripts/qbraid_ahs_readiness_probe.py --hardware --shots 50 --out artifacts/ahs_probe_hw.json

Typical local use:
    pip install amazon-braket-sdk qbraid-cli
    qbraid configure
    python scripts/qbraid_ahs_readiness_probe.py --check-qbraid --simulate

Notes:
- qBraid organization/credit selection is account-side; confirm the active org in
  account.qbraid.com or qBraid Lab before any --hardware run.
- This script does not store or print API keys.
- Aquila supports Analog Hamiltonian Simulation, not gate circuits.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"


@dataclass
class ProbeConfig:
    shots: int
    spacing_um: float
    n_atoms: int
    time_max_s: float
    time_ramp_s: float
    omega_max_rad_s: float
    detuning_multiple: float


def run_cmd(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    try:
        p = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip()[-2000:],
            "stderr": p.stderr.strip()[-2000:],
        }
    except Exception as exc:  # noqa: BLE001
        return {"cmd": cmd, "error": repr(exc)}


def qbraid_diagnostics() -> dict[str, Any]:
    qbraid_path = shutil.which("qbraid")
    out: dict[str, Any] = {"qbraid_on_path": qbraid_path}
    if qbraid_path is None:
        return out

    # Keep this conservative: qBraid CLI surface changes; help/version are stable.
    out["version"] = run_cmd(["qbraid", "--version"])
    out["help"] = run_cmd(["qbraid", "--help"])
    out["account_help"] = run_cmd(["qbraid", "account", "--help"])
    out["devices_help"] = run_cmd(["qbraid", "devices", "--help"])
    out["jobs_help"] = run_cmd(["qbraid", "jobs", "--help"])
    return out


def build_minimal_ahs(cfg: ProbeConfig):
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries

    if cfg.n_atoms < 2:
        raise ValueError("n_atoms must be >= 2")

    spacing_m = cfg.spacing_um * 1e-6
    x0 = -0.5 * (cfg.n_atoms - 1) * spacing_m

    register = AtomArrangement()
    for k in range(cfg.n_atoms):
        register.add([x0 + k * spacing_m, 0.0])

    t0 = 0.0
    tr = cfg.time_ramp_s
    tm = cfg.time_max_s
    if not (0.0 < tr < tm / 2):
        raise ValueError("Require 0 < time_ramp_s < time_max_s/2")

    omega_max = cfg.omega_max_rad_s
    delta_start = -cfg.detuning_multiple * omega_max
    delta_end = cfg.detuning_multiple * omega_max

    omega = TimeSeries().put(t0, 0.0).put(tr, omega_max).put(tm - tr, omega_max).put(tm, 0.0)
    phase = TimeSeries().put(t0, 0.0).put(tm, 0.0)
    detuning = TimeSeries().put(t0, delta_start).put(tm, delta_end)

    drive = DrivingField(amplitude=omega, phase=phase, detuning=detuning)
    return AnalogHamiltonianSimulation(register=register, hamiltonian=drive)


def measurement_to_state(measurement: Any) -> str:
    # Braket AHS returns pre_sequence/post_sequence booleans per site.
    # e = initially empty/lost site; u = Rydberg/up; d = ground/down.
    state = []
    for pre, post in zip(measurement.pre_sequence, measurement.post_sequence, strict=True):
        if not pre:
            state.append("e")
        elif post:
            state.append("u")
        else:
            state.append("d")
    return "".join(state)


def summarize_ahs_result(result: Any) -> dict[str, Any]:
    counts = Counter(measurement_to_state(m) for m in result.measurements)
    shots = sum(counts.values())
    empty_site_shots = sum(v for k, v in counts.items() if "e" in k)
    mean_rydberg = 0.0
    if shots:
        mean_rydberg = sum(k.count("u") * v for k, v in counts.items()) / shots
    return {
        "shots_returned": shots,
        "n_unique_states": len(counts),
        "top_counts": counts.most_common(20),
        "empty_site_shot_fraction": empty_site_shots / shots if shots else None,
        "mean_rydberg_excitations": mean_rydberg,
    }


def braket_device_check() -> dict[str, Any]:
    from braket.aws import AwsDevice

    dev = AwsDevice(AQUILA_ARN)
    props = dev.properties.dict()
    out = {
        "arn": AQUILA_ARN,
        "name": getattr(dev, "name", None),
        "status": getattr(dev, "status", None),
        "provider_name": getattr(dev, "provider_name", None),
        "type": getattr(dev, "type", None),
        "properties_top_keys": sorted(props.keys()),
    }
    paradigm = props.get("paradigm", {})
    rydberg = paradigm.get("rydberg", {}) if isinstance(paradigm, dict) else {}
    service = props.get("service", {})
    out["service"] = service
    out["rydberg_top_keys"] = sorted(rydberg.keys()) if isinstance(rydberg, dict) else None
    out["raw_properties_excerpt"] = {
        "paradigm": paradigm,
        "action": props.get("action", {}),
        "deviceParameters": props.get("deviceParameters", {}),
    }
    return out


def run_local_sim(program: Any, shots: int) -> dict[str, Any]:
    from braket.devices import LocalSimulator

    sim = LocalSimulator("braket_ahs")
    task = sim.run(program, shots=shots)
    result = task.result()
    return summarize_ahs_result(result)


def run_hardware(program: Any, shots: int, experimental: bool = False) -> dict[str, Any]:
    from braket.aws import AwsDevice

    dev = AwsDevice(AQUILA_ARN)
    discretized = program.discretize(dev)
    kwargs: dict[str, Any] = {"shots": shots}
    if experimental:
        kwargs["experimental_capabilities"] = "ALL"
    task = dev.run(discretized, **kwargs)
    meta = task.metadata()
    result = task.result()
    return {"metadata": meta, "summary": summarize_ahs_result(result)}


def main() -> int:
    parser = argparse.ArgumentParser(description="qBraid / Aquila AHS readiness probe")
    parser.add_argument("--check-qbraid", action="store_true", help="Run qBraid CLI diagnostics")
    parser.add_argument("--device-check", action="store_true", help="Read Aquila device properties; no task submission")
    parser.add_argument("--simulate", action="store_true", help="Run local AHS simulator")
    parser.add_argument("--hardware", action="store_true", help="Submit to Aquila QPU; costs credits/money")
    parser.add_argument("--experimental", action="store_true", help="Enable experimental Aquila capabilities for hardware task")
    parser.add_argument("--shots", type=int, default=50)
    parser.add_argument("--n-atoms", type=int, default=4)
    parser.add_argument("--spacing-um", type=float, default=6.0)
    parser.add_argument("--time-max-s", type=float, default=3.0e-6)
    parser.add_argument("--time-ramp-s", type=float, default=0.3e-6)
    parser.add_argument("--omega-max-rad-s", type=float, default=6.3e6)
    parser.add_argument("--detuning-multiple", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ahs_readiness_probe.json"))
    args = parser.parse_args()

    if args.hardware and args.shots > 100:
        raise SystemExit("Refusing >100 hardware shots in readiness mode. Override by editing script intentionally.")
    if args.shots <= 0:
        raise SystemExit("shots must be positive")

    cfg = ProbeConfig(
        shots=args.shots,
        spacing_um=args.spacing_um,
        n_atoms=args.n_atoms,
        time_max_s=args.time_max_s,
        time_ramp_s=args.time_ramp_s,
        omega_max_rad_s=args.omega_max_rad_s,
        detuning_multiple=args.detuning_multiple,
    )

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "python": sys.version,
        "aquila_arn": AQUILA_ARN,
    }

    if args.check_qbraid:
        report["qbraid"] = qbraid_diagnostics()

    program = build_minimal_ahs(cfg)
    report["ahs_ir_preview"] = str(program.to_ir())[:4000]

    if args.device_check:
        report["aquila_device"] = braket_device_check()

    if args.simulate:
        report["local_ahs_simulator"] = run_local_sim(program, args.shots)

    if args.hardware:
        report["hardware_warning"] = "Submitted to Aquila QPU; this incurs qBraid/AWS/Braket cost."
        report["aquila_hardware"] = run_hardware(program, args.shots, experimental=args.experimental)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
