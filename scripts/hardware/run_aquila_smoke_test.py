#!/usr/bin/env python3
"""Minimal qBraid Aquila access smoke test.

Purpose: verify that the current qBraid environment can discover Aquila, submit one
valid AHS program, and retrieve measurement counts. This is not a QRC benchmark.

Dry run (no hardware submission):
    python scripts/hardware/run_aquila_smoke_test.py

Hardware run (one small task):
    python scripts/hardware/run_aquila_smoke_test.py \
        --hardware --shots 20 --confirm SUBMIT_AQUILA_SMOKE_TEST

The script writes a checkpoint immediately after submission so the job ID is not
lost if the notebook/session disconnects while the task is queued.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

QBRAID_AQUILA_DEVICE_ID = "aws:quera:qpu:aquila"
CONFIRM_TOKEN = "SUBMIT_AQUILA_SMOKE_TEST"


def build_program():
    """Build a four-atom interacting AHS program with a simple global sweep."""
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.timings.time_series import TimeSeries

    # Four atoms, 7 um spacing: small but genuinely interacting.
    register = AtomArrangement()
    for x_um in (-10.5, -3.5, 3.5, 10.5):
        register.add([x_um * 1e-6, 0.0])

    t0 = 0.0
    t1 = 0.3e-6
    t2 = 2.7e-6
    t3 = 3.0e-6
    omega = 6.3e6

    amplitude = (
        TimeSeries()
        .put(t0, 0.0)
        .put(t1, omega)
        .put(t2, omega)
        .put(t3, 0.0)
    )
    phase = TimeSeries().put(t0, 0.0).put(t3, 0.0)
    detuning = TimeSeries().put(t0, -4.0 * omega).put(t3, 4.0 * omega)

    drive = DrivingField(amplitude=amplitude, phase=phase, detuning=detuning)
    return AnalogHamiltonianSimulation(register=register, hamiltonian=drive)


def normalize_counts(raw_counts: Any) -> dict[str, int]:
    """Convert qBraid measurement counts to a JSON-safe string-key dictionary."""
    return {str(state): int(count) for state, count in dict(raw_counts).items()}


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    ranked = Counter(counts).most_common(20)
    return {
        "shots_returned": total,
        "n_unique_states": len(counts),
        "top_counts": ranked,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true", help="Submit one Aquila task")
    parser.add_argument("--shots", type=int, default=20)
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scratch/aquila_smoke_test/report.json"),
    )
    args = parser.parse_args()

    if args.shots <= 0 or args.shots > 100:
        raise SystemExit("Smoke-test shots must be in 1..100")
    if args.hardware and args.confirm != CONFIRM_TOKEN:
        raise SystemExit(
            f"Hardware submission blocked. Re-run with --confirm {CONFIRM_TOKEN}"
        )

    program = build_program()
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "minimal qBraid Aquila access smoke test; not a benchmark",
        "qbraid_device_id": QBRAID_AQUILA_DEVICE_ID,
        "shots_requested": args.shots,
        "program": {
            "n_atoms": 4,
            "spacing_um": 7.0,
            "duration_us": 3.0,
            "global_omega_max_rad_s": 6.3e6,
            "global_detuning_start_rad_s": -25.2e6,
            "global_detuning_end_rad_s": 25.2e6,
        },
        "ahs_ir_preview": str(program.to_ir())[:12000],
        "hardware_submitted": False,
    }

    if not args.hardware:
        write_report(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nDry run only. Wrote {args.out}")
        return 0

    from qbraid.runtime import QbraidProvider

    provider = QbraidProvider()
    device = provider.get_device(QBRAID_AQUILA_DEVICE_ID)
    report["device"] = {
        "status": str(device.status()),
        "experiment_type": str(device.profile.experiment_type),
        "program_spec": str(device.profile.program_spec),
    }

    job = device.run(
        program,
        shots=args.shots,
        tags={"project": "qpitome-phase3", "test": "aquila-smoke"},
    )
    report["hardware_submitted"] = True
    report["job_id"] = str(job.id)
    report["job_status_initial"] = str(job.status())
    write_report(args.out, report)
    print(f"Submitted Aquila smoke test: job {job.id}")
    print(f"Checkpoint written to {args.out}")

    result = job.result()
    counts = normalize_counts(result.data.measurement_counts)
    report["job_status_final"] = str(job.status())
    report["measurement_counts"] = counts
    report["result_summary"] = summarize_counts(counts)
    write_report(args.out, report)

    print(json.dumps(report["result_summary"], indent=2, sort_keys=True))
    print(f"Final report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
