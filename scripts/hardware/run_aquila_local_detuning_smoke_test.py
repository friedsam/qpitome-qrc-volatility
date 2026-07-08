#!/usr/bin/env python3
"""Guarded qBraid Aquila local-detuning smoke test.

Dry run validates and serializes locally without hardware submission.
Hardware mode submits exactly one qBraid Aquila task after an explicit token.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

QBRAID_AQUILA_DEVICE_ID = "aws:quera:qpu:aquila"
CONFIRM_TOKEN = "SUBMIT_AQUILA_LOCAL_DETUNING_TEST"
EXPERIMENTAL_RUNTIME_OPTIONS = {"experimental_capabilities": "ALL"}


def install_qbraid_decimal_encoder_patch() -> None:
    """Work around qBraid 0.12.1 analog JSON serialization of Decimal values."""
    from qbraid.programs.analog._model import AnalogHamiltonianEncoder

    original_default = AnalogHamiltonianEncoder.default

    def patched_default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return original_default(self, obj)

    AnalogHamiltonianEncoder.default = patched_default


def build_program():
    from braket.ahs.analog_hamiltonian_simulation import AnalogHamiltonianSimulation
    from braket.ahs.atom_arrangement import AtomArrangement
    from braket.ahs.driving_field import DrivingField
    from braket.ahs.local_detuning import LocalDetuning
    from braket.timings.time_series import TimeSeries

    register = AtomArrangement()
    for x_um in (-10.5, -3.5, 3.5, 10.5):
        register.add([x_um * 1e-6, 0.0])

    t0 = 0.0
    t1 = 3.0e-6
    omega = 6.3e6

    drive = DrivingField(
        amplitude=(
            TimeSeries()
            .put(t0, 0.0)
            .put(0.3e-6, omega)
            .put(2.7e-6, omega)
            .put(t1, 0.0)
        ),
        phase=TimeSeries().put(t0, 0.0).put(t1, 0.0),
        detuning=TimeSeries().put(t0, -4.0 * omega).put(t1, 4.0 * omega),
    )

    local = LocalDetuning.from_lists(
        times=[t0, t1],
        values=[0.0, 1.0e6],
        pattern=[0.0, 0.33, 0.67, 1.0],
    )

    return AnalogHamiltonianSimulation(
        register=register,
        hamiltonian=drive + local,
    )


def normalize_counts(raw_counts: Any) -> dict[str, int]:
    if raw_counts is None:
        raise RuntimeError("Completed analog result contains no measurement counts")
    return {str(state): int(count) for state, count in dict(raw_counts).items()}


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    return {
        "shots_returned": sum(counts.values()),
        "n_unique_states": len(counts),
        "top_counts": Counter(counts).most_common(20),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def extract_counts(result: Any) -> dict[str, int]:
    return normalize_counts(result.data.get_counts())


def job_metadata(job: Any) -> dict[str, Any]:
    try:
        return dict(job.metadata())
    except Exception:
        return dict(getattr(job, "_cache_metadata", {}) or {})


def retrieve_job(job_id: str, out: Path) -> int:
    from qbraid.runtime.native.job import QbraidJob

    job = QbraidJob(job_id)
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "retrieve existing qBraid Aquila local-detuning smoke-test job",
        "job_id": job_id,
        "job_status_before_result": str(job.status()),
        "hardware_submitted": False,
    }
    result = job.result()
    report["job_status_final"] = str(job.status())
    report["job_metadata"] = job_metadata(job)
    report["result_success"] = bool(result.success)
    report["cost"] = str(result.cost)

    if not result.success:
        write_report(out, report)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        print(f"\nJob failed; diagnostic report written to {out}")
        return 2

    counts = extract_counts(result)
    report["measurement_counts"] = counts
    report["result_summary"] = summarize_counts(counts)
    write_report(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nRetrieved existing job only. Wrote {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--confirm", default="")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scratch/aquila_local_detuning_smoke/report.json"),
    )
    args = parser.parse_args()

    if args.hardware and args.job_id:
        raise SystemExit("Choose either --hardware or --job-id, not both")
    if args.shots <= 0 or args.shots > 100:
        raise SystemExit("Smoke-test shots must be in 1..100")
    if args.hardware and args.confirm != CONFIRM_TOKEN:
        raise SystemExit(
            f"Hardware submission blocked. Re-run with --confirm {CONFIRM_TOKEN}"
        )
    if args.job_id:
        return retrieve_job(args.job_id, args.out)

    from qbraid.runtime import QbraidProvider

    install_qbraid_decimal_encoder_patch()
    program = build_program()
    device = QbraidProvider().get_device(QBRAID_AQUILA_DEVICE_ID)
    device.validate([program])
    prepared = device.prepare(program)

    ir = program.to_ir()
    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "minimal qBraid Aquila local-detuning smoke test",
        "qbraid_device_id": QBRAID_AQUILA_DEVICE_ID,
        "shots_requested": args.shots,
        "runtime_options": EXPERIMENTAL_RUNTIME_OPTIONS,
        "program": {
            "n_atoms": 4,
            "spacing_um": 7.0,
            "duration_us": 3.0,
            "global_omega_max_rad_s": 6.3e6,
            "global_detuning_start_rad_s": -25.2e6,
            "global_detuning_end_rad_s": 25.2e6,
            "local_detuning_end_rad_s": 1.0e6,
            "local_pattern": [0.0, 0.33, 0.67, 1.0],
        },
        "qbraid_validation_passed": True,
        "qbraid_prepare_passed": True,
        "prepared_format": str(getattr(prepared, "format", type(prepared).__name__)),
        "ahs_ir_preview": str(ir)[:12000],
        "hardware_submitted": False,
    }

    if not args.hardware:
        write_report(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"\nDry run only. Wrote {args.out}")
        return 0

    report["device"] = {
        "status": str(device.status()),
        "experiment_type": str(device.profile.experiment_type),
        "program_spec": str(device.profile.program_spec),
    }

    job = device.run(
        program,
        shots=args.shots,
        runtime_options=EXPERIMENTAL_RUNTIME_OPTIONS,
        tags={
            "project": "qpitome-phase3",
            "test": "aquila-local-detuning-smoke",
        },
    )
    report["hardware_submitted"] = True
    report["job_id"] = str(job.id)
    report["job_status_initial"] = str(job.status())
    write_report(args.out, report)
    print(f"Submitted Aquila local-detuning smoke test: job {job.id}")
    print(f"Checkpoint written immediately to {args.out}")

    result = job.result()
    report["job_status_final"] = str(job.status())
    report["job_metadata"] = job_metadata(job)
    report["result_success"] = bool(result.success)
    report["cost"] = str(result.cost)

    if not result.success:
        write_report(args.out, report)
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        print(f"Final failed-job report written to {args.out}")
        return 2

    counts = extract_counts(result)
    report["measurement_counts"] = counts
    report["result_summary"] = summarize_counts(counts)
    write_report(args.out, report)

    print(json.dumps(report["result_summary"], indent=2, sort_keys=True))
    print(f"Final report written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
