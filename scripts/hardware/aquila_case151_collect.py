#!/usr/bin/env python3
"""Collect three existing Aquila Case151 jobs; never submit new hardware work."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import runpy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROBES = (10, 20, 40)
POSITIONS_UM = np.asarray(
    [
        [-8.2, 4.4], [0.0, 4.4], [8.2, 4.4],
        [-8.2, -4.4], [0.0, -4.4], [8.2, -4.4],
    ],
    dtype=float,
)
DEFAULT_CASE = Path("reference/case151/freeze_001/case151_freeze.npz")
DEFAULT_SUPPORT = Path("scripts/hardware/aquila_case151_support.py")
DEFAULT_CONTRACT = Path("config/case151/expected_metrics.json")
DEFAULT_OUTDIR = Path(
    "results/transition_forecasting/qrc/aquila_case151_hardware/run/"
    "aquila_case151_existing_jobs_001"
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _jsonable(value.to_dict())
        except Exception:
            pass
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native_series(encoded: np.ndarray, probe: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if probe not in PROBES:
        raise ValueError(f"unsupported probe: {probe}")
    times_ns = [0]
    omega = [0.0]
    targets: list[float] = []
    for step in range(probe):
        omega_value = 6.0e6 * (1.0 + 0.6 * float(encoded[step, 1]))
        delta_value = 6.0e6 + 4.0e6 * float(encoded[step, 0])
        times_ns.append(50 + 95 * step)
        omega.append(float(int(round(omega_value / 400.0)) * 400))
        targets.append(float(round(delta_value / 0.2) * 0.2))
    delta = [targets[0], *targets]
    times_ns.append(50 + 95 * probe)
    omega.append(0.0)
    delta.append(delta[-1])
    times = np.asarray(times_ns, dtype=float) * 1e-9
    if np.any(np.diff(times) < 50e-9 - 1e-15) or times[-1] > 4e-6 + 1e-15:
        raise RuntimeError("hardware-native schedule violates timing limits")
    return times, np.asarray(omega), np.asarray(delta)


def _observable_names() -> tuple[str, ...]:
    occupations = tuple(f"n_{site}" for site in range(6))
    pairs = tuple(f"n_{left}_n_{right}" for left, right in itertools.combinations(range(6), 2))
    return occupations + pairs


def _job_manifest(outdir: Path, contract_path: Path) -> list[dict[str, Any]]:
    path = outdir / "job_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("jobs", [])
    else:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        rows = [
            {
                "probe_steps": int(probe),
                "job_id": str(job_id),
                "shots": 1000,
                "generated_during_this_run": False,
            }
            for probe, job_id in sorted(
                contract["hardware_job_ids"].items(), key=lambda item: int(item[0])
            )
        ]
        payload = {
            "schema_version": 1,
            "provider": "Amazon Braket via qBraid",
            "backend": "Aquila",
            "generated_during_this_run": False,
            "jobs": rows,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if len(rows) != 3 or {int(row["probe_steps"]) for row in rows} != set(PROBES):
        raise ValueError("job manifest must contain probes 10, 20, and 40")
    return rows


def _connect_job(job_id: str):
    from qbraid.runtime.native import QbraidJob, QbraidProvider

    device = QbraidProvider().get_device("aws:quera:qpu:aquila")
    return QbraidJob(job_id=job_id, device=device)


def _measurements(result: Any) -> tuple[np.ndarray, dict[str, int]]:
    raw = result.data.measurements or []
    used: list[np.ndarray] = []
    counts = {"total": len(raw), "successful": 0, "full_pre_sequence": 0, "used": 0}
    for measurement in raw:
        if not bool(getattr(measurement, "success", False)):
            continue
        counts["successful"] += 1
        pre = np.asarray(getattr(measurement, "pre_sequence", []), dtype=int)
        post = np.asarray(getattr(measurement, "post_sequence", []), dtype=int)
        if pre.shape != (6,) or post.shape != (6,) or not np.all(pre == 1):
            continue
        counts["full_pre_sequence"] += 1
        used.append(1 - post)
    if not used:
        raise RuntimeError("no usable full-pre-sequence measurements")
    counts["used"] = len(used)
    return np.stack(used).astype(float), counts


def _simulator_reference(support: dict[str, Any], case: dict[str, np.ndarray]) -> dict[int, np.ndarray]:
    bits = support["occupation_bits"](6)
    pair_bits = np.stack(
        [bits[:, left] * bits[:, right] for left, right in itertools.combinations(range(6), 2)],
        axis=1,
    )
    reference: dict[int, np.ndarray] = {}
    for probe in PROBES:
        times, omega, delta = _native_series(case["encoded_sequence"], probe)
        probability = support["simulate_linear_program"](
            POSITIONS_UM, times, omega, delta, dt_max_us=0.001
        )
        reference[probe] = np.concatenate([probability @ bits, probability @ pair_bits])
    return reference


def _write_contract_files(outdir: Path, freeze_json: Path) -> None:
    source = json.loads(freeze_json.read_text(encoding="utf-8"))
    (outdir / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_commit": source["source_model"]["source_commit"],
                "source_run_id": source["source_model"]["run_id"],
                "sample_id": source["sample_id"],
                "hardware_claim": source["hardware_claim"],
                "generated_during_this_run": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (outdir / "device_and_mapping.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "Aquila",
                "logical_to_physical": {str(index): index for index in range(6)},
                "positions_um": POSITIONS_UM.tolist(),
                "probe_steps": list(PROBES),
                "hardware_native_schedule": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (outdir / "predictions.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "not_applicable",
                "reason": (
                    "Observable transfer under an adapted hardware-native schedule; "
                    "not an end-to-end hardware volatility forecast."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (outdir / "runtime.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "retrieval_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "new_hardware_jobs_submitted": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (outdir / "README.md").write_text(
        "# Aquila Case151 observable transfer\n\n"
        "Retrieval and analysis of three previously completed Aquila jobs. The 63 "
        "occupations/pairs validate observable transfer under an adapted schedule; "
        "this is not an end-to-end hardware forecast. No new hardware job is submitted.\n",
        encoding="utf-8",
    )


def _artifact_manifest(outdir: Path) -> None:
    files = [
        {"path": path.name, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(outdir.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    (outdir / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": 1, "files": files}, indent=2) + "\n",
        encoding="utf-8",
    )


def _status(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        status = _connect_job(row["job_id"]).status()
        print(f"probe={int(row['probe_steps']):2d} job={row['job_id']} status={status}")


def _collect(support: dict[str, Any], case: dict[str, np.ndarray], rows: list[dict[str, Any]], outdir: Path) -> None:
    reference = _simulator_reference(support, case)
    names = _observable_names()
    pairs = tuple(itertools.combinations(range(6), 2))
    comparison: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    raw_archive: dict[str, np.ndarray] = {}
    for row in rows:
        probe = int(row["probe_steps"])
        job = _connect_job(row["job_id"])
        status = job.status()
        if getattr(status, "name", str(status)) != "COMPLETED":
            raise RuntimeError(f"probe {probe} is not completed: {status}")
        result = job.result()
        rydberg, counts = _measurements(result)
        hardware = np.concatenate(
            [
                rydberg.mean(axis=0),
                np.asarray([np.mean(rydberg[:, left] * rydberg[:, right]) for left, right in pairs]),
            ]
        )
        simulator = reference[probe]
        slope = float(np.dot(simulator, hardware) / np.dot(simulator, simulator))
        correlation = float(np.corrcoef(simulator, hardware)[0, 1])
        rmse = float(np.sqrt(np.mean((hardware - simulator) ** 2)))
        metrics.append(
            {
                "probe_steps": probe,
                "job_id": row["job_id"],
                "requested_shots": int(row["shots"]),
                "used_measurements": counts["used"],
                "total_measurements": counts["total"],
                "retention": counts["used"] / counts["total"],
                "attenuation_through_origin": slope,
                "pearson_correlation": correlation,
                "rmse": rmse,
                "reported_cost": _jsonable(job.metadata()).get("cost"),
            }
        )
        for index, name in enumerate(names):
            comparison.append(
                {
                    "probe_steps": probe,
                    "observable_index": index,
                    "observable": name,
                    "kind": "occupation" if index < 6 else "pair",
                    "simulator_expectation": float(simulator[index]),
                    "hardware_expectation": float(hardware[index]),
                    "difference": float(hardware[index] - simulator[index]),
                }
            )
        raw_archive[f"probe_{probe}_rydberg_bits"] = rydberg
        raw_archive[f"probe_{probe}_simulator_observables"] = simulator
        raw_archive[f"probe_{probe}_hardware_observables"] = hardware
    with (outdir / "raw_counts_or_observables.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    with (outdir / "metrics_by_probe.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    np.savez_compressed(outdir / "raw_counts_or_observables.npz", **raw_archive)
    simulator = np.asarray([row["simulator_expectation"] for row in comparison])
    hardware = np.asarray([row["hardware_expectation"] for row in comparison])
    pooled = {
        "attenuation_through_origin": float(np.dot(simulator, hardware) / np.dot(simulator, simulator)),
        "pearson_correlation": float(np.corrcoef(simulator, hardware)[0, 1]),
        "rmse": float(np.sqrt(np.mean((hardware - simulator) ** 2))),
        "n_observables": len(comparison),
    }
    (outdir / "metrics_pooled.json").write_text(json.dumps(pooled, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "status", "collect"))
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--support", type=Path, default=DEFAULT_SUPPORT)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    support = runpy.run_path(str(args.support))
    case = support["load_case"](args.case)
    rows = _job_manifest(args.outdir, args.contract)
    if args.mode == "status":
        _status(rows)
        return 0
    reference = _simulator_reference(support, case)
    np.savez_compressed(
        args.outdir / "simulator_reference.npz",
        **{f"probe_{probe}_observables": values for probe, values in reference.items()},
    )
    if args.mode == "collect":
        _collect(support, case, rows, args.outdir)
    _write_contract_files(args.outdir, args.case.with_suffix(".json"))
    _artifact_manifest(args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
