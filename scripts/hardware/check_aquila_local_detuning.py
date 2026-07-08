#!/usr/bin/env python3
"""Inspect qBraid's Aquila device/profile for local-detuning support.

No AWS credentials are used and no hardware task is submitted.
"""
from __future__ import annotations

import json

from qbraid.runtime import QbraidProvider

QBRAID_AQUILA_DEVICE_ID = "aws:quera:qpu:aquila"


def serialize(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return value


def main() -> int:
    provider = QbraidProvider()
    device = provider.get_device(QBRAID_AQUILA_DEVICE_ID)

    payload = {
        "status": str(device.status()),
        "profile": serialize(device.profile),
    }
    text = json.dumps(payload, indent=2, default=str)
    matches = [
        line.strip()
        for line in text.splitlines()
        if "local" in line.lower() or "detun" in line.lower()
    ]

    print(f"qbraid_device_id={QBRAID_AQUILA_DEVICE_ID}")
    print(f"status={device.status()}")
    print(f"experiment_type={device.profile.experiment_type}")
    print(f"program_spec={device.profile.program_spec}")
    print("\nqBraid device/profile lines containing 'local' or 'detun':")
    if matches:
        for line in matches:
            print(line)
    else:
        print("<none>")

    lowered = text.lower()
    advertised = (
        "localdetuning" in lowered
        or "local_detuning" in lowered
        or ("local" in lowered and "detuning" in lowered)
    )
    print(f"\nlocal_detuning_advertised_by_qbraid_profile={advertised}")
    print("No task was submitted. No AWS credentials were used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
