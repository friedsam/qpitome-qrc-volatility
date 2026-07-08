#!/usr/bin/env python3
"""Inspect the Aquila device schema for local-detuning support without submitting a task."""
from __future__ import annotations

import json
from braket.aws import AwsDevice

AQUILA_ARN = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"


def main() -> int:
    device = AwsDevice(AQUILA_ARN)
    properties = device.properties

    if hasattr(properties, "model_dump"):
        payload = properties.model_dump()
    elif hasattr(properties, "dict"):
        payload = properties.dict()
    else:
        payload = properties

    text = json.dumps(payload, indent=2, default=str)
    matches = [
        line.strip()
        for line in text.splitlines()
        if "local" in line.lower() or "detun" in line.lower()
    ]

    print(f"device={device.name}")
    print(f"arn={device.arn}")
    print(f"status={device.status}")
    print("\nSchema lines containing 'local' or 'detun':")
    if matches:
        for line in matches:
            print(line)
    else:
        print("<none>")

    lowered = text.lower()
    has_local_detuning = (
        "localdetuning" in lowered
        or "local_detuning" in lowered
        or ("local" in lowered and "detuning" in lowered)
    )
    print(f"\nlocal_detuning_advertised={has_local_detuning}")
    print("No task was submitted.")
    return 0 if has_local_detuning else 2


if __name__ == "__main__":
    raise SystemExit(main())
