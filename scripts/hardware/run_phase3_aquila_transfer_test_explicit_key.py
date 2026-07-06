#!/usr/bin/env python3
"""Run the Phase 3 Aquila transfer test with an explicit qBraid API key.

This wrapper refuses to run unless QBRAID_API_KEY is set, then forces
QbraidProvider to use that exact key before delegating to the main transfer-test
script. This prevents paid jobs from silently using a different credential from
~/.qbraid/qbraidrc.
"""

from __future__ import annotations

import os

from qbraid import runtime as qbraid_runtime

import run_phase3_aquila_transfer_test as transfer_test


class ExplicitKeyQbraidProvider(qbraid_runtime.QbraidProvider):
    """QbraidProvider bound to the current QBRAID_API_KEY environment variable."""

    def __init__(self, *args, **kwargs):
        api_key = os.environ.get("QBRAID_API_KEY")
        if not api_key:
            raise RuntimeError(
                "QBRAID_API_KEY is not set. Refusing to use implicit qBraid credentials."
            )
        super().__init__(api_key=api_key)


def main() -> int:
    api_key = os.environ.get("QBRAID_API_KEY")
    if not api_key:
        raise SystemExit(
            "QBRAID_API_KEY is not set. Refusing to use implicit qBraid credentials."
        )

    qbraid_runtime.QbraidProvider = ExplicitKeyQbraidProvider
    return transfer_test.main()


if __name__ == "__main__":
    raise SystemExit(main())
