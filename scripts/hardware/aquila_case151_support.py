#!/usr/bin/env python3
"""Retrieval-only support for the frozen GE151 Aquila observable-transfer run.

This module contains no task-submission entry point. It only loads the frozen case,
connects to the existing Aquila job interface, and reproduces the exact simulator
reference for the hardware-native linear waveform.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


def load_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: np.asarray(bundle[name]) for name in bundle.files}


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
                return provider.get_device(identifier)
            except Exception as exc:  # pragma: no cover - provider/version dependent
                errors.append(f"QbraidProvider {identifier}: {exc!r}")
    except Exception as exc:  # pragma: no cover - optional qBraid runtime
        errors.append(f"QbraidProvider import/init: {exc!r}")

    try:
        from qbraid.runtime import BraketProvider

        provider = BraketProvider()
        return provider.get_device(
            "arn:aws:braket:us-east-1::device/qpu/quera/Aquila"
        )
    except Exception as exc:  # pragma: no cover - optional qBraid runtime
        errors.append(f"BraketProvider: {exc!r}")

    raise RuntimeError("Could not connect to Aquila:\n" + "\n".join(errors))


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
    scale = max(
        abs(omega_rad_us),
        abs(delta_rad_us),
        float(np.max(np.abs(interaction_energy))),
        1e-12,
    )
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
