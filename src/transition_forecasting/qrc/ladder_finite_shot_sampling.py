"""Exact probability validation and feature conversion for Case151."""
from __future__ import annotations

import numpy as np

from transition_forecasting.qrc.ladder_mode_readout_tools import ladder_mode_weights


def validate_probe_probabilities(
    probabilities: np.ndarray,
    *,
    n_atoms: int = 6,
    atol: float = 1e-12,
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 3:
        raise ValueError("probabilities must have shape (samples, probes, states)")
    expected_states = 2**int(n_atoms)
    if values.shape[2] != expected_states:
        raise ValueError(
            f"probability state width must be {expected_states}, got {values.shape[2]}"
        )
    if not np.isfinite(values).all():
        raise ValueError("probabilities contain non-finite values")
    if np.any(values < -atol):
        raise ValueError("probabilities contain negative values")
    totals = values.sum(axis=2)
    if not np.allclose(totals, 1.0, atol=atol, rtol=0.0):
        raise ValueError("probability vectors must sum to one")
    return np.clip(values, 0.0, 1.0)


def probabilities_to_occupations(
    probabilities: np.ndarray,
    occupation_bits: np.ndarray | None = None,
) -> np.ndarray:
    values = validate_probe_probabilities(probabilities)
    bits = (
        np.asarray(occupation_bits, dtype=float)
        if occupation_bits is not None
        else np.stack(
            [((np.arange(64) >> (5 - site)) & 1) for site in range(6)], axis=1
        ).astype(float)
    )
    if bits.shape != (64, 6):
        raise ValueError("occupation_bits must have shape (64, 6)")
    return np.einsum("rps,sa->rpa", values, bits)


def occupations_to_symmetric_modes(occupations: np.ndarray) -> np.ndarray:
    values = np.asarray(occupations, dtype=float)
    if values.ndim != 3 or values.shape[2] != 6:
        raise ValueError("occupations must have shape (samples, probes, 6)")
    if not np.isfinite(values).all():
        raise ValueError("occupations contain non-finite values")
    weights = np.asarray(ladder_mode_weights()[:, :3], dtype=float)
    return np.einsum("rpa,am->rpm", values, weights).reshape(len(values), -1)


def probabilities_to_symmetric_modes(probabilities: np.ndarray) -> np.ndarray:
    return occupations_to_symmetric_modes(probabilities_to_occupations(probabilities))
