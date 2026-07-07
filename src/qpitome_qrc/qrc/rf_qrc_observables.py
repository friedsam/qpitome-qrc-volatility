"""Selected-observable readout variants for the shared RF-QRC state map."""

from __future__ import annotations

from typing import Literal

import numpy as np

from qpitome_qrc.qrc.rf_qrc_reservoir import RFQRCMap

ZZMode = Literal["ring", "ring_plus_next", "all"]


def selected_zz_pairs(n_qubits: int, mode: ZZMode) -> list[tuple[int, int]]:
    """Return the de-duplicated unordered ZZ pairs used by the scaling study."""

    pairs: list[tuple[int, int]] = []
    if mode in {"ring", "ring_plus_next"}:
        pairs.extend((i, (i + 1) % n_qubits) for i in range(n_qubits))
    if mode == "ring_plus_next":
        pairs.extend((i, (i + 2) % n_qubits) for i in range(n_qubits))
    if mode == "all":
        pairs.extend(
            (i, j)
            for i in range(n_qubits)
            for j in range(i + 1, n_qubits)
        )
    if mode not in {"ring", "ring_plus_next", "all"}:
        raise ValueError(f"Unknown ZZ mode: {mode}")

    unique: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, j in pairs:
        pair = tuple(sorted((i, j)))
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    return unique


def selected_z_zz_features(
    state: np.ndarray,
    n_qubits: int,
    zz_mode: ZZMode,
) -> np.ndarray:
    """Return all Z values plus the requested subset of ZZ observables."""

    probs = np.abs(state) ** 2
    zvals = np.empty((len(state), n_qubits), dtype=float)
    for idx in range(len(state)):
        for q in range(n_qubits):
            zvals[idx, q] = 1.0 if ((idx >> q) & 1) == 0 else -1.0

    z = probs @ zvals
    zz = [
        probs @ (zvals[:, i] * zvals[:, j])
        for i, j in selected_zz_pairs(n_qubits, zz_mode)
    ]
    return np.concatenate([z, np.asarray(zz, dtype=float)])


class SelectedObservableRFQRCMap(RFQRCMap):
    """Shared second-encoding ring RF-QRC with a selected ZZ readout set."""

    def __init__(
        self,
        n_qubits: int,
        input_scale: float,
        random_scale: float,
        seed: int,
        zz_mode: ZZMode,
    ) -> None:
        super().__init__(
            n_qubits=n_qubits,
            second_encoding=True,
            entangler="ring",
            input_scale=input_scale,
            seed=seed,
            random_scale=random_scale,
        )
        self.zz_mode = zz_mode

    def one(self, u: np.ndarray) -> np.ndarray:
        return selected_z_zz_features(self.state(u), self.n, self.zz_mode)
