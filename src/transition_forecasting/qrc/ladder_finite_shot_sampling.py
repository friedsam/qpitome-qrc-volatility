from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from transition_forecasting.qrc.ladder_mode_readout_tools import (
    ladder_mode_weights,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    Condition,
    TemporalRydbergChainConfig,
    _evolve_step_batch,
    _fresh_states,
    _resolve_probe_steps,
    encode_drives,
)
from transition_forecasting.qrc.temporal_rydberg_ladder import (
    StaggeredLadderGeometryConfig,
    precompute_ladder,
)


CACHE_SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


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


def evolve_ladder_probe_probabilities(
    scaled_windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    *,
    interaction_scale: float = 1.0,
    condition: Condition = "ordered",
) -> tuple[np.ndarray, dict[str, object]]:
    """Evolve the ladder once and retain exact basis probabilities at each probe."""

    reservoir.validate()
    geometry.validate()
    if reservoir.n_atoms != 6:
        raise ValueError("finite-shot ladder probabilities require six atoms")
    if reservoir.shots is not None:
        raise ValueError("probability extraction requires reservoir.shots=None")
    if interaction_scale < 0:
        raise ValueError("interaction_scale must be nonnegative")

    windows = np.asarray(scaled_windows, dtype=float)
    if (
        windows.ndim != 3
        or windows.shape[2] != 2
        or not np.isfinite(windows).all()
    ):
        raise ValueError(
            "scaled_windows must be finite with shape (samples, time, 2)"
        )

    precomputed = precompute_ladder(
        reservoir,
        geometry,
        interaction_scale=float(interaction_scale),
    )
    delta, omega = encode_drives(windows, reservoir, condition=condition)
    samples, steps, _ = windows.shape
    probe_steps = _resolve_probe_steps(steps, reservoir.probe_fractions)
    reset_each_step = condition == "reset"
    interactions = condition != "interaction_off" and interaction_scale > 0
    states = _fresh_states(samples, precomputed.n_atoms)
    blocks: list[np.ndarray] = []

    for step in range(steps):
        if reset_each_step:
            states = _fresh_states(samples, precomputed.n_atoms)
        states = _evolve_step_batch(
            states,
            omega[:, step],
            delta[:, step],
            reservoir,
            precomputed,
            interactions=interactions,
        )
        if step + 1 in probe_steps:
            probabilities = np.abs(states) ** 2
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            blocks.append(probabilities)

    stacked = validate_probe_probabilities(
        np.stack(blocks, axis=1),
        n_atoms=precomputed.n_atoms,
    )
    metadata: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "condition": condition,
        "geometry": "staggered_asymmetric_ladder_2x3",
        "interaction_scale": float(interaction_scale),
        "probe_steps": [int(value) for value in probe_steps],
        "samples": int(samples),
        "states": int(2**precomputed.n_atoms),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
        "positions_um": precomputed.positions.tolist(),
        "interaction_matrix_rad_us": precomputed.interaction_matrix.tolist(),
        "total_evolution_time_us": float(steps * reservoir.step_duration_us),
    }
    return stacked, metadata


def probabilities_to_occupations(
    probabilities: np.ndarray,
    occupation_bits: np.ndarray | None = None,
) -> np.ndarray:
    values = validate_probe_probabilities(probabilities)
    bits = (
        np.asarray(occupation_bits, dtype=float)
        if occupation_bits is not None
        else np.stack(
            [
                ((np.arange(64) >> (5 - site)) & 1)
                for site in range(6)
            ],
            axis=1,
        ).astype(float)
    )
    if bits.shape != (64, 6):
        raise ValueError("occupation_bits must have shape (64, 6)")
    return np.einsum("rps,sa->rpa", values, bits)


def occupations_to_symmetric_modes(
    occupations: np.ndarray,
) -> np.ndarray:
    values = np.asarray(occupations, dtype=float)
    if values.ndim != 3 or values.shape[2] != 6:
        raise ValueError("occupations must have shape (samples, probes, 6)")
    if not np.isfinite(values).all():
        raise ValueError("occupations contain non-finite values")
    weights = np.asarray(ladder_mode_weights()[:, :3], dtype=float)
    modes = np.einsum("rpa,am->rpm", values, weights)
    return modes.reshape(len(values), -1)


def probabilities_to_symmetric_modes(
    probabilities: np.ndarray,
) -> np.ndarray:
    return occupations_to_symmetric_modes(
        probabilities_to_occupations(probabilities)
    )


def stable_measurement_seed(
    base_seed: int,
    *,
    fold: int,
    sample_id: str,
    probe_step: int,
) -> int:
    payload = (
        f"qpitome-shot-v1|{int(base_seed)}|{int(fold)}|"
        f"{str(sample_id)}|{int(probe_step)}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sample_probe_probabilities(
    probabilities: np.ndarray,
    *,
    shots: int,
    base_seed: int,
    fold: int,
    sample_ids: np.ndarray,
    probe_steps: tuple[int, ...] | list[int],
) -> np.ndarray:
    """Sample each sample/probe independently and invariantly to row ordering."""

    exact = validate_probe_probabilities(probabilities)
    if int(shots) < 1:
        raise ValueError("shots must be positive")
    ids = np.asarray(sample_ids).astype(str)
    probes = tuple(int(value) for value in probe_steps)
    if ids.shape != (len(exact),):
        raise ValueError("sample_ids must align with probability rows")
    if len(probes) != exact.shape[1]:
        raise ValueError("probe_steps must align with probability probes")
    if len(set(probes)) != len(probes):
        raise ValueError("probe_steps must be unique")

    sampled = np.empty_like(exact)
    for row, sample_id in enumerate(ids):
        for probe_index, probe_step in enumerate(probes):
            seed = stable_measurement_seed(
                int(base_seed),
                fold=int(fold),
                sample_id=str(sample_id),
                probe_step=int(probe_step),
            )
            rng = np.random.default_rng(seed)
            counts = rng.multinomial(int(shots), exact[row, probe_index])
            sampled[row, probe_index] = counts / float(shots)
    return validate_probe_probabilities(sampled)


def shot_modes_from_probabilities(
    probabilities: np.ndarray,
    *,
    shots: int,
    base_seed: int,
    fold: int,
    sample_ids: np.ndarray,
    probe_steps: tuple[int, ...] | list[int],
) -> np.ndarray:
    sampled = sample_probe_probabilities(
        probabilities,
        shots=shots,
        base_seed=base_seed,
        fold=fold,
        sample_ids=sample_ids,
        probe_steps=probe_steps,
    )
    return probabilities_to_symmetric_modes(sampled)


def cache_identity(
    *,
    fold: int,
    sample_ids: np.ndarray,
    encoded_windows: np.ndarray,
    reservoir: TemporalRydbergChainConfig,
    geometry: StaggeredLadderGeometryConfig,
    interaction_scale: float,
    probe_steps: tuple[int, ...] | list[int],
    selection_parameters: dict[str, object],
) -> dict[str, object]:
    ids = np.asarray(sample_ids).astype(str)
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fold": int(fold),
        "sample_ids_sha256": array_sha256(ids.astype("U")),
        "encoded_windows_sha256": array_sha256(np.asarray(encoded_windows, dtype=float)),
        "reservoir_config": reservoir.to_dict(),
        "geometry_config": geometry.to_dict(),
        "interaction_scale": float(interaction_scale),
        "probe_steps": [int(value) for value in probe_steps],
        "selection_parameters": dict(selection_parameters),
    }


def write_probability_cache(
    path: Path,
    *,
    probabilities: np.ndarray,
    exact_modes: np.ndarray,
    sample_ids: np.ndarray,
    fold_splits: np.ndarray,
    leads: np.ndarray,
    labels: np.ndarray,
    episode_ids: np.ndarray,
    origin_dates: np.ndarray,
    identity: dict[str, object],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite probability cache: {target}")

    probs = validate_probe_probabilities(probabilities)
    modes = np.asarray(exact_modes, dtype=float)
    ids = np.asarray(sample_ids).astype(str)
    if modes.shape != (len(probs), probs.shape[1] * 3):
        raise ValueError("exact_modes must contain three modes per probe")
    arrays = {
        "sample_id": ids,
        "fold_split": np.asarray(fold_splits).astype(str),
        "lead": np.asarray(leads, dtype=int),
        "label": np.asarray(labels, dtype=int),
        "episode_id": np.asarray(episode_ids).astype(str),
        "origin_date": np.asarray(origin_dates).astype(str),
    }
    if any(np.asarray(values).shape != (len(probs),) for values in arrays.values()):
        raise ValueError("cache metadata arrays must align with probability rows")

    identity_json = _canonical_json(identity)
    identity_hash = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            probabilities=probs,
            exact_modes=modes,
            cache_identity_json=np.asarray(identity_json),
            cache_identity_sha256=np.asarray(identity_hash),
            **arrays,
        )
    temporary.replace(target)


def load_probability_cache(
    path: Path,
    *,
    expected_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"missing probability cache: {target}")
    with np.load(target, allow_pickle=False) as bundle:
        required = {
            "probabilities",
            "exact_modes",
            "sample_id",
            "fold_split",
            "lead",
            "label",
            "episode_id",
            "origin_date",
            "cache_identity_json",
            "cache_identity_sha256",
        }
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"probability cache missing arrays: {sorted(missing)}")
        identity_json = str(np.asarray(bundle["cache_identity_json"]).item())
        stored_hash = str(np.asarray(bundle["cache_identity_sha256"]).item())
        if hashlib.sha256(identity_json.encode("utf-8")).hexdigest() != stored_hash:
            raise ValueError("probability cache identity hash is invalid")
        identity = json.loads(identity_json)
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("probability cache identity does not match requested run")
        probabilities = validate_probe_probabilities(bundle["probabilities"])
        exact_modes = np.asarray(bundle["exact_modes"], dtype=float)
        sample_ids = np.asarray(bundle["sample_id"]).astype(str)
        if exact_modes.shape != (len(probabilities), probabilities.shape[1] * 3):
            raise ValueError("cached exact_modes have an invalid shape")
        if sample_ids.shape != (len(probabilities),):
            raise ValueError("cached sample IDs do not align")
        return {
            "probabilities": probabilities,
            "exact_modes": exact_modes,
            "sample_id": sample_ids,
            "fold_split": np.asarray(bundle["fold_split"]).astype(str),
            "lead": np.asarray(bundle["lead"], dtype=int),
            "label": np.asarray(bundle["label"], dtype=int),
            "episode_id": np.asarray(bundle["episode_id"]).astype(str),
            "origin_date": np.asarray(bundle["origin_date"]).astype(str),
            "identity": identity,
            "identity_sha256": stored_hash,
            "path": target,
        }
