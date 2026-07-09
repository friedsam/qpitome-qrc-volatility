"""Bounded ESN architecture variants for branch-resolution auditing.

This module intentionally leaves ``numpy_esn.py`` untouched. It exists only to
compare a small set of architecture repairs against the frozen primitive branch
ESN on the recovery-versus-relapse task.
"""

from __future__ import annotations

import numpy as np

from qpitome_qrc.baselines.numpy_esn import make_esn_weights


# Fixed max-absolute scales derived from the existing causal clip bounds.
# Zero remains zero and every channel is mapped to approximately [-1, 1].
PATH_MAX_ABS_SCALE = np.asarray([0.15, 0.15, 2.0, 1.5, 0.60, 3.0], dtype=float)


def fixed_scale_windows(windows: np.ndarray) -> np.ndarray:
    """Apply deterministic pre-reservoir scaling with no fitted statistics."""
    x = np.asarray(windows, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"Expected windows with shape (episodes, time, channels); got {x.shape}")
    if x.shape[2] != len(PATH_MAX_ABS_SCALE):
        raise ValueError(
            f"Expected {len(PATH_MAX_ABS_SCALE)} path channels; got {x.shape[2]}"
        )
    if not np.isfinite(x).all():
        raise ValueError("Path windows contain non-finite values")
    return x / PATH_MAX_ABS_SCALE[None, None, :]


def _prepare_inputs(windows: np.ndarray, *, normalized: bool, bias: bool) -> np.ndarray:
    x = fixed_scale_windows(windows) if normalized else np.asarray(windows, dtype=float)
    if bias:
        ones = np.ones((*x.shape[:2], 1), dtype=float)
        x = np.concatenate([x, ones], axis=2)
    return x


def reset_esn_audit_features(
    windows: np.ndarray,
    *,
    n_reservoir: int,
    spectral_radius: float,
    input_scale: float,
    leak: float,
    seed: int,
    normalized: bool,
    bias: bool,
    trajectory_mean: bool,
) -> np.ndarray:
    """Return reset-window ESN features for one bounded architecture variant.

    Primitive-compatible output:
        [final_state, final_raw_input]

    Trajectory-summary output:
        [final_state, mean_state, final_raw_input]

    The final raw input is always taken from the original six-channel path so
    the readout-side endpoint control remains matched across variants.
    """
    raw = np.asarray(windows, dtype=float)
    x = _prepare_inputs(raw, normalized=normalized, bias=bias)

    w_in, w = make_esn_weights(
        n_inputs=x.shape[2],
        n_reservoir=n_reservoir,
        spectral_radius=spectral_radius,
        input_scale=input_scale,
        seed=seed,
    )

    rows: list[np.ndarray] = []
    for raw_window, window in zip(raw, x, strict=True):
        h = np.zeros(n_reservoir, dtype=float)
        states: list[np.ndarray] = []
        for u_t in window:
            h_new = np.tanh(w_in @ u_t + w @ h)
            h = (1.0 - leak) * h + leak * h_new
            states.append(h.copy())

        if trajectory_mean:
            feature = np.concatenate(
                [h, np.mean(np.vstack(states), axis=0), raw_window[-1]]
            )
        else:
            feature = np.concatenate([h, raw_window[-1]])
        rows.append(feature)

    return np.asarray(rows, dtype=float)
