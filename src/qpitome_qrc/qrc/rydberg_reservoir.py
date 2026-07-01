"""True temporal Rydberg analog reservoir (exact-state simulation).

This module replaces the static response-grid surrogate with a genuine
temporal reservoir: for each date, the trailing market window is encoded into
the *time dependence* of the global drive, and ONE quantum state is evolved
through the whole window, so inputs at different anchors interact through the
Rydberg dynamics. This mirrors the Phase 2 TFIM-QRC design under the control
constraints of neutral-atom AHS hardware (QuEra Aquila).

Physics
-------
H(t)/hbar = (Omega(t)/2) * sum_i sigma^x_i
            - Delta(t) * sum_i n_i
            + sum_{i<j} C6 / r_ij^6 * n_i n_j

with n_i = |r><r|_i (bit value 1 = Rydberg). Omega and Delta are *global*
controls: per-anchor single-qubit rotations (as in the gate-based TFIM) are
not available on Aquila, so market input is injected as piecewise-constant
plateaus of the drive:

    level channel  u_lvl(k)  ->  Delta_k = delta_center + delta_span * u_lvl
    rate  channel  u_rate(k) ->  Omega_k = omega_base * (1 + omega_mod_frac * u_rate)

This ports the Bell-model level x rate structure to two physically distinct
global channels. Spatial heterogeneity (multi-timescale reservoir) comes from
geometry: a tightly spaced "slow" sublattice (strong blockade, long collective
memory) and a widely spaced "fast" sublattice, cf. the dual_chain geometry.

Readout is hardware-native: site occupations <n_i> and all-pairs correlators
<n_i n_j> only (projective Z-basis measurement; no X observables). Anchor-wise
trajectory features correspond, on hardware, to re-running the same waveform
truncated at each anchor time (shot cost scales linearly in anchor count).

Simulation is exact-statevector with second-order Trotter splitting of the
diagonal (interaction + detuning) and transverse (Omega) terms, batched over
samples. Optional shot noise samples bitstrings from |psi|^2.

Controls built in (internal ablations):
- memory_mode="memoryless": fresh state per anchor, one segment of evolution,
  then measurement. This is the honest version of the response-grid surrogate
  and isolates the value of temporal quantum memory.
- shuffle_anchors=True: fixed permutation of anchor order; if performance is
  insensitive, the reservoir is not using temporal structure.
- reverse_anchors=True: inject selected anchors newest-to-oldest.
- omega_mode="constant": disables the rate channel (level-only encoding).

Hardware caveat: Aquila constraint constants below reflect publicly documented
specs and MUST be re-verified against current QuEra/Braket documentation
before designing hardware waveforms (specs have changed over time). The
simulated piecewise-constant plateaus additionally need finite-slew ramps on
hardware; `validate_aquila_feasibility` estimates that time cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.evaluation.metrics import (
    VolatilityForecastMetrics,
    evaluate_volatility_forecast,
)
from qpitome_qrc.qrc.tfim_reservoir import (
    fit_qrc_readout,
    predict_qrc_readout,
    select_anchor_indices,
)

Geometry = Literal["chain", "dual_chain"]
MemoryMode = Literal["temporal", "memoryless"]
OmegaMode = Literal["encode_rate", "constant"]
ObservableMode = Literal["n", "n_nn"]

# --- Aquila-like hardware constants (VERIFY against current QuEra docs) -----
C6_RAD_UM6_PER_US = 5.42e6  # C6 / hbar for Rb 70S, rad * um^6 / us


@dataclass(frozen=True)
class AquilaConstraints:
    """Publicly documented Aquila limits. Re-verify before hardware runs."""

    omega_max_rad_us: float = 15.8          # ~2*pi*2.5 MHz
    delta_abs_max_rad_us: float = 125.0     # global detuning
    total_time_max_us: float = 4.0
    min_spacing_um: float = 4.0
    area_width_um: float = 75.0
    area_height_um: float = 76.0
    omega_slew_rad_us2: float = 250.0
    delta_slew_rad_us2: float = 2500.0
    max_shots_per_task: int = 1000


@dataclass(frozen=True)
class RydbergQRCConfig:
    """Temporal Rydberg reservoir configuration.

    Defaults are tuned to a GENTLE PHASE BUDGET: per-segment accumulated
    phases (V_nn * t_seg ~ 3 rad, Omega * t_seg ~ 1.8 rad, input-differential
    detuning phase ~ 1.2 rad) comparable to the working Phase 2 TFIM regime.
    Empirically, "hot" regimes (V * t_seg >> 1) produce hyper-oscillatory
    input->feature maps that fit train and collapse out-of-sample; the
    memoryless control then BEATS the temporal reservoir. Keep phase budgets
    of order unity per anchor. The default drive still sweeps Delta/Omega
    across the ordered-phase transition region of the slow sublattice.
    """

    # Geometry
    geometry: Geometry = "dual_chain"
    n_atoms_slow: int = 4
    n_atoms_fast: int = 4
    spacing_slow_um: float = 9.0
    spacing_fast_um: float = 15.0
    row_gap_um: float = 14.0
    chain_atoms: int = 8            # used only for geometry="chain"
    chain_spacing_um: float = 6.5   # used only for geometry="chain"

    # Temporal encoding
    lookback_days: int = 40
    anchor_count: int = 6
    anchor_policy: Literal["even", "recent"] = "even"
    total_time_us: float = 1.8

    # Drive encoding (inputs assumed robust-scaled to ~[-1, 1])
    delta_center_rad_us: float = 6.0
    delta_span_rad_us: float = 4.0
    omega_base_rad_us: float = 6.0
    omega_mod_frac: float = 0.5
    omega_mode: OmegaMode = "encode_rate"

    # Reservoir controls / ablations
    memory_mode: MemoryMode = "temporal"
    reverse_anchors: bool = False
    shuffle_anchors: bool = False
    shuffle_seed: int = 1234

    # Readout
    observable_mode: ObservableMode = "n_nn"
    collect_anchor_features: bool = True
    shots: int | None = None
    shot_seed: int = 7

    # Numerics
    max_phase_per_substep: float = 0.25  # rad; controls Trotter substep count
    max_substeps_per_segment: int = 2000

    # Regression readout (mirrors TFIMQRCConfig fields used by fit_qrc_readout)
    ridge_alpha: float = 10.0
    target_transform: Literal["log", "none"] = "log"
    seed: int = 42


@dataclass
class RydbergQRCResult:
    config: RydbergQRCConfig
    target: str
    train_metrics: VolatilityForecastMetrics
    val_metrics: VolatilityForecastMetrics
    test_metrics: VolatilityForecastMetrics
    train_predictions: np.ndarray
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    train_features: np.ndarray
    val_features: np.ndarray
    test_features: np.ndarray
    readout: Ridge
    feature_scaler: StandardScaler | None


# --------------------------------------------------------------------------
# Geometry and precomputation
# --------------------------------------------------------------------------

def atom_positions(config: RydbergQRCConfig) -> np.ndarray:
    """Return (n_atoms, 2) positions in um."""
    if config.geometry == "chain":
        n = config.chain_atoms
        xs = np.arange(n) * config.chain_spacing_um
        return np.column_stack([xs, np.zeros(n)])
    if config.geometry == "dual_chain":
        ns, nf = config.n_atoms_slow, config.n_atoms_fast
        xs_slow = np.arange(ns) * config.spacing_slow_um
        xs_fast = np.arange(nf) * config.spacing_fast_um
        # Center each row around x = 0 to keep cross-row distances symmetric.
        xs_slow = xs_slow - xs_slow.mean()
        xs_fast = xs_fast - xs_fast.mean()
        slow = np.column_stack([xs_slow, np.zeros(ns)])
        fast = np.column_stack([xs_fast, np.full(nf, config.row_gap_um)])
        return np.vstack([slow, fast])
    raise ValueError(f"Unknown geometry: {config.geometry}")


def interaction_matrix(positions: np.ndarray) -> np.ndarray:
    """All-pairs van der Waals couplings V_ij = C6 / r_ij^6 in rad/us."""
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.sqrt((diff**2).sum(axis=-1))
    with np.errstate(divide="ignore"):
        v = C6_RAD_UM6_PER_US / r**6
    np.fill_diagonal(v, 0.0)
    return v


@dataclass
class _Precomputed:
    n_atoms: int
    positions: np.ndarray
    v_matrix: np.ndarray
    occ_bits: np.ndarray        # (2^n, n) float
    e_int: np.ndarray           # (2^n,) interaction energy per basis state
    n_occ: np.ndarray           # (2^n,) total occupation per basis state
    pair_index: list[tuple[int, int]]
    pair_bits: np.ndarray       # (2^n, n_pairs) float


def precompute(config: RydbergQRCConfig) -> _Precomputed:
    positions = atom_positions(config)
    n = positions.shape[0]
    if n > 14:
        raise ValueError(f"{n} atoms exceeds exact-state practical limit (14)")
    v = interaction_matrix(positions)

    dim = 2**n
    indices = np.arange(dim)
    # Bit convention matches tfim_reservoir: qubit q is bit (n-1-q).
    occ = np.stack([((indices >> (n - 1 - q)) & 1) for q in range(n)], axis=1)
    occ = occ.astype(float)

    e_int = np.einsum("si,ij,sj->s", occ, np.triu(v, k=1), occ)
    n_occ = occ.sum(axis=1)

    pair_index = [(a, b) for a in range(n - 1) for b in range(a + 1, n)]
    if pair_index:
        pair_bits = np.stack([occ[:, a] * occ[:, b] for a, b in pair_index], axis=1)
    else:
        pair_bits = np.zeros((dim, 0))

    return _Precomputed(
        n_atoms=n,
        positions=positions,
        v_matrix=v,
        occ_bits=occ,
        e_int=e_int,
        n_occ=n_occ,
        pair_index=pair_index,
        pair_bits=pair_bits,
    )


# --------------------------------------------------------------------------
# Batched exact-state evolution
# --------------------------------------------------------------------------

def _apply_global_rx_batch(states: np.ndarray, thetas: np.ndarray, n: int) -> np.ndarray:
    """Apply Rx(theta_s) on every qubit, batched over samples.

    states: (S, 2^n) complex; thetas: (S,) per-sample angles.
    """
    c = np.cos(thetas / 2.0)
    s = np.sin(thetas / 2.0)
    for q in range(n):
        left = 2**q
        right = 2 ** (n - 1 - q)
        arr = states.reshape(-1, left, 2, right)
        a0 = arr[:, :, 0, :].copy()
        a1 = arr[:, :, 1, :]
        cb = c[:, None, None]
        sb = s[:, None, None]
        arr[:, :, 0, :] = cb * a0 - 1j * sb * a1
        arr[:, :, 1, :] = -1j * sb * a0 + cb * a1
        states = arr.reshape(-1, 2**n)
    return states


def _evolve_segment_batch(
    states: np.ndarray,
    omega: np.ndarray,
    delta: np.ndarray,
    t_seg: float,
    pre: _Precomputed,
    config: RydbergQRCConfig,
) -> np.ndarray:
    """Evolve batch of states one piecewise-constant segment.

    Second-order Trotter: half diagonal, full X, half diagonal, with
    consecutive half-diagonals merged across substeps.
    omega, delta: (S,) per-sample drive values for this segment.
    """
    v_scale = float(pre.e_int.max(initial=0.0))
    scale = max(np.abs(delta).max(initial=0.0), np.abs(omega).max(initial=0.0), v_scale, 1e-9)
    m = int(np.ceil(scale * t_seg / config.max_phase_per_substep))
    m = int(np.clip(m, 1, config.max_substeps_per_segment))
    dt = t_seg / m

    # Diagonal generator per sample per basis state:
    #   D = E_int(s) - Delta * N_occ(s)
    diag = pre.e_int[None, :] - delta[:, None] * pre.n_occ[None, :]
    half = np.exp(-0.5j * dt * diag)
    full = half * half
    thetas = omega * dt

    states = states * half
    for step in range(m):
        states = _apply_global_rx_batch(states, thetas, pre.n_atoms)
        states = states * (half if step == m - 1 else full)
    return states


def _drive_from_window(
    windows: np.ndarray,
    anchor_indices: np.ndarray,
    config: RydbergQRCConfig,
    hw: AquilaConstraints,
) -> tuple[np.ndarray, np.ndarray]:
    """Map (S, lookback, 2) level/rate windows to per-anchor drives.

    Returns (deltas, omegas), each (S, K).
    """
    level = windows[:, anchor_indices, 0]
    rate = windows[:, anchor_indices, 1]

    deltas = config.delta_center_rad_us + config.delta_span_rad_us * level
    deltas = np.clip(deltas, -hw.delta_abs_max_rad_us, hw.delta_abs_max_rad_us)

    if config.omega_mode == "encode_rate":
        omegas = config.omega_base_rad_us * (1.0 + config.omega_mod_frac * rate)
    elif config.omega_mode == "constant":
        omegas = np.full_like(deltas, config.omega_base_rad_us)
    else:
        raise ValueError(f"Unknown omega_mode: {config.omega_mode}")
    omegas = np.clip(omegas, 0.0, hw.omega_max_rad_us)
    return deltas, omegas


def _measure_features(
    states: np.ndarray,
    pre: _Precomputed,
    config: RydbergQRCConfig,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Occupations and pair correlators from a batch of states.

    Exact expectations if config.shots is None; otherwise shot-noise estimates
    from multinomial bitstring sampling.
    """
    probs = np.abs(states) ** 2
    probs = probs / probs.sum(axis=1, keepdims=True)

    if config.shots is not None:
        if rng is None:
            raise ValueError("rng required when shots is set")
        counts = rng.multinomial(config.shots, probs)
        probs = counts / float(config.shots)

    feats = [probs @ pre.occ_bits]
    if config.observable_mode == "n_nn":
        feats.append(probs @ pre.pair_bits)
    return np.concatenate(feats, axis=1)


def select_rydberg_anchor_indices(config: RydbergQRCConfig) -> np.ndarray:
    """Select and optionally reorder anchor indices for Rydberg drive injection."""
    anchor_indices = select_anchor_indices(
        config.lookback_days, config.anchor_count, config.anchor_policy
    )
    if config.reverse_anchors:
        anchor_indices = anchor_indices[::-1]
    if config.shuffle_anchors:
        perm_rng = np.random.default_rng(config.shuffle_seed)
        anchor_indices = anchor_indices[perm_rng.permutation(len(anchor_indices))]
    return np.asarray(anchor_indices, dtype=int)


def build_rydberg_feature_matrix(
    X_windows: np.ndarray,
    config: RydbergQRCConfig,
    *,
    hw: AquilaConstraints | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Build reservoir features for (S, lookback, 2) level/rate windows.

    Evolution is batched over all samples simultaneously.
    """
    if X_windows.ndim != 3 or X_windows.shape[2] != 2:
        raise ValueError(f"Expected windows (S, lookback, 2), got {X_windows.shape}")
    hw = hw or AquilaConstraints()
    pre = precompute(config)
    S = X_windows.shape[0]
    dim = 2**pre.n_atoms

    anchor_indices = select_rydberg_anchor_indices(config)
    K = len(anchor_indices)
    t_seg = config.total_time_us / K

    deltas, omegas = _drive_from_window(X_windows, anchor_indices, config, hw)
    shot_rng = np.random.default_rng(config.shot_seed) if config.shots is not None else None

    def fresh_states() -> np.ndarray:
        st = np.zeros((S, dim), dtype=complex)
        st[:, 0] = 1.0
        return st

    per_anchor: list[np.ndarray] = []
    if config.memory_mode == "temporal":
        states = fresh_states()
        for k in range(K):
            if verbose:
                print(f"Rydberg segment {k + 1}/{K} (batch {S})")
            states = _evolve_segment_batch(states, omegas[:, k], deltas[:, k], t_seg, pre, config)
            if config.collect_anchor_features or k == K - 1:
                per_anchor.append(_measure_features(states, pre, config, shot_rng))
    elif config.memory_mode == "memoryless":
        for k in range(K):
            if verbose:
                print(f"Rydberg memoryless segment {k + 1}/{K} (batch {S})")
            states = _evolve_segment_batch(fresh_states(), omegas[:, k], deltas[:, k], t_seg, pre, config)
            per_anchor.append(_measure_features(states, pre, config, shot_rng))
    else:
        raise ValueError(f"Unknown memory_mode: {config.memory_mode}")

    return np.concatenate(per_anchor, axis=1)


# --------------------------------------------------------------------------
# Level/rate sequence construction (leakage-safe)
# --------------------------------------------------------------------------

def ensure_market_scalar(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add derived market scalar columns used for drive encoding."""
    out = df.copy()
    if col in out.columns:
        return out
    if col == "rv_accel_log_5_20":
        out[col] = np.log(out["rv_5d"].clip(lower=1e-12) / out["rv_20d"].clip(lower=1e-12))
    elif col == "vix_rv_spread":
        out[col] = np.log(
            (out["vix_close"].clip(lower=1e-12) / 100.0) / out["rv_20d"].clip(lower=1e-12)
        )
    else:
        raise ValueError(f"Unknown scalar column '{col}'")
    return out


def _robust_scale_params(train_values: np.ndarray, q_low: float, q_high: float) -> tuple[float, float]:
    lo = float(np.nanquantile(train_values, q_low))
    hi = float(np.nanquantile(train_values, q_high))
    med = float(np.nanmedian(train_values))
    half = 0.5 * (hi - lo)
    if not np.isfinite(half) or half <= 0:
        raise ValueError("Invalid robust scale")
    return med, half


def make_level_rate_sequence_splits(
    splits: dict[str, pd.DataFrame],
    *,
    level_col: str,
    rate_col: str,
    target_column: str,
    lookback_days: int,
    clip_q_low: float = 0.01,
    clip_q_high: float = 0.99,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.Series]]:
    """Build (S, lookback, 2) level/rate windows per split.

    Robust scaling to [-1, 1] is fitted on the training split ONLY and applied
    to all splits (leakage-safe). Window construction matches
    make_sequence_arrays: X[t] holds the trailing `lookback` rows ending at t,
    y[t] is the (already forward-looking) target at t.
    """
    prepared = {
        name: ensure_market_scalar(ensure_market_scalar(df, level_col), rate_col)
        for name, df in splits.items()
    }
    train = prepared["train"]
    params = {
        col: _robust_scale_params(train[col].to_numpy(float), clip_q_low, clip_q_high)
        for col in (level_col, rate_col)
    }

    out: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]] = {}
    for name, df in prepared.items():
        channels = []
        for col in (level_col, rate_col):
            med, half = params[col]
            channels.append(np.clip((df[col].to_numpy(float) - med) / half, -1.0, 1.0))
        values = np.column_stack(channels)
        targets = df[target_column].to_numpy(float)
        dates = df["date"].reset_index(drop=True)

        X, y, y_dates = [], [], []
        for end in range(lookback_days - 1, len(df)):
            X.append(values[end - lookback_days + 1 : end + 1])
            y.append(targets[end])
            y_dates.append(dates.iloc[end])
        out[name] = (np.asarray(X, float), np.asarray(y, float), pd.Series(y_dates))
    return out


# --------------------------------------------------------------------------
# End-to-end regressor (mirrors fit_tfim_qrc_regressor)
# --------------------------------------------------------------------------

def fit_rydberg_qrc_regressor(
    sequence_splits: dict[str, tuple[np.ndarray, np.ndarray, pd.Series]],
    *,
    config: RydbergQRCConfig,
    target: str,
    verbose: bool = False,
) -> RydbergQRCResult:
    X_train, y_train, _ = sequence_splits["train"]
    X_val, y_val, _ = sequence_splits["val"]
    X_test, y_test, _ = sequence_splits["test"]

    H_train = build_rydberg_feature_matrix(X_train, config, verbose=verbose)
    H_val = build_rydberg_feature_matrix(X_val, config, verbose=verbose)
    H_test = build_rydberg_feature_matrix(X_test, config, verbose=verbose)

    readout, scaler = fit_qrc_readout(H_train, y_train, config=config)
    train_pred = predict_qrc_readout(readout, scaler, H_train, config=config)
    val_pred = predict_qrc_readout(readout, scaler, H_val, config=config)
    test_pred = predict_qrc_readout(readout, scaler, H_test, config=config)

    return RydbergQRCResult(
        config=config,
        target=target,
        train_metrics=evaluate_volatility_forecast(y_train, train_pred),
        val_metrics=evaluate_volatility_forecast(y_val, val_pred),
        test_metrics=evaluate_volatility_forecast(y_test, test_pred),
        train_predictions=train_pred,
        val_predictions=val_pred,
        test_predictions=test_pred,
        train_features=H_train,
        val_features=H_val,
        test_features=H_test,
        readout=readout,
        feature_scaler=scaler,
    )


def summarize_rydberg_result(result: RydbergQRCResult) -> dict:
    row = asdict(result.config)
    row.update(
        {
            "model": "rydberg_temporal_reservoir_exact",
            "target": result.target,
            "train_n": len(result.train_predictions),
            "val_n": len(result.val_predictions),
            "test_n": len(result.test_predictions),
            "n_reservoir_features": result.train_features.shape[1],
        }
    )
    for split, metrics in [
        ("train", result.train_metrics),
        ("val", result.val_metrics),
        ("test", result.test_metrics),
    ]:
        row[f"{split}_rmse"] = metrics.rmse
        row[f"{split}_qlike"] = metrics.qlike
        row[f"{split}_mz_alpha"] = metrics.mz_alpha
        row[f"{split}_mz_beta"] = metrics.mz_beta
        row[f"{split}_mz_r2"] = metrics.mz_r2
    return row


# --------------------------------------------------------------------------
# Hardware feasibility
# --------------------------------------------------------------------------

def validate_aquila_feasibility(
    config: RydbergQRCConfig,
    hw: AquilaConstraints | None = None,
) -> dict:
    """Static feasibility report for an Aquila realization of this config.

    NOTE: constants reflect publicly documented specs and must be re-verified
    against current QuEra/Braket documentation before hardware submission.
    """
    hw = hw or AquilaConstraints()
    pre = precompute(config)
    pos = pre.positions

    diff = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt((diff**2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    min_spacing = float(dists.min())

    width = float(pos[:, 0].max() - pos[:, 0].min())
    height = float(pos[:, 1].max() - pos[:, 1].min())

    delta_lo = config.delta_center_rad_us - abs(config.delta_span_rad_us)
    delta_hi = config.delta_center_rad_us + abs(config.delta_span_rad_us)
    omega_hi = config.omega_base_rad_us * (1.0 + abs(config.omega_mod_frac))
    omega_lo = max(config.omega_base_rad_us * (1.0 - abs(config.omega_mod_frac)), 0.0)

    # Worst-case ramp time between plateaus under slew limits (hardware
    # waveforms cannot jump discontinuously).
    ramp_omega = (omega_hi - omega_lo) / hw.omega_slew_rad_us2
    ramp_delta = (delta_hi - delta_lo) / hw.delta_slew_rad_us2
    ramp_total = (config.anchor_count - 1) * max(ramp_omega, ramp_delta)

    checks = {
        "min_spacing_ok": min_spacing >= hw.min_spacing_um,
        "area_ok": width <= hw.area_width_um and height <= hw.area_height_um,
        "omega_range_ok": omega_hi <= hw.omega_max_rad_us,
        "delta_range_ok": max(abs(delta_lo), abs(delta_hi)) <= hw.delta_abs_max_rad_us,
        "total_time_ok": config.total_time_us <= hw.total_time_max_us,
        "time_with_ramps_ok": config.total_time_us + ramp_total <= hw.total_time_max_us,
        "shots_ok": (config.shots or 0) <= hw.max_shots_per_task,
    }
    return {
        "feasible": all(checks.values()),
        "checks": checks,
        "min_spacing_um": min_spacing,
        "array_width_um": width,
        "array_height_um": height,
        "delta_range_rad_us": (delta_lo, delta_hi),
        "omega_range_rad_us": (omega_lo, omega_hi),
        "estimated_ramp_overhead_us": ramp_total,
        "n_atoms": pre.n_atoms,
        "nn_interaction_slow_rad_us": float(
            C6_RAD_UM6_PER_US / config.spacing_slow_um**6
        )
        if config.geometry == "dual_chain"
        else float(C6_RAD_UM6_PER_US / config.chain_spacing_um**6),
        "verify_specs_note": "Re-verify Aquila limits against current QuEra/Braket docs.",
    }
