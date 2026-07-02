# Phase 3: True Temporal Rydberg Reservoir — Design Note

**Status:** instrument validated; no performance claims. Results live in
`scratch/` until promoted. This replaces the static response-grid surrogate,
which is parked as a negative result (it was a memoryless 1D nonlinearity of
one scalar — equivalent to a spline basis — and cannot demonstrate anything
about Rydberg reservoirs).

## What it is

`src/qpitome_qrc/qrc/rydberg_reservoir.py` +
`scripts/run_phase3_rydberg_temporal_reservoir.py` +
`tests/test_rydberg_reservoir.py`

Exact-statevector simulation of the Aquila-class AHS Hamiltonian
(global Omega(t), Delta(t); vdW n_i n_j couplings from geometry).
For each date, ONE state is evolved through a piecewise-constant waveform
encoding the trailing market window, so anchors interact through the quantum
dynamics — the true temporal analog of the Phase 2 TFIM-QRC under
global-drive-only hardware constraints.

- **Level x rate encoding (Bell-model port):** stress level -> Delta(t),
  stress rate -> Omega(t). Two market channels on two physical global controls.
- **Multi-timescale geometry:** dual chain, tight "slow" sublattice
  (blockaded, long memory) + loose "fast" sublattice; cross-sublattice
  n_i n_j correlators are the Rydberg analog of Phase 2 level-rate ZZ.
- **Hardware-native readout only:** occupations + all-pairs correlators
  (Z-basis projective); anchor trajectory features = truncated re-runs on
  hardware (shot cost linear in anchors). Optional multinomial shot noise.
- **Built-in ablations:** `memory_mode="memoryless"` (fresh state per anchor
  — the honest surrogate), `shuffle_anchors=True` (temporal-order control),
  `omega_mode="constant"` (level-only).
- **`validate_aquila_feasibility`** checks spacing/area/Omega/Delta/time/
  slew-ramp budget/shots. Constants must be re-verified against current
  QuEra/Braket docs before hardware submission.

Physics tests: single-atom Rabi vs analytic, blockade suppression of
double excitation, unitarity, shot-noise convergence, control semantics.

## Key finding: the phase budget controls generalization

With per-segment accumulated phases far above unity
(V_nn*t_seg ~ 43 rad, Omega*t_seg ~ 5.4 rad — "hot" regime), the reservoir
fits train and collapses out of sample, and the memoryless control BEATS the
temporal reservoir (unitary scrambling: the input->feature map becomes
hyper-oscillatory, so nearby inputs decorrelate). Reducing the budget to
order unity per anchor (spacing 9 um, t_seg 0.3 us, Omega ~ 6,
delta span 4 => V*t ~ 3 rad, Omega*t ~ 1.8 rad) restored the expected
ordering in smoke runs: temporal > memoryless and temporal > shuffled at q95.
This matches the Phase 2 TFIM working point (coupling*t ~ 0.35 rad) and is
the Rydberg design rule going forward. Defaults encode the gentle regime.

## Honest current result (single-split, full data, defaults, no tuning)

Raw level/rate baseline still wins q90/q95 on the fixed test window;
temporal does not yet beat memoryless there. Do NOT interpret single-window
tail metrics (~10 effective independent observations); conclusions require
the purged walk-forward CV across all crisis episodes.

## Next steps

1. Sweep the phase-budget neighborhood (delta span/center, Omega, t_seg,
   spacings) under walk-forward CV, not the single split.
2. Add ESN / RFF-of-same-inputs as external baselines (repo already has ESN).
3. Shot-noise budget curve (feature quality vs shots) before any Aquila runs.
4. Hardware: same waveforms, closed-simulator vs open-hardware comparison =
   the dissipation-as-feature test.

---

## Update: temporal-memory diagnostic + Landau-Zener ramp encoding

### Cross-time nonlinear capacity diagnostic (CONCLUSIVE)

`scripts/run_phase3_temporal_memory_diagnostic.py` implements the IPC-style
probe (Dambre et al. 2012): synthetic i.i.d. anchor inputs, out-of-sample
capacity for degree-1 recall, same-time nonlinearity P2(u_k), and cross-time
products u_a*u_b. Memoryless per-anchor features + linear readout form an
additive model with PROVABLY zero cross-time capacity; measured results
confirm every prediction (v1, Omega constant so Delta is the only channel):

  variant             cross-time capacity vs separation
  classical_additive  0 everywhere (analytic zero reference)
  plateau_memoryless  0 everywhere (confirmed numerically)
  ramp_memoryless     0.73 at separation 1, ZERO beyond (sees adjacent pair)
  plateau_temporal    0.77 -> 0.07 decaying fading-memory kernel
  plateau_shuffled    present but scrambled vs separation (order broken)
  classical_products  1.0 ceiling

Defensible sentence: the temporal Rydberg reservoir has substantial
out-of-sample capacity for cross-time nonlinear functions of its inputs,
which the memoryless control provably and measurably lacks. This also
explains the mixed market-side memoryless comparison: shuffled retains
cross-time capacity (misordered), so order-sensitivity is the cleaner
market ablation — and temporal beats shuffled in every valid fold.

### Landau-Zener ramp encoding

`encoding="ramp"`: Delta(t) piecewise-LINEAR through anchor level values;
stress rate = sweep slope, sensed natively via diabatic (Landau-Zener)
transitions. Validated against the analytic LZ formula at three sweep rates
(test_ramp_encoding_matches_landau_zener). Recommended omega_mode="constant".

Market-side status (stride-subsampled walk-forward, NOT final):
- At plateau-tuned parameters (delta 6+/-4) the sweep never crosses the
  transition region -> LZ channel inactive -> ramp underperforms.
- Re-centered to cross the transition (delta 3+/-6, 1.1 us) the control
  ordering becomes correct (temporal > memoryless ~= shuffled) but median
  q95 AP remains below the plateau+Omega-rate variant.
- Conclusion: ramp needs its own validation-selected parameter sweep
  (delta window placement relative to the many-body crossing region, total
  time, Omega); do not compare encodings at shared parameters.

### Walk-forward additions

`run_phase3_rydberg_purged_walkforward.py` now supports variants:
`rydberg_ramp`, `rydberg_ramp_memoryless`, `rydberg_ramp_shuffled`, and
`raw_products` (classical anchor-product-augmented baseline: if temporal only
matches it, reservoir memory is effectively 2nd order and classically
replicable; exceeding it evidences higher-order many-body memory).

## Dual-chain geometry diagnostic

A dual-chain diagnostic was added to test whether the proposed geometry actually induces two distinct dynamical regimes. The diagnostic evolves synthetic level/rate probes through the same temporal Rydberg reservoir and measures mean occupation and connected correlations separately for the slow chain, fast chain, and cross-chain pairs.

The result supports the intended geometry interpretation. The slow chain develops much stronger connected correlations than the fast chain:

- mean within-slow |connected correlation| = 0.0210
- mean within-fast |connected correlation| = 0.0025
- slow / fast connected-correlation ratio = 8.39

Cross-chain connected correlations are nonzero but weaker:

- mean cross slow-fast |connected correlation| = 0.00184
- cross / within connected-correlation ratio = 0.156

The occupation trajectories also separate late in the evolution: the slow chain rises and saturates, while the fast chain rises, peaks, and then relaxes downward. This supports the interpretation of the reservoir as a weakly coupled two-timescale system rather than a homogeneous atom array.

This diagnostic is mechanistic rather than predictive: it validates the physical role of the dual-chain geometry, but it does not by itself establish market-side predictive advantage.

## Ramp encoding check

The reservoir also supports Landau-Zener-style ramp encoding, where each segment evolves under a linear detuning ramp rather than a piecewise-constant plateau. This is a more explicitly hardware-native mechanism for rate sensitivity, because the sweep rate through detuning space can produce diabatic excitation differences.

A frozen-parameter purged walk-forward check was run with the same timing and anchor order as the main plateau result:

- total_time_us = 0.55
- anchors = 8
- anchor_policy = even
- reverse_anchors = true
- encoding = ramp

Median test metrics for ramp temporal were:

- q90 AUC = 0.806
- q90 AP = 0.407
- q90 F1 = 0.349
- q95 AUC = 0.665
- q95 AP = 0.197
- q95 F1 = 0.165

For comparison, the main plateau temporal configuration achieved q90 AP = 0.440 and q95 AP = 0.212. Thus ramp encoding is viable and produces comparable AUC/F1 behavior, but under the frozen tt055/a8/reverse setting it does not improve rare-event AP over plateau encoding.

Ramp encoding is therefore treated as a mechanistic extension and control, while plateau temporal encoding remains the primary market result.

