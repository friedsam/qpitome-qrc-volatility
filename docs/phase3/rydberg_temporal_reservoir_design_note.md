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
