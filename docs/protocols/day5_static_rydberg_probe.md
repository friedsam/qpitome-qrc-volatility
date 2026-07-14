# Day-5 Local-Detuning Rydberg Probe

## Purpose

Test whether a native site-resolved Rydberg feature map improves the locked D1 branch-direction model after:

- temporal ESN failed;
- fixed path extrema improved D1;
- a 32-dimensional random tanh map improved D1;
- a generic quadratic map failed.

The Rydberg model is not asked to recreate the first-passage physics. Its role is to represent nonlinear state dependence left unresolved by D1.

## Static input state

Five independent coordinates are used:

1. day-5 return from branch entry;
2. distance to recovery barrier;
3. distance to relapse barrier;
4. closest approach to relapse during days 1–5;
5. closest approach to recovery during days 1–5.

`barrier_width` is excluded from the Rydberg input because it is exactly the sum of the two barrier distances. It remains in D1 only to reproduce the locked baseline.

## Native local-detuning encoding

The five coordinates are standardized inside each outer training fold and mapped into an eight-site local-detuning pattern:

- sites 0–4: one coordinate per site;
- site 5: endpoint-versus-recovery-distance contrast;
- site 6: relapse-versus-recovery closest-approach contrast;
- site 7: combined barrier and extrema asymmetry.

All site coefficients are compressed to `[0, 1]` with a fixed tanh transform. No mask search or feature selection is performed.

The Hamiltonian is

\[
H/\hbar = \frac{\Omega}{2}\sum_i \sigma_i^x
- \Delta_g \sum_i n_i
- \Delta_\ell \sum_i h_i(x)n_i
+ \sum_{i<j}V_{ij}n_i n_j.
\]

Each financial sample defines one local-detuning spatial pattern and therefore one fresh-state program. This is a static nonlinear feature map, not temporal market memory.

## Frozen Rydberg configuration

Reuse the shared exact-state simulator through `qpitome_qrc.qrc.local_detuning_reservoir`:

- geometry: dual chain;
- atoms: 4 slow + 4 fast;
- slow spacing: 9 μm;
- fast spacing: 15 μm;
- row gap: 14 μm;
- evolution time: 0.55 μs;
- global Rabi amplitude: 6 rad μs⁻¹;
- global detuning: 6 rad μs⁻¹;
- local-detuning amplitude: 4 rad μs⁻¹;
- observables: 8 occupations and 28 pair correlators;
- feature dimension: 36;
- exact expectation values; no shot noise;
- no parameter sweep.

The Rydberg feature count is slightly larger than the prior 32-dimensional random-tanh control. Therefore any apparent gain must be interpreted against both the extrema model and the random-feature result; feature count alone is not evidence of a reservoir-specific advantage.

## Hardware interpretation

The local spatial pattern is sample-dependent. Samples are therefore separate programs, not free shot repetitions of a shared waveform. A future hardware study must report program count, shots, finite-slew conversion, runtime, and simulator-versus-hardware agreement.

This stage is exact-state simulation only.

## Evaluation

Use the locked day-5 protocol:

- unresolved day-5 risk set;
- recovery before relapse within 120 days;
- post-1990 calendar-prequential evaluation;
- crisis-cluster purge: `landmark_date < cluster_start`;
- minimum training size 30;
- D1 oracle equivalence;
- all-market, non-SPY, leave-Nikkei-out, and Nikkei-only slices;
- row-weighted and equal-cluster log loss/Brier;
- matched-cohort deltas versus D1.

Input scaling is fitted separately inside every outer training fold. Future rows never affect scaling or readout fitting.

## Models

- `D1`;
- `D1_plus_local_rydberg_joint`: D1 coordinates plus 36 local-detuning Rydberg features in a regularized logistic model.

The readout uses logistic `C=0.1`, matching the successful random-tanh joint control. No offset version is run in this stage.

## Decision rule

Continuation is justified only if the local-detuning model improves D1 in matched log loss and Brier on both:

- all post-1990 predictions;
- non-SPY validation.

Interpretation relative to existing controls:

- worse than D1: close this Rydberg lane;
- better than D1 but weaker than extrema/random tanh: useful physical map, no reservoir-specific advantage;
- comparable to or better than both across proper scores: candidate for finite-shot and scaling studies, still not quantum advantage.

## Outputs

Default: `results/modeling/day5_branching/static_rydberg/static_rydberg_probe`

- `predictions.csv`;
- `summary_metrics.csv`;
- `paired_score_deltas.csv`;
- `cluster_weighted_metrics.csv`;
- `run_manifest.json`.
