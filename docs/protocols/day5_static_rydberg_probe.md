# Day-5 State-Conditioned Rydberg Probe

## Purpose

Test whether a physically generated, memoryless Rydberg feature map improves the locked D1 branch-direction model after:

- temporal ESN failed;
- fixed path extrema improved D1;
- a matched 32-dimensional random tanh map improved D1;
- a generic quadratic map failed.

The Rydberg model is not asked to recreate the first-passage physics. Its narrow role is to represent nonlinear state dependence left unresolved by D1.

## Static input state

Five independent coordinates are used:

1. day-5 return from branch entry;
2. distance to recovery barrier;
3. distance to relapse barrier;
4. closest approach to relapse during days 1–5;
5. closest approach to recovery during days 1–5.

`barrier_width` is excluded from the reservoir input because it is exactly the sum of the two barrier distances. It remains in D1 solely to reproduce the locked baseline.

## Why masks are required

The neutral-atom reservoir has two global input controls:

- detuning `Delta`;
- Rabi amplitude `Omega`.

A five-dimensional static state therefore cannot be injected independently in one global-control program. The probe uses four fixed masks. Each mask projects the five standardized coordinates into one `(level, rate)` pair, which is then mapped to `(Delta, Omega)` by the existing Rydberg encoder.

This is a static multi-mask feature map, not temporal market memory. Every mask starts from a fresh quantum state.

## Fixed mask design

- mask count: 4;
- mask seed: 20260711;
- Gaussian mask coefficients scaled by `1/sqrt(5)`;
- projected channels compressed with `tanh` to the encoder range;
- masks are fixed globally and never selected by performance.

## Frozen Rydberg configuration

Reuse `qpitome_qrc.qrc.rydberg_reservoir`:

- geometry: dual chain;
- atoms: 4 slow + 4 fast;
- slow spacing: 9 μm;
- fast spacing: 15 μm;
- row gap: 14 μm;
- memory mode: memoryless;
- observable mode: occupations only;
- four mask programs;
- effective evolution time per mask: 0.55 μs;
- total simulator waveform time field: 2.20 μs so the existing four-anchor implementation allocates 0.55 μs to each fresh-state mask;
- detuning center/span: 6/4 rad μs⁻¹;
- Rabi base/modulation: 6 rad μs⁻¹ / 0.5;
- exact expectation values, no shot noise.

Because only eight occupations are measured per mask, the final Rydberg feature dimension is `4 × 8 = 32`, matching the random-tanh control.

## Hardware interpretation and cost

The four masks correspond to four separate fresh-state programs per financial sample. They are not free repetitions of one program. A future hardware implementation must report:

- four waveform constructions per sample;
- shots per mask;
- total task count and queue/runtime cost;
- finite-slew waveform conversion;
- simulator-versus-hardware feature agreement.

This probe is exact-state simulation only.

## Evaluation

Use the same locked protocol:

- unresolved day-5 risk set;
- recovery before relapse within 120 days;
- post-1990 calendar-prequential evaluation;
- crisis-cluster purge: `landmark_date < cluster_start`;
- minimum training size 30;
- D1 oracle equivalence;
- all-market, non-SPY, leave-Nikkei-out, and Nikkei-only slices;
- row-weighted and equal-cluster log loss/Brier;
- matched-cohort deltas versus D1.

Scaling of the five static inputs is fitted separately inside each outer training fold. Future rows are transformed but never used to fit scaling or readout parameters.

## Models

- `D1`;
- `D1_plus_rydberg_joint`: D1 coordinates plus 32 Rydberg features in one regularized logistic model.

The primary readout uses logistic `C=0.1`, matching the successful random-tanh joint control. No offset version and no parameter sweep are run in this stage.

## Decision rule

A Rydberg continuation is justified only if the joint model improves D1 in matched log loss and Brier on both:

- all post-1990 predictions;
- non-SPY validation.

The result must also not collapse on both market-holdout diagnostics.

Interpretation relative to classical controls:

- worse than D1: close the state-conditioned Rydberg lane;
- better than D1 but no better than random tanh: useful physical feature map, no reservoir-specific advantage;
- comparable to or better than extrema and random tanh across proper scores: candidate for finite-shot and scaling studies, still not quantum advantage.

## Outputs

Default: `/tmp/qpitome_branch_rydberg_probe`

- `predictions.csv`;
- `summary_metrics.csv`;
- `paired_score_deltas.csv`;
- `cluster_weighted_metrics.csv`;
- `run_manifest.json`.
