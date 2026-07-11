# Differential-Pair Local-Detuning Outer Probe — 2026-07-11

## Status

Completed. Negative outer result despite strong synthetic nonlinear-capacity diagnostics.

## Frozen configuration

- ten-atom chain;
- five complementary local-detuning pairs;
- 7.5 μm spacing;
- evolution time 0.55 μs;
- global Omega = 6 rad/μs;
- global Delta = 6 rad/μs;
- local-detuning amplitude = 4 rad/μs;
- 10 occupations + 45 pair correlators = 55 features;
- joint D1 + Rydberg logistic readout with C = 0.1;
- exact-state simulation;
- post-1990 calendar-prequential, crisis-cluster purged evaluation.

## Results

| Group | Model | n | AUC | PR AUC | Log loss | Brier |
|---|---|---:|---:|---:|---:|---:|
| All post-1990 | D1 | 299 | 0.7194 | 0.7429 | 0.6146 | 0.2111 |
| All post-1990 | D1 + differential local Rydberg | 299 | 0.7079 | 0.7220 | 0.6340 | 0.2173 |
| Non-SPY | D1 | 254 | 0.7190 | 0.7259 | 0.6129 | 0.2112 |
| Non-SPY | D1 + differential local Rydberg | 254 | 0.7074 | 0.7214 | 0.6325 | 0.2176 |
| Leave-Nikkei-out | D1 | 186 | 0.7557 | 0.8129 | 0.5839 | 0.1978 |
| Leave-Nikkei-out | D1 + differential local Rydberg | 186 | 0.7364 | 0.7770 | 0.6104 | 0.2063 |
| Nikkei-only | D1 | 113 | 0.6761 | 0.6332 | 0.6652 | 0.2329 |
| Nikkei-only | D1 + differential local Rydberg | 113 | 0.6626 | 0.6189 | 0.6728 | 0.2355 |

Matched deltas versus D1, where positive is worse:

| Group | Δ log loss | Δ Brier | Equal-cluster Δ log loss | Equal-cluster Δ Brier |
|---|---:|---:|---:|---:|
| All post-1990 | +0.0193 | +0.0062 | +0.0161 | +0.0056 |
| Non-SPY | +0.0196 | +0.0065 | +0.0153 | +0.0052 |
| Leave-Nikkei-out | +0.0265 | +0.0084 | +0.0235 | +0.0078 |
| Nikkei-only | +0.0076 | +0.0026 | +0.0055 | +0.0020 |

The model is worse than D1 on every discrimination and proper-scoring metric, in every market slice, under both row-weighted and equal-cluster evaluation.

## Interpretation

This result must be interpreted together with the preceding mechanism diagnostic:

- differential local Rydberg branch-motivated capacity: 0.9219;
- product capacity: 0.7828;
- square capacity: 0.6953;
- participation ratio: 4.474;
- n95: 9.

Therefore the failure is not a basic inability to represent nonlinear functions. It is a failure of statistical alignment and/or readout efficiency on the small market sample.

Supported conclusion:

> The tested differential local-detuning feature map has strong generic nonlinear capacity but its 55-feature joint readout does not extract stable incremental information beyond D1 from the available branch dataset.

Likely mechanisms requiring diagnosis:

- feature-to-target misalignment despite generic capacity;
- unstable high-dimensional readout relative to 299 outer observations and much smaller early training folds;
- occupations and all-pair correlators contributing different signal/noise profiles;
- collinearity and fold-dependent scaling;
- useful information concentrated in a small latent subspace;
- joint fitting perturbing the strong D1 solution rather than learning a protected residual correction.

## Decision

Do not declare the Rydberg research program closed.

Do not tune the physical operating point directly against these outer labels.

Next stage: training-only structure and readout diagnostics using the frozen differential feature map:

1. occupations-only versus pair-correlator-only versus combined features;
2. connected correlators rather than raw pair expectations;
3. foldwise effective rank and condition number;
4. PCA or supervised bottlenecks selected only within historical training data;
5. protected residual/offset readout versus joint logistic fitting;
6. coefficient and prediction stability across adjacent prequential folds;
7. intermediate-time observables after the readout bottleneck is understood.

The successful classical extrema and random-tanh controls remain the benchmark. No quantum advantage is supported.

## Reproducibility

Runner: `scripts/modeling/run_cross_market_day5_differential_local_rydberg_probe.py`

Diagnostic result: `docs/results/day5_differential_local_rydberg_diagnostic_20260711.md`

Expected outputs:

- `predictions.csv`;
- `summary_metrics.csv`;
- `paired_score_deltas.csv`;
- `cluster_weighted_metrics.csv`;
- `run_manifest.json`.
