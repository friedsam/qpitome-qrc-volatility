# Day-5 Local-Detuning Rydberg Probe — 2026-07-11

## Status

Completed. Negative result. The predeclared continuation criterion was not met, so this state-conditioned Rydberg lane is closed without tuning.

## Configuration

- eight atoms in a dual-chain geometry;
- five direct static coordinates plus three fixed contrast channels;
- sample-specific local-detuning spatial pattern;
- evolution time 0.55 μs;
- global Rabi amplitude 6 rad μs⁻¹;
- global detuning 6 rad μs⁻¹;
- local-detuning amplitude 4 rad μs⁻¹;
- 8 occupations + 28 pair correlators = 36 features;
- joint D1 + Rydberg logistic readout, `C=0.1`;
- exact-state simulation, no shots;
- post-1990 calendar-prequential, crisis-cluster purged evaluation.

All focused tests passed: 14 tests in 3.35 s.

## Results

| Group | Model | n | AUC | PR AUC | Log loss | Brier |
|---|---|---:|---:|---:|---:|---:|
| All post-1990 | D1 | 299 | 0.7194 | 0.7429 | 0.6146 | 0.2111 |
| All post-1990 | D1 + local Rydberg | 299 | 0.7147 | 0.7331 | 0.6244 | 0.2143 |
| Non-SPY | D1 | 254 | 0.7190 | 0.7259 | 0.6129 | 0.2112 |
| Non-SPY | D1 + local Rydberg | 254 | 0.7124 | 0.7213 | 0.6264 | 0.2158 |
| Leave-Nikkei-out | D1 | 186 | 0.7557 | 0.8129 | 0.5839 | 0.1978 |
| Leave-Nikkei-out | D1 + local Rydberg | 186 | 0.7533 | 0.8072 | 0.5881 | 0.1995 |
| Nikkei-only | D1 | 113 | 0.6761 | 0.6332 | 0.6652 | 0.2329 |
| Nikkei-only | D1 + local Rydberg | 113 | 0.6638 | 0.6215 | 0.6841 | 0.2385 |

Matched proper-score deltas versus D1, where positive is worse:

| Group | Δ log loss | Δ Brier | Equal-cluster Δ log loss | Equal-cluster Δ Brier |
|---|---:|---:|---:|---:|
| All post-1990 | +0.0097 | +0.0032 | +0.0093 | +0.0029 |
| Non-SPY | +0.0135 | +0.0046 | +0.0111 | +0.0037 |
| Leave-Nikkei-out | +0.0042 | +0.0017 | +0.0045 | +0.0017 |
| Nikkei-only | +0.0189 | +0.0056 | +0.0175 | +0.0052 |

The Rydberg model is worse than D1 on every reported discrimination and proper-scoring metric, in every market slice, under both row-weighted and equal-cluster summaries.

## Interpretation

Supported conclusion:

> Under the frozen native local-detuning architecture, the eight-atom Rydberg feature map does not add predictive information beyond the locked D1 geometry and instead degrades out-of-sample calibration and discrimination.

The result does not prove that every local-detuning architecture must fail. It does show that further tuning is not justified by the current evidence because:

- the static extrema model improves D1 consistently;
- the matched random-tanh joint model improves D1 consistently;
- the local-detuning Rydberg map fails consistently;
- the result is not confined to one market or one metric.

This indicates that the remaining exploitable structure is accessible to simple classical features and that the tested Rydberg representation is not aligned with it.

## Decision

- Do not tune global detuning, local-detuning amplitude, evolution time, geometry, or readout regularization on this outer evaluation.
- Do not add shots or hardware runs for this branch target.
- Do not revive the temporal Rydberg variant after the negative temporal ESN result.
- Retain the implementation and negative result as a documented control.
- Use `D1_extrema_linear` as the strongest current branch-direction model.

## Reproducibility

Protocol: `docs/protocols/day5_static_rydberg_probe.md`

Runner: `scripts/modeling/run_cross_market_day5_static_rydberg_probe.py`

Shared simulator: `src/qpitome_qrc/qrc/local_detuning_reservoir.py`

Expected outputs:

- `predictions.csv`;
- `summary_metrics.csv`;
- `paired_score_deltas.csv`;
- `cluster_weighted_metrics.csv`;
- `run_manifest.json`.
