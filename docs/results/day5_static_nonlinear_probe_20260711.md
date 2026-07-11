# Day-5 Static Nonlinear Probe — 2026-07-11

## Status

Completed. The result supports a state-conditioned nonlinear route and closes the temporal ESN lane for this target.

## Models

- `D1`: locked geometric baseline.
- `D1_extrema_linear`: D1 plus closest approach to relapse and recovery during days 1–5.
- `D1_static_quadratic`: compact degree-2 polynomial logistic model.
- `D1_random_tanh_joint`: D1 plus a fixed 32-dimensional random tanh feature map, averaged over five predeclared seeds.
- `D1_random_tanh_offset`: honest D1 logit offset plus the same random tanh correction.

## Main results

All post-1990 predictions:

| Model | n | AUC | PR AUC | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| D1 | 299 | 0.7194 | 0.7429 | 0.6146 | 0.2111 |
| D1 + extrema | 299 | 0.7373 | 0.7628 | 0.5984 | 0.2048 |
| Static quadratic | 299 | 0.7220 | 0.7403 | 0.6283 | 0.2120 |
| D1 + random tanh joint | 299 | 0.7313 | 0.7472 | 0.6023 | 0.2065 |
| D1 + random tanh offset | 287 | 0.7187 | 0.7241 | 0.6215 | 0.2119 |

Matched proper-score deltas versus D1, where negative is better:

| Model | All Δ log loss | Non-SPY Δ log loss | Leave-Nikkei-out Δ log loss | Nikkei-only Δ log loss |
|---|---:|---:|---:|---:|
| D1 + extrema | -0.0162 | -0.0121 | -0.0157 | -0.0170 |
| Static quadratic | +0.0137 | +0.0078 | +0.0215 | +0.0009 |
| D1 + random tanh joint | -0.0123 | -0.0087 | -0.0143 | -0.0090 |
| D1 + random tanh offset | +0.0037 | +0.0040 | +0.0066 | -0.0014 |

The equal-cluster results agree with the row-weighted results:

- extrema improves cluster-mean log loss and Brier in every slice;
- random tanh joint improves in every slice;
- quadratic worsens in every major slice;
- random tanh offset is not consistently useful.

## Interpretation

Supported conclusions:

1. The day-5 target contains reproducible information beyond the locked D1 geometry.
2. The fixed closest-approach extrema are the strongest tested addition.
3. Generic static nonlinear expansion can help: the random tanh joint model improves D1 across all reported slices.
4. A generic quadratic basis is not sufficient and tends to overfit or misrepresent the useful structure.
5. Residual/logit-offset learning is not automatically superior. For this static feature map, joint fitting is clearly better.
6. The useful signal is state-conditioned and order-invariant at the level tested here; temporal recurrence is not required.

Not established:

- independent confirmation of the extrema candidate, because the same overall dataset informed its discovery;
- quantum advantage;
- that a Rydberg map will outperform the random tanh control;
- that the signal is specifically quantum rather than generic nonlinear geometry.

## Decision

Proceed to one bounded memoryless/state-conditioned Rydberg feature-map experiment using the validated Rydberg reservoir implementation.

The primary comparison should be joint fitting, not offset correction:

- D1;
- D1 + extrema;
- D1 + random tanh joint;
- D1 + memoryless Rydberg features.

The Rydberg model must use the same static coordinates, the same purged prequential evaluation, and matched proper-score reporting. A temporal variant is not part of this branch experiment.

## Reproducibility

Runner: `scripts/modeling/run_cross_market_day5_static_nonlinear_probe.py`

Protocol: `docs/protocols/day5_static_nonlinear_probe.md`

Expected outputs:

- `predictions.csv`
- `summary_metrics.csv`
- `paired_score_deltas.csv`
- `cluster_weighted_metrics.csv`
- `run_manifest.json`
