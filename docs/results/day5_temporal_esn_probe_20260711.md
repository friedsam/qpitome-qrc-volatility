# Day-5 Temporal ESN Probe — 2026-07-11

## Status

Completed primary configuration. The predeclared stop rule was met: temporal ESN exploration stops here unless a future result provides new positive evidence.

## Configuration

- Reservoir nodes: 64
- Spectral radius: 0.9
- Input scale: 1.0
- Leak: 1.0
- Seeds: 7, 42, 123, 1001, 2026
- Inputs: five ordered daily steps, four causal channels per step
- Evaluation: post-1990 calendar-prequential, crisis-cluster purged
- Baseline: locked D1 geometry

## Primary results

All post-1990 predictions:

| Model | n | AUC | PR AUC | Log loss | Brier |
|---|---:|---:|---:|---:|---:|
| D1 | 299 | 0.7194 | 0.7429 | 0.6146 | 0.2111 |
| ESN direct | 299 | 0.7039 | 0.7284 | 0.6335 | 0.2191 |
| D1 + ESN joint | 299 | 0.7065 | 0.7417 | 0.6345 | 0.2193 |
| D1 + ESN offset | 287 | 0.7066 | 0.7218 | 0.6551 | 0.2215 |

The direct and joint ESN models are worse than D1 across all major slices and both row-weighted and equal-cluster proper scores. The small leave-Nikkei-out PR-AUC increase for the joint model is not persuasive because AUC, log loss, and Brier all worsen.

The offset model initially produced 287 predictions because honest prequential D1 offsets are unavailable for the earliest warm-up rows. The runner now writes matched-cohort D1 comparisons to `paired_score_deltas.csv`; those values must be used for the final residual comparison.

## Interpretation

Supported statement:

> The tested five-step temporal ESN does not add stable predictive information beyond day-5 first-passage geometry.

Not supported:

- that all temporal information is absent;
- that every ESN architecture must fail;
- that the negative result transfers automatically to unrelated targets.

Given the repeated temporal failures elsewhere in the project, a broader ESN search is not justified.

## Decision

- Do not run the 32-node sensitivity configuration.
- Do not run temporal permutation tests.
- Do not tune ESN hyperparameters.
- Do not pursue temporal QRC on this branch target.
- Proceed to a bounded static nonlinear probe using the fixed extrema candidate, a compact quadratic control, and matched random tanh features.

## Reproducibility

Runner: `scripts/modeling/run_cross_market_day5_esn_probe.py`

Protocol: `docs/protocols/day5_esn_incremental_probe.md`

Expected output files:

- `predictions.csv`
- `summary_metrics.csv`
- `paired_score_deltas.csv`
- `cluster_weighted_metrics.csv`
- `run_manifest.json`
