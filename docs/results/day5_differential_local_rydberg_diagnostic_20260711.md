# Differential-Pair Local-Detuning Diagnostic — 2026-07-11

## Status

Passed. The revised ten-atom complementary-pair encoding has sufficient held-out nonlinear capacity to justify one frozen outer market evaluation.

## Test status

Nineteen focused tests passed in 3.22 seconds.

## Results

Mean held-out capacity by target class:

| Feature map | Branch-motivated | Linear | Product | Square |
|---|---:|---:|---:|---:|
| Raw linear coordinates | 0.6116 | 1.0000 | -0.0062 | -0.0064 |
| Random tanh, 55 features | 0.9374 | 0.9944 | 0.8098 | 0.8129 |
| Differential local Rydberg | 0.9219 | 0.9438 | 0.7828 | 0.6953 |

Feature effective rank:

| Feature map | Participation ratio | n95 |
|---|---:|---:|
| Raw linear coordinates | 3.825 | 5 |
| Random tanh, 55 features | 4.614 | 9 |
| Differential local Rydberg | 4.474 | 9 |

## Interpretation

The differential local-detuning map passes both predeclared gates:

1. it has strong held-out capacity for branch-motivated nonlinear targets;
2. it is competitive with a matched 55-feature random tanh map and does not collapse to a lower effective rank.

The Rydberg map trails the random tanh control modestly:

- branch-motivated capacity difference: -0.0155;
- product capacity difference: -0.0270;
- square capacity difference: -0.1176.

This is not evidence of quantum advantage. It is evidence that the revised physical encoding can represent the relevant nonlinear function classes and that the earlier eight-atom failure was architecture-specific rather than a general incapacity of local-detuning Rydberg features.

## Decision

Proceed to exactly one frozen outer market evaluation using:

- ten-atom chain;
- five complementary local-detuning pairs;
- 7.5 μm spacing;
- 0.55 μs evolution;
- global Omega = 6 rad/μs;
- global Delta = 6 rad/μs;
- local-detuning amplitude = 4 rad/μs;
- occupations and pair correlators;
- joint D1 + 55 Rydberg features;
- logistic C = 0.1.

No operating-point tuning is permitted from the outer market result. The earlier eight-atom local-detuning result remains separately documented as a negative architecture control.

## Reproducibility

Runner: `scripts/modeling/run_day5_differential_local_rydberg_diagnostic.py`

Protocol: `docs/protocols/day5_differential_local_rydberg_diagnostic.md`

Outputs:

- `target_capacities.csv`;
- `capacity_by_class.csv`;
- `feature_rank.csv`;
- `run_manifest.json`.
