# Differential-Pair Local-Detuning Diagnostic

## Purpose

Evaluate a revised local-detuning encoding before any further use of the outer day-5 market evaluation.

The previous eight-atom encoding failed. Its main architectural weakness was asymmetric compression of each signed variable onto a single site plus three hand-built contrast sites. The revised encoding assigns one complementary atom pair to each of the five independent state coordinates.

## Encoding

For standardized coordinate `z_i`:

- positive-site coefficient: `(1 + tanh(z_i)) / 2`;
- negative-site coefficient: `(1 - tanh(z_i)) / 2`.

Each pair has constant total local-detuning weight equal to one. Sign is carried by the within-pair imbalance rather than common-mode amplitude.

Five coordinates require ten atoms:

1. day-5 endpoint return;
2. recovery-barrier distance;
3. relapse-barrier distance;
4. closest approach to relapse;
5. closest approach to recovery.

## Diagnostic only

This stage uses synthetic correlated five-dimensional states. It does not read market outcomes and does not touch the 299-row outer evaluation.

Feature maps:

- raw five coordinates;
- fixed 55-dimensional random tanh map;
- ten-atom differential local-detuning Rydberg map.

The 55-dimensional classical map matches the Rydberg observable count:

- 10 occupations;
- 45 pair correlators.

## Frozen Rydberg configuration

- ten-atom chain;
- spacing 7.5 μm;
- evolution time 0.55 μs;
- global Rabi amplitude 6 rad μs⁻¹;
- global detuning 6 rad μs⁻¹;
- local-detuning amplitude 4 rad μs⁻¹;
- exact expectations;
- no shots;
- no operating-point search in the first run.

## Targets

Held-out linear Ridge readouts reconstruct:

- each coordinate;
- each centered square;
- all ten pairwise products;
- extrema asymmetry;
- endpoint × barrier asymmetry;
- closest-barrier soft minimum;
- one mixed branch-motivated nonlinear score.

Capacity is `1 - MSE / Var` on an independent test set.

Also report feature participation ratio and the number of singular directions needed for 95% variance.

## Decision rule

Do not proceed to another outer market test unless the differential Rydberg map demonstrates both:

1. nontrivial held-out capacity for branch-motivated nonlinear targets;
2. capacity competitive with the matched random tanh map on at least some target class, without collapsing to a very low effective rank.

If it fails this diagnostic, stop. If it passes, any operating-point choice must be made using synthetic diagnostics or nested training data only, then frozen before the outer market evaluation.

## Outputs

Default directory: `/tmp/qpitome_day5_differential_rydberg_diagnostic`

- `target_capacities.csv`;
- `capacity_by_class.csv`;
- `feature_rank.csv`;
- `run_manifest.json`.
