# Day-5 Static Nonlinear Probe

## Purpose

After the negative temporal ESN result, test whether order-invariant path extrema or cheap static nonlinear maps add information beyond D1.

This is a bounded mechanism-selection experiment, not a hyperparameter search.

## Models

- `D1`: locked geometric baseline.
- `D1_extrema_linear`: D1 plus Claude's fixed closest-to-relapse and closest-to-recovery features.
- `D1_static_quadratic`: degree-2 polynomial logistic model on an independent five-coordinate static representation.
- `D1_random_tanh_joint`: D1 plus a 32-dimensional fixed random tanh map; probabilities averaged over five predeclared seeds.
- `D1_random_tanh_offset`: honest D1 logit offset plus the same random tanh correction.

## Static coordinates

The nonlinear controls use:

- current day-5 return;
- distance to recovery barrier;
- distance to relapse barrier;
- closest approach to relapse during days 1–5;
- closest approach to recovery during days 1–5.

`barrier_width` is omitted from the nonlinear input because it is exactly the sum of the two barrier distances. It remains in D1 solely to reproduce the locked historical baseline.

## Extrema definitions

Let `r_d1 ... r_d5` be running returns from branch entry. Let

- `lower = endpoint - distance_to_relapse_barrier`;
- `upper = endpoint + distance_to_recovery_barrier`.

Then:

- `closest_to_relapse = min(r_d1 ... r_d5) - lower`;
- `closest_to_recovery = upper - max(r_d1 ... r_d5)`.

These features encode early barrier proximity but not temporal ordering.

## Evaluation

Identical to the locked D1 protocol:

- unresolved day-5 risk set;
- recovery versus relapse first within 120 days;
- post-1990 calendar-prequential evaluation;
- crisis-cluster purge using `landmark_date < cluster_start`;
- minimum training size 30;
- all, non-SPY, leave-Nikkei-out, and Nikkei-only slices;
- row-weighted and equal-cluster proper scores;
- matched-cohort D1 deltas for every comparator.

## Fixed nonlinear controls

Quadratic model:

- standardized inputs;
- degree 2, no bias term from the polynomial transformer;
- logistic `C=0.1`.

Random tanh map:

- 32 features;
- seeds 7, 42, 123, 1001, 2026;
- Gaussian weights scaled by inverse square root of input dimension;
- fixed uniform biases;
- no feature or seed selection;
- joint logistic `C=0.1`;
- offset correction L2 penalty 10.

## Stop rule

If neither the extrema model nor either cheap nonlinear control improves D1 consistently in matched log loss/Brier across all-market and non-SPY slices, stop the branch-target nonlinear reservoir lane.

A state-conditioned Rydberg feature map is justified only if at least one cheap static nonlinear model provides stable incremental evidence that D1 leaves exploitable nonlinear structure.

## Outputs

Default directory: `/tmp/qpitome_branch_static_probe`

- `predictions.csv`
- `summary_metrics.csv`
- `paired_score_deltas.csv`
- `cluster_weighted_metrics.csv`
- `run_manifest.json`
