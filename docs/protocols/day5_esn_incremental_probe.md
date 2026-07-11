# Day-5 ESN Incremental Probe

## Status

Exploratory, bounded mechanism-selection experiment. This is not yet a promoted benchmark or final challenge result.

## Scientific question

Does a short-path echo-state network add reproducible information beyond the locked day-5 geometric baseline (D1) for first-passage recovery versus relapse?

The experiment deliberately separates three questions:

1. Can an ESN predict the branch outcome directly?
2. Can ESN features improve a joint model containing D1?
3. Can ESN features improve an offset model in which D1 retains the solved component and the ESN learns only an incremental log-odds correction?

Residual correction is a hypothesis, not an assumed advantage. It may perform worse if D1 leaves mostly noise.

## Locked target and evaluation geometry

- Landmark: day 5 after branch entry.
- Risk set: episodes unresolved at day 5.
- Outcome: recovery barrier reached before relapse barrier within the existing 120-day first-passage horizon.
- Outer evaluation: calendar-prequential and crisis-cluster purged.
- Training rule for test episode `i`: `landmark_date < cluster_start_i`.
- Evaluation starts on 1990-01-01.
- Minimum training size: 30 episodes.

These rules must match `run_cross_market_day5_confirmatory_baselines.py` exactly.

## Frozen baseline

D1 uses the existing four-column representation:

- `current_return_5d_from_branch`
- `distance_to_recovery_barrier`
- `distance_to_relapse_barrier`
- `barrier_width`

The three distance variables are algebraically redundant because
`distance_to_recovery_barrier + distance_to_relapse_barrier = barrier_width`.
This representation is retained only to reproduce the locked historical baseline. It must not be interpreted as four independent coordinates.

Before interpreting ESN results, the runner must reproduce the locked D1 prediction file. Any mismatch is an implementation failure, not a scientific result.

## Temporal inputs

Five causal steps, days 1 through 5, with four channels at each step:

- signed return divided by local volatility;
- downside shock pressure;
- change in log short/medium realized-volatility ratio;
- drawdown repair divided by local volatility.

The endpoint return is not added as a fifth sequence channel because D1 already contains the day-5 endpoint return.

## ESN mechanics

The probe reuses `qpitome_qrc.baselines.numpy_esn`:

- state reset to zero for each episode;
- tanh recurrent update;
- final reservoir state plus final input as features;
- input scaling fit only on the outer training set;
- fixed seed ensemble; no seed selection.

Primary exploratory configuration:

- 64 reservoir nodes;
- spectral radius 0.9;
- input scale 1.0;
- leak 1.0;
- seeds 7, 42, 123, 1001, 2026.

A single 32-node sensitivity configuration exists but is not run unless the primary result warrants it.

## Models

- `D1`: locked logistic baseline.
- `ESN_direct`: temporal ESN features only.
- `D1_plus_ESN_joint`: D1 and ESN features fitted jointly.
- `D1_plus_ESN_offset`: fixed D1 logit plus a regularized ESN correction.

For offset training, each historical training row receives a D1 probability produced by its own earlier, cluster-purged prequential fit. In-sample D1 probabilities are forbidden.

## Metrics

Primary:

- log loss;
- Brier score.

Secondary:

- ROC AUC;
- PR AUC.

Report both row-weighted and equal-cluster mean log loss/Brier for:

- all post-1990 predictions;
- non-SPY validation;
- leave-Nikkei-out evaluation;
- Nikkei-only diagnostic.

## Stop rule

Stop ESN exploration after the primary run if neither the joint nor offset model improves D1 consistently in proper scoring rules across the all-market and non-SPY slices.

Only if an ESN model is promising:

1. run a small number of fixed joint multichannel path permutations;
2. test whether ordered paths outperform permuted paths;
3. compare against compact quadratic and matched random-feature controls.

Do not begin a broad hyperparameter search. Do not promote D1X or alter the locked D1 baseline inside this experiment.

## Inputs

- `results/baselines/cross_market_day5_direction_v1/day5_landmark_frame.csv`
- portability raw data already used by the cross-market branch pipeline;
- `results/diagnostics/cross_market_crisis_clusters_v2/branch_sync_cluster_detail.csv`
- locked D1 predictions in `results/baselines/cross_market_day5_confirmatory_v1/purged_calendar_prequential_predictions.csv`

## Outputs

Exploratory outputs default to `/tmp/qpitome_branch_esn_probe`:

- `predictions.csv`
- `summary_metrics.csv`
- `cluster_weighted_metrics.csv`
- `run_manifest.json`

No generated output is committed unless the experiment is promoted after review.

## Promotion criteria

Promotion requires all of the following:

- exact D1 oracle equivalence;
- passing focused tests;
- no leakage in scaling or offsets;
- reproducible seed ensemble;
- proper-scoring improvement that is not confined to a single market slice;
- a concise interpretation that distinguishes direct prediction, incremental correction, temporal ordering, and generic nonlinearity.
