# Day-5 Spatial Rydberg Standard Assay

## Four experimental layers

1. **Input features** — which financial state variables enter the quantum system.
2. **Quantum architecture** — geometry, controls, evolution, and interactions.
3. **Output observables** — occupations, raw correlations, connected correlations, and bottlenecks.
4. **Readout/optimization** — Rydberg-only, joint fitting, protected D1 offset, residualization, and regularization.

The current assay freezes layers 1 and 2 and studies layers 3 and 4 systematically.

## Frozen input and architecture

- five day-5 state coordinates;
- ten complementary local-detuning atom pairs;
- ten-atom chain, spacing 7.5 μm;
- evolution time 0.55 μs;
- global Omega = 6 rad/μs;
- global Delta = 6 rad/μs;
- local Delta = 4 rad/μs;
- exact-state simulation;
- no temporal memory;
- no shots.

## Output blocks

- `occupations`: 10 site occupations;
- `raw_pairs`: 45 raw pair expectations;
- `connected_pairs`: 45 connected correlators;
- `occ_plus_connected`: 55 features;
- `all_raw`: 55 occupations plus raw pair expectations;
- `pca3`, `pca5`, `pca9`: foldwise PCA of `occ_plus_connected`.

Connected correlators are

`<n_i n_j> - <n_i><n_j>`.

## Readout ladder

- `rydberg_only`: logistic model using only the selected Rydberg block;
- `joint`: D1 and the selected block optimized together;
- `offset`: D1 fitted first; its logit is frozen; only the Rydberg correction is optimized.

The primary offset penalties are `l2 = 10, 100, 1000`. Joint and Rydberg-only logistic models use the predeclared `C = 0.001, 0.01, 0.1` ladder.

## Null controls

- `gaussian55`: 55 fixed-seed Gaussian nuisance features;
- `duplicate_d1`: exact standardized duplicates of D1 variables;
- `permuted_all_raw`: real Rydberg features with deterministic within-training row permutation;
- `constant55`: constant columns, a pipeline oracle.

Nulls are evaluated under joint and protected-offset readouts where meaningful.

## Required outputs

- per-row predictions for every model;
- matched proper-score deltas versus D1;
- all-market and market-slice metrics;
- equal-cluster metrics;
- foldwise feature rank, participation ratio, and condition diagnostics;
- run manifest.

## Parallel execution

The expensive outer rows are deterministically sharded. Recommended six-shard allocation:

- Mac-2: shards 0, 1, 2;
- Mac-1: shards 3, 4;
- Mac-3: shard 5.

Every machine must use the same branch, commit, `--num-shards 6`, and distinct output directories. Shards are merged only after all six complete.

## Interpretation order

1. Compare null controls with the failed 55-feature joint model.
2. Determine whether occupations or correlations carry direction signal.
3. Determine whether connected correlations reduce redundancy.
4. Determine whether PCA rescues small-sample stability.
5. Determine whether a protected offset extracts incremental information without damaging D1.
6. Only then redesign inputs or quantum architecture.
