# Day-5 Spatial Rydberg Results and Submission-Relevant Lessons

## Scope

This note records the completed static spatial Rydberg assay, the representation diagnostics, the residualized confirmation test, and the resulting bounded conclusions. It distinguishes exploratory findings from confirmation-grade findings.

The current experiments use the frozen local-detuning architecture and the current static financial inputs. They do **not** establish a general ceiling for all possible QRC inputs or architectures.

## 1. Standard assay: output and readout comparison

The standard assay compared occupations, raw pair expectations, connected correlations, combined blocks, foldwise PCA, protected offsets, joint readouts, and matched null controls.

Main findings:

- The original jointly fitted D1 + 55-feature Rydberg model was unstable and degraded prediction.
- Strong regularization and a frozen-D1 offset prevented most of that damage.
- Occupations were the least harmful real Rydberg block.
- Raw pairs, connected correlations, and combined blocks did not provide stable incremental signal.
- Connected correlations did not solve the redundancy problem.
- Foldwise PCA at ranks 3, 5, and 9 was approximately neutral rather than beneficial.
- A Gaussian nuisance control matched or exceeded the small apparent gain from occupations in the first protected-offset assay.
- Constant and duplicate-D1 controls showed that part of the apparent joint-model improvement came from changing D1 regularization rather than adding new information.

Bounded conclusion:

> Under the current inputs and local-detuning architecture, output-block selection and regularization can preserve D1, but the standard assay did not demonstrate a reliable incremental Rydberg signal.

## 2. PCA representation diagnostic

A separate foldwise reconstruction diagnostic measured how well Rydberg output principal components reproduced the D1 decision function.

Cumulative held-out D1-logit reconstruction:

| Output block | PCs | R² |
|---|---:|---:|
| occupations | 1 | 0.5621 |
| occupations | 2 | 0.8326 |
| occupations | 3 | 0.8709 |
| occupations | 10 | 0.9119 |
| all raw | 1 | 0.4833 |
| all raw | 2 | 0.7528 |
| all raw | 4 | 0.9086 |
| all raw | 15 | 0.9566 |
| occupations + connected | 1 | 0.3028 |
| occupations + connected | 4 | 0.8430 |
| occupations + connected | 8 | 0.9378 |
| occupations + connected | 15 | 0.9648 |

Interpretation:

- The dominant Rydberg output directions largely reconstruct the existing D1 decision geometry.
- Occupations are the most compact D1-like representation.
- Pair features add dimensions, but much of their capacity still reproduces D1 rather than adding orthogonal information.
- PCA preserved high-variance D1-like directions; it did not isolate complementary residual directions.

Submission-relevant lesson:

> Representation capacity is not the same as incremental information. The reservoir was expressive enough to reconstruct a strong classical baseline, but that alone did not improve prediction.

## 3. Residualized Rydberg discovery assay

To test for information beyond D1, each Rydberg output block was residualized against D1 within each historical fold. Training residuals were cross-fitted, the held-out cluster used a residualizer fitted on the full historical training set, and the D1 logit was frozen.

The exploratory 24-model screen found a best configuration:

- output block: occupations;
- residualizer: quadratic D1 basis;
- Ridge alpha: 10;
- frozen-D1 correction L2: 100.

Exploratory all-market deltas versus D1:

- delta log-loss: -0.0034;
- delta Brier: -0.0019;
- equal-cluster delta log-loss: -0.0027;
- equal-cluster delta Brier: -0.0016.

This was treated as discovery-grade because the candidate was selected after examining 24 configurations.

## 4. Strict cross-fitted D1-offset confirmation

The selected candidate was frozen and rerun with strictly historical cross-fitted D1 logits for correction-training rows. The outer held-out cluster retained the D1 model fitted on all prior data.

Confirmed score deltas:

| Evaluation group | Delta log-loss | Delta Brier | Equal-cluster delta log-loss | Equal-cluster delta Brier |
|---|---:|---:|---:|---:|
| all post-1990 purged | +0.0005 | -0.0005 | +0.0031 | +0.0008 |
| non-SPY validation | -0.0008 | -0.0007 | +0.0018 | +0.0004 |
| leave Nikkei out | +0.0049 | +0.0011 | +0.0092 | +0.0031 |
| Nikkei only | -0.0068 | -0.0031 | -0.0072 | -0.0030 |

Interpretation:

- The apparent global gain did not survive the clean stacking test.
- The earlier gain was largely attributable to mismatch between in-sample D1 logits used during correction training and out-of-sample D1 logits used during evaluation.
- The corrected overall result is effectively null, with equal-cluster metrics leaning unfavorable.
- Nikkei improves while non-Nikkei markets degrade. This heterogeneity is post hoc and exploratory; no market-specific tuning was pursued.

Confirmation-grade conclusion:

> For the current static inputs and local-detuning architecture, residualized occupations do not improve D1 overall after strict historical cross-fitting.

Recorded but not pursued:

> The Nikkei-only slice improved while the leave-Nikkei-out slice degraded. Revisit this discrepancy only if independent later architectures or input sets reproduce it.

## 5. What the null does and does not mean

The result is conditional on the current input representation.

It establishes:

- the current local-detuning map mostly transforms information already summarized by D1;
- improved output handling cannot create missing information;
- the tested residual correction is not generally predictive.

It does not establish:

- that all static inputs are exhausted;
- that all local-detuning encodings must fail;
- that spatial position encoding must fail;
- that temporal QRC cannot help;
- that no residual branch information exists in path-shape variables not currently supplied to the reservoir.

The next input-design question is therefore:

> Do explicitly non-D1 path-shape variables contain stable incremental branch information under the same grouped prequential protocol?

Candidate classes include extremum timing, approach-retreat asymmetry, dwell time near barriers, barrier revisits, path curvature, evolving volatility, and sequence order.

These should be tested classically before being encoded into another quantum architecture.

## 6. Hardware architecture note

Local detuning remains feasible, but current neutral-atom hardware considerations make position encoding comparatively attractive for a later architecture study:

- sample-specific atom geometry is a standard control channel;
- pairwise distances directly modify the interaction through the approximately inverse-sixth-power Rydberg interaction;
- local detuning is more constrained, using a shared temporal waveform with static site coefficients;
- local coefficient error can reduce effective feature precision;
- position encoding has its own sensitivity because distance error is amplified in the interaction strength.

This is architecture-selection evidence, not evidence of predictive advantage. It does not alter the interpretation of the completed local-detuning experiments.

## 7. Methodological contribution

The negative result remains submission relevant because the project developed and applied a standardized evaluation framework:

1. freeze input and architecture layers while testing observables and readouts;
2. use matched null controls;
3. protect the classical baseline with a frozen offset;
4. apply all transformations within historical training folds;
5. evaluate both row-weighted and equal-cluster proper scores;
6. separate exploratory screens from frozen confirmation tests;
7. retain null confirmations rather than promoting favorable exploratory estimates.

This framework exposed two otherwise misleading effects:

- redundant Rydberg features that mainly reconstructed D1;
- an apparent residual gain caused by train/test stacking mismatch.

That methodological finding is stronger and more defensible than claiming a quantum advantage unsupported by the data.

## 8. Reproducibility pointers

Primary scripts:

- `scripts/modeling/day5_branching/spatial_rydberg/run_day5_spatial_rydberg_assay_shard.py`
- `scripts/modeling/day5_branching/spatial_rydberg/merge_day5_spatial_rydberg_assay.py`
- `scripts/modeling/day5_branching/spatial_rydberg/run_day5_rydberg_pca_reconstruction.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_residualized_rydberg_shard.py`
- `scripts/modeling/day5_branching/residual_confirmation/merge_day5_residualized_rydberg.py`
- `scripts/modeling/day5_branching/residual_confirmation/run_day5_residualized_rydberg_crossfit_shard.py`
- `scripts/modeling/day5_branching/residual_confirmation/merge_day5_residualized_rydberg_crossfit.py`

Relevant frozen confirmation candidate:

- occupations;
- quadratic D1 residualization;
- Ridge alpha 10;
- frozen D1 offset;
- correction L2 100.
