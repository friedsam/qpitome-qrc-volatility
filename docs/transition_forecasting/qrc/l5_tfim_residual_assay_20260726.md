# Existing Phase-2 TFIM on the corrected L5 residual — 2026-07-26

## Scope

This assay asks whether the project's existing six-qubit TFIM reservoir can transform the only admitted upstream signal—the last ten daily changes in log volatility—into a useful correction to the frozen causal MSE-HAR residual path.

Evaluation uses folds 4–6 for construction and folds 7–8 for untouched confirmation. Financial test rows are not used. The readout has no intercept, no constant feature, no QLIKE offset, and no post-hoc calibration.

## Reservoir

The quantum physics are inherited from the canonical Phase-2 TFIM:

- six exact-state qubits;
- full disordered ZZ graph, coupling scale 0.7;
- transverse X field 0.5;
- evolution time 0.5 per anchor;
- three Trotter steps and three virtual nodes per anchor;
- ten ordered scalar anchors;
- RY input injection on qubit 0;
- six Z, six X, and five nearest-neighbor ZZ observables at every virtual node.

The resulting feature matrix has 510 raw columns. Fourteen train-constant columns are removed without using outcomes, leaving 496 columns. A train-standardized multi-output Ridge predicts all ten residual horizons with `fit_intercept=False`.

## Integrity and non-flatness

Across folds, the retained TFIM feature matrix has effective rank about 18.7–23.8. Median raw feature standard deviation is about 0.026–0.037. Manual matrix multiplication reproduces every correction exactly, and zero standardized features produce zero correction exactly.

The reservoir therefore is not literally flat. The relevant failure is task alignment and out-of-fold generalization.

## Alpha screen

The historical readout alpha 1,000 substantially harms confirmation fold 8. Increasing alpha shrinks the correction. Alpha 30,000 gives a small selection-fold compromise, but the confirmation result remains inconsistent. Extending the screen through alpha 1,000,000 only drives the correction toward zero and does not reverse the fold-8 failure.

## Confirmation result through horizon 5

| Fold | Population | Delta QLIKE | Delta RMSE | Mean correction |
|---:|---|---:|---:|---:|
| 7 | control | -0.000546 | +0.000185 | +0.000579 |
| 7 | transition | -0.010235 | -0.002659 | +0.001056 |
| 8 | control | +0.000281 | +0.000114 | -0.001519 |
| 8 | transition | +0.015071 | +0.001755 | +0.001955 |

Negative loss deltas indicate improvement. The ordered TFIM improves both transition losses in fold 7 and worsens both in fold 8.

The matched raw rate-10 Ridge improves transition QLIKE and RMSE in both confirmation folds. The TFIM therefore weakens and destabilizes the admitted classical signal.

## Controls

- The reversed TFIM achieves a larger fold-8 QLIKE improvement than the ordered TFIM while slightly worsening fold-8 RMSE.
- The shuffled TFIM also fails to establish a chronology-specific advantage.
- The ZZ-interaction-off TFIM is generally worse, showing that interactions change the features, but the interacting model still does not yield a stable useful correction.
- Two-channel level/rate and six-channel derived-state adapters do not rescue the TFIM.

## Attribution

Pooled folds 7–8 validation-row permutations (512 iterations) and training-residual permutations (256 refits) fail to reject the null for:

- correction–residual correlation;
- positive transition correction;
- transition-minus-control correction gap;
- transition QLIKE improvement;
- transition RMSE improvement.

All one-sided p-values exceed 0.17.

## Decision

The existing TFIM produces a nontrivial, moderately high-rank quantum feature map, but it does not produce a fold-stable or attributable L5 crisis-warning correction. It does not beat the raw ten-rate classical readout and does not rescue the submission model.
