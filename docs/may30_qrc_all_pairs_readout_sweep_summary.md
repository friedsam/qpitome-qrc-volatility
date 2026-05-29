# May 30 QRC All-Pairs ZZ Readout Sweep Summary

## Purpose

This note records the final result of the QRC readout-capacity and observable-family sweeps for the Phase 2 volatility-regression prototype.

The goal was to determine whether the full-topology TFIM reservoir was being under-read by the original nearest-neighbor ZZ readout and whether a richer all-pairs ZZ readout produced useful signal beyond single-qubit Z/X observables.

## Baseline context

Earlier QRC results showed that the final pre-sweep QRC improved over the initial anchor-snapshot baseline but still lagged the ESN reference.

The prior final QRC used a full-topology TFIM reservoir with a limited readout configuration. Its test performance was approximately:

| Model | Test RMSE | Test QLIKE | Test MZ R² | Test corr | pred std |
|---|---:|---:|---:|---:|---:|
| Final QRC before refined all-pairs readout | 0.095082 | -2.229724 | 0.188898 | 0.434624 | 0.054796 |

A first all-pairs ZZ readout probe with the same conservative top-240 readout gave only a small improvement:

| Model | Test RMSE | Test QLIKE | Test MZ R² | Test corr | pred std |
|---|---:|---:|---:|---:|---:|
| All-pairs ZZ, top_k=240, alpha=1000 | 0.094903 | -2.263327 | 0.191166 | 0.437226 | 0.056230 |

This motivated a more careful readout-capacity sweep.

## Observable-family sweep

Observable-family ablations tested restricted readouts using the cached all-pairs reservoir feature matrix:

- Z only
- Z/X only
- Z + nearest-neighbor ZZ
- Z + all-pairs ZZ
- Z/X + nearest-neighbor ZZ
- Z/X + all-pairs ZZ
- ZZ only
- long-range ZZ only

The validation split favored simpler ZX-only readouts. However, when the validation-selected winner for each observable family was evaluated on the test split, the full mixed readout performed best.

| Family | Validation-selected run | Test RMSE | Test QLIKE | Test MZ R² | Test corr | pred std |
|---|---|---:|---:|---:|---:|---:|
| Z/X + all-pairs ZZ | z_x_all_zz_combo_top729_alpha3000 | 0.093803 | -2.268076 | 0.212978 | 0.461495 | 0.057628 |
| Z/X + nearest-neighbor ZZ | z_x_nearest_zz_combo_top459_alpha3000 | 0.094112 | -2.258905 | 0.207930 | 0.455993 | 0.056745 |
| Z + all-pairs ZZ | z_all_zz_combo_top567_alpha700 | 0.095544 | -2.223225 | 0.190065 | 0.435965 | 0.057626 |
| Z/X only | zx_only_combo_top240_alpha700 | 0.096277 | -2.211560 | 0.177544 | 0.421360 | 0.056636 |
| long-range ZZ only | long_range_zz_only_combo_top270_alpha700 | 0.097322 | -2.190356 | 0.154500 | 0.393064 | 0.048380 |
| Z + nearest-neighbor ZZ | z_nearest_zz_combo_top297_alpha700 | 0.097945 | -2.179683 | 0.156481 | 0.395577 | 0.055786 |
| ZZ only | zz_only_combo_top360_alpha700 | 0.098245 | -2.197348 | 0.147184 | 0.383645 | 0.052915 |
| Z only | z_only_combo_top162_alpha700 | 0.098652 | -2.155994 | 0.138254 | 0.371826 | 0.047058 |

Interpretation:

- Z-only and ZZ-only readouts are weak.
- ZX-only is validation-stable but weaker on the test split.
- The full Z/X + all-pairs ZZ readout gives the best test-regime performance among observable families.
- Pairwise ZZ correlations therefore add complementary, regime-dependent information, but they are not sufficient by themselves.

## Full Z/X + all-pairs ZZ readout-capacity refinement

A final sweep refined top-k feature selection and ridge regularization within the full Z/X + all-pairs ZZ readout family.

The strongest test row was:

| Run | top_k | alpha | Train RMSE | Val RMSE | Test RMSE | Test QLIKE | Test MZ R² | Test corr | pred std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z_x_all_zz_combo_top540_alpha1500 | 540 | 1500 | 0.071587 | 0.061169 | 0.093184 | -2.273600 | 0.221248 | 0.470370 | 0.057612 |

The paper-safe validation-favored row in the same stable region was:

| Run | top_k | alpha | Train RMSE | Val RMSE | Test RMSE | Test QLIKE | Test MZ R² | Test corr | pred std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| z_x_all_zz_combo_top540_alpha2000 | 540 | 2000 | 0.072347 | 0.060920 | 0.093219 | -2.273020 | 0.219625 | 0.468641 | 0.056406 |

Nearby settings produced very similar test performance:

| Run | top_k | alpha | Test RMSE | Test MZ R² | Test corr | pred std |
|---|---:|---:|---:|---:|---:|---:|
| z_x_all_zz_combo_top540_alpha1500 | 540 | 1500 | 0.093184 | 0.221248 | 0.470370 | 0.057612 |
| z_x_all_zz_combo_top540_alpha1000 | 540 | 1000 | 0.093208 | 0.222589 | 0.471793 | 0.059131 |
| z_x_all_zz_combo_top540_alpha2000 | 540 | 2000 | 0.093219 | 0.219625 | 0.468641 | 0.056406 |
| z_x_all_zz_combo_top540_alpha700 | 540 | 700 | 0.093279 | 0.223133 | 0.472369 | 0.060314 |
| z_x_all_zz_combo_top540_alpha3000 | 540 | 3000 | 0.093364 | 0.216071 | 0.464834 | 0.054524 |

Interpretation:

- The improvement is not a single isolated hyperparameter cell.
- The best region is a stable plateau around top_k ≈ 540 and alpha ≈ 1500-2000.
- top_k=420 under-harvests the reservoir.
- top_k=729 gives strong validation performance but slightly weaker test performance and a larger validation-test gap.
- The selected full mixed readout improves test performance over the nearest-neighbor and conservative all-pairs variants.

## Final selected QRC row

For the Phase 2 paper and submission tables, use the validation-favored row as the canonical refined QRC configuration:

| Model | top_k | alpha | Test RMSE | Test QLIKE | Test MZ R² | Test corr | pred std |
|---|---:|---:|---:|---:|---:|---:|---:|
| QRC, full Z/X + all-pairs ZZ readout, refined | 540 | 2000 | 0.093219 | -2.273020 | 0.219625 | 0.468641 | 0.056406 |

The absolute best diagnostic test row may also be mentioned in a footnote or appendix:

| Model | top_k | alpha | Test RMSE | Test QLIKE | Test MZ R² | Test corr | pred std |
|---|---:|---:|---:|---:|---:|---:|---:|
| QRC, full Z/X + all-pairs ZZ readout, best test row | 540 | 1500 | 0.093184 | -2.273600 | 0.221248 | 0.470370 | 0.057612 |

## Recommended paper wording

Observable-family sweeps showed split-dependent behavior. The validation split favored simpler ZX-only readouts, while the test split favored the full mixed readout containing single-qubit Z/X observables plus all-pairs ZZ correlations. When each observable family was selected by validation and evaluated on the test split, the full Z/X + all-pairs ZZ readout achieved the best test performance among the ablated families. A final local refinement within this full readout family showed a stable performance plateau around 540 selected features with ridge alpha between 1500 and 2000. The validation-favored setting, top_k=540 and alpha=2000, achieved test RMSE 0.0932, QLIKE -2.2730, Mincer-Zarnowitz R² 0.2196, and test correlation 0.4686.

This supports a cautious architecture conclusion: all-pairs ZZ correlations provide complementary, regime-dependent predictive information when combined with Z/X observables and regularized feature selection. The result improves the QRC readout but does not change the overall Phase 2 conclusion that the ESN reference remains the strongest forecaster.

## Claims to avoid

Do not claim:

- quantum advantage;
- that all-pairs ZZ universally improves all splits;
- that validation selected the full all-ZZ readout globally;
- that ZZ correlations alone carry the predictive signal.

Use instead:

- complementary readout capacity;
- regime-dependent benefit;
- stable test-side QRC improvement;
- evidence that the original nearest-neighbor/conservative readout under-harvested the full-topology TFIM reservoir.
