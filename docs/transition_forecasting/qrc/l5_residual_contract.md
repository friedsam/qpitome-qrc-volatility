# L5 residual contract

This document freezes the residual target used by any second-stage L5 expert, including QRC and ESN models.

## Scope

This is a downstream modeling and evaluation contract. It does not modify the canonical transition dataset or fold construction.

## Protected baseline

The protected baseline is plain causal MSE-HAR. For each fold and forecast origin, HAR is fitted only on training rows whose complete target path is available before the scoring origin. Validation rows, test rows, same-origin rows, and rows with unavailable targets are excluded.

The baseline is fitted on the full eligible fold population before any expensive-model row subsampling.

## Frozen residual

For horizon h in the ten-step L5 target path:

```text
residual[t, h] = observed_log_volatility[t, h] - causal_mse_har[t, h]
```

The residual branch prohibits:

- a QLIKE offset;
- a second-stage intercept;
- a constant feature;
- segmented constant corrections;
- post-hoc lambda or level calibration.

QLIKE calibration remains a separate baseline diagnostic and must not be credited to the second-stage model.

## Data lineage

- Development folds only.
- Folds 4-6 are construction/selection folds.
- Folds 7-8 are untouched development-confirmation folds.
- Financial test rows are prohibited.
- Fold data are joined by `(fold, sample_id)`.
- Exact `target_end_date` is preferred for target availability; a documented conservative embargo may be used only when that field is absent.

## Shared use by QRC and ESN

Any QRC or ESN model that claims to predict the same HAR residual must consume the same frozen residual registry. A model may instead forecast the direct target, but then it is not a residual-expert comparison and must be labeled separately.

No model-specific sample budget may change the HAR fit population or the residual values.

## QRC readout contract

For a QRC residual expert:

```text
HAR_plus_QRC = HAR + StandardScaler(QRC_features) @ coefficients.T
```

with `Ridge(fit_intercept=False)`.

The implementation must verify numerically that:

1. manual matrix multiplication reproduces every correction;
2. zero standardized QRC features reproduce HAR exactly;
3. no explicit constant column is appended;
4. identical baseline predictions and sample rows are used across QRC and controls.

## Required crisis-warning success pattern

Controls:

```text
E[QRC correction | control] approximately 0
absolute control correction small and bounded
```

Transitions:

```text
E[QRC correction | transition] > 0 before or near horizon 5
HAR + QRC agrees better with the realized future path
```

The transition correction must exceed the control correction on each confirmation fold.

## Feature-attribution gate

A candidate is not promoted unless the desired effect disappears or materially weakens under matched nulls and controls, including:

- validation-row permutation of QRC corrections/features;
- training-residual permutation;
- endpoint-preserving temporal shuffle;
- interaction-off dynamics;
- matched raw linear and quadratic input maps.

A significant feature-residual association alone is insufficient. The predeclared control/transition correction pattern and foldwise confirmation are both required.

## Current result

The first full-63-feature palindrome test passed the narrow feature-attribution check but failed the crisis-warning gate. Its correction was small, fold-unstable, and did not improve transition QLIKE and RMSE on both confirmation folds. This is evidence against the current `level + instability`, 40-step representation, not evidence against every possible input/window choice.

## Canonical implementation

- `src/transition_forecasting/modeling/l5_residual_foundation.py`
- `src/transition_forecasting/qrc/l5_no_intercept_residual_attribution.py`
- `scripts/transition_forecasting/qrc/run_l5_no_intercept_residual_attribution.py`

New results belong under `results/transition_forecasting/qrc/`.
