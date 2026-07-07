# Phase 3 Test Plan

Purpose: freeze the current Phase 2-derived behavior before cleanup or restructuring.

## 1. Data Pipeline Tests

Goal: verify that the processed modeling dataset and chronological split logic still work.

Checks:

- processed dataset file exists;
- required target and date columns exist;
- required feature columns exist or can be generated;
- train, validation, and test splits are chronological;
- train/validation/test windows do not overlap;
- preprocessing objects are fit on train only.

Candidate test file:

```text
tests/test_data_pipeline.py
```

## 2. Metric Tests

Goal: ensure core evaluation functions remain stable.

Checks:

- RMSE returns expected value on simple arrays;
- QLIKE handles positive inputs correctly;
- QLIKE does not fail on near-zero protected inputs;
- Mincer-Zarnowitz regression diagnostics return finite values;
- correlation logic handles constant or near-constant arrays safely.

Candidate test file:

```text
tests/test_metrics.py
```

## 3. QRC Core Tests

Goal: verify that the digital TFIM-QRC machinery still imports and produces valid features.

Checks:

- `TFIMQRCConfig` imports;
- reservoir object initializes from a small config;
- tiny synthetic input produces a finite feature matrix;
- feature shape is stable for a fixed synthetic input;
- no NaN or infinite features are produced.

Candidate test file:

```text
tests/test_tfim_qrc_core.py
```

## 4. Prediction Export / Smoke Tests

Goal: make sure key Phase 2 reference outputs still exist and are readable.

Checks:

- key baseline metric tables exist;
- QRC prediction export files exist;
- ESN prediction export files exist;
- prediction tables contain target and prediction columns;
- metric tables contain expected model, split, and metric columns.

Candidate test file:

```text
tests/test_prediction_exports.py
```

## 5. Exclusions for Now

Do not test notebooks directly. They are too noisy and contain execution-state artifacts.

Do not test live QPU execution. Phase 3 should remain simulator/emulator-first until access is confirmed.

## 6. Order of Implementation

1. Metric tests.
2. Data pipeline tests.
3. Prediction export smoke tests.
4. Minimal QRC core tests.

This order gives quick protection before touching slower QRC code.
