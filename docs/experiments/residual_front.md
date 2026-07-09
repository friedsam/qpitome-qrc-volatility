# Residual front: cheap classical coverage before reservoirs

Last updated: 2026-07-08

## Purpose

The Phase 3 architecture is moving from one-model forecasting toward a modular decomposition:

```text
cheap classical predictor
        +
reservoir correction on unresolved structure
```

The immediate objective is not to optimize a final score. It is to create trustworthy out-of-fold predictions that define what remains unresolved before ESN/QRC modeling begins.

## Current development substrate

Use the smaller SPY/VIX dataset for the main development loop.

Default data resolution order:

1. `data/processed/phase3_spy_vix_volatility_extended.csv`, if present;
2. `data/processed/phase2_spy_vix_volatility.csv` otherwise.

The long `paper_monthly` history remains a compatibility/robustness path and can be exercised opportunistically on a spare machine. It is not on the critical development path.

## Canonical target

```text
rv_innovation_20d = log(future_rv_20d / rv_20d)
```

Zero innovation is persistence / no change:

```text
predicted future RV20 = current RV20
```

All residuals must be formed from genuinely out-of-fold predictions:

```text
residual_innovation = true_innovation - oof_innovation_prediction
```

Never train a downstream residual model on residuals from an in-sample upstream fit.

## First primitive controls

Runner:

```text
scripts/baselines/run_residual_front_baselines.py
```

Models:

1. `persistence`
   - predicts zero innovation;
   - therefore reconstructed future RV equals current RV20.

2. `har_ridge`
   - compact HAR-style multiscale control;
   - causal features: log RV5, log RV20, log RV60;
   - Ridge alpha selected on the validation block;
   - final fold model refit on train + validation;
   - test prediction remains strictly out of fold.

This is deliberately not a model zoo. More elaborate classical models are being developed separately.

## Shared target utilities

```text
src/qpitome_qrc/evaluation/targets.py
```

Owns only:

- construction of the log-RV innovation target;
- reconstruction of future RV from innovation predictions;
- residual algebra.

It does not own features, model fitting, fold construction, or reporting.

Historical canonical runners and the frozen historical NumPy ESN remain untouched.

## Output contract

Default output directory:

```text
results/baselines/residual_front_v1/
```

Files:

```text
oof_predictions.csv
fold_metrics.csv
summary.csv
run_manifest.json
```

The OOF artifact is long-form by model and contains:

```text
date
fold
model
y_true_innovation
oof_innovation_prediction
residual_innovation
current_rv_20d
future_rv_20d
reconstructed_future_rv_20d
```

Downstream ESN/QRC runners should consume this artifact rather than reconstruct upstream predictions independently.

## Metrics

Development diagnostic:

- innovation R².

Challenge-facing reconstructed-RV metrics:

- RMSE;
- QLIKE;
- Mincer-Zarnowitz intercept, slope, and R².

## Current short path

1. Run persistence and HAR-style Ridge quickly.
2. Inspect OOF residuals and determine where errors concentrate.
3. Compare direct ESN versus residual ESN.
4. Investigate whether reservoir inputs should differ from the classical feature space.
5. Compare PCA strategies under matched input budget.
6. Treat QRC primarily as an additive correction unless evidence favors a direct model.

## Feature-space rule

Do not assume the feature representation useful for the classical component is the representation useful for ESN/QRC.

The planned input-reduction comparison is:

- PCA on the original/classical feature space;
- PCA on redesigned nonlinear/residual features;
- redesigned features without PCA when dimension permits;
- leading versus trailing versus mixed PCs;
- random projections;
- later, residual-supervised directions.

PCA is a variance criterion, not a quantum-relevance criterion.

## Rerun rule

Every stronger upstream model changes the residual target. Therefore:

```text
better classical model
-> new OOF predictions
-> new residuals
-> rerun ESN
-> rerun input-selection tests
-> rerun QRC
```

This is expected. The three-machine setup exists to support this rolling front.

Each upstream artifact must therefore be versioned and accompanied by a run manifest.
