# Causal HAR branch baseline

Last updated: 2026-07-08

## Purpose

Test whether ordinary multiscale volatility dynamics already resolve the branch problem before moving to nonlinear static models, ESN, or QRC.

This is not a classifier merely named after HAR. It uses the actual historical canonical HAR model to generate one causal volatility forecast at each branch point.

## Preserved HAR model

Exact historical formulation:

```text
inputs:
    rv_5d
    rv_10d
    rv_20d
    rv_60d
    vix_close

StandardScaler
-> Ridge(alpha=1.0)
-> future_rv_20d
```

This matches `scripts/canonical/run_canonical_har.py`.

## Causal forecast construction

For a branch at row `b`, HAR training uses only daily rows satisfying:

```text
row_index + 20 < b
```

Therefore every `future_rv_20d` target used to fit HAR is fully observable before the branch point.

HAR is refit separately for every historical episode. Each episode therefore carries a frozen causal HAR forecast that could have been produced at that date.

## Derived branch signal

Primary HAR diagnostic:

```text
log(HAR predicted future RV20 / current RV20)
```

Interpretation:

```text
negative -> HAR expects volatility normalization
positive -> HAR expects persistence or worsening
```

The experiment does not assume in advance that either sign maps perfectly to recovery or relapse. A small prequential logistic readout learns the historical association.

## Readout ladder

### HAR forecast gap

```text
log(predicted future RV20 / current RV20)
```

### HAR forecast level + gap

```text
predicted future RV20
log(predicted future RV20 / current RV20)
```

### HAR plus current branch state

Adds:

```text
stress ratio
120-day drawdown
RV5 / RV20
recent 5-day return
```

### HAR plus motion

Adds the same two motion/history terms that improved ranking in the first boring baseline:

```text
5-day change in RV5
worst recent 5-day return
```

## Evaluation

Uses the accepted leakage-safe episode prequential protocol.

Mixed episodes remain in chronological availability logic but are excluded from binary recovery-versus-relapse fitting and scoring.

No hyperparameter tuning is performed.

## Outputs

```text
results/baselines/branch_har_front_v1/
```

Saved files:

```text
oos_predictions.csv
summary_metrics.csv
har_episode_features.csv
protocol_steps.csv
run_manifest.json
```

The main question is whether a causal HAR forecast improves on:

```text
historical class rate
VIX only
current state
state + motion
```

If HAR fails, the temporal information needed for branch resolution is not captured by ordinary multiscale volatility forecasting alone.
