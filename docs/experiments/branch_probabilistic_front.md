# First branch probabilistic baseline ladder

Last updated: 2026-07-08

## Purpose

This is the first model stage after the branching-state extractor, provisional outcome labels, and leakage-safe prequential geometry were established.

The goal is to let simple classical models kill an uninteresting branch-prediction problem before any ESN or QRC work.

## Task

Primary first-pass task:

```text
recovery versus relapse
```

Positive class:

```text
recovery
```

Mixed episodes remain part of the chronological episode sequence and therefore affect which prior labels are available at each test date, but they are excluded from binary model fitting and binary scoring.

## Evaluation geometry

Uses:

```text
src/qpitome_qrc/evaluation/episode_prequential.py
```

For each eligible binary test episode:

```text
collect all earlier episodes whose full 40-day outcomes are already known
-> retain recovery/relapse training episodes
-> fit model using only that historical training set
-> predict one OOS recovery probability
```

Every test episode receives exactly one OOS probability.

## Baseline ladder

### 1. Historical class-rate prior

No market features.

```text
p(recovery) = historical recovery fraction among available binary training episodes
```

This is the minimum probabilistic reference.

### 2. VIX only

```text
vix_close
```

This directly tests whether the branching problem is trivial once current market-implied stress is known.

### 3. Current causal state

```text
stress_ratio = rv20 / causal expanding stress threshold
120-day drawdown
RV5 / RV20
recent 5-day return
```

These encode current stress, damage, short-versus-long volatility shape, and immediate stabilization direction.

### 4. Current state plus motion/history

Adds:

```text
5-day change in RV5
worst recent 5-day return in the prior window
```

This is still deliberately small. It asks whether immediate volatility motion and recent damage history improve on the current state without introducing a broad feature search.

## Model class

All feature models use:

```text
StandardScaler
-> logistic regression
```

Fixed configuration:

```text
C = 1.0
solver = lbfgs
no hyperparameter tuning
```

The scaler and classifier are refit from scratch on each prequential historical training set.

## Saved outputs

```text
results/baselines/branch_probabilistic_front_v1/
```

Outputs:

```text
oos_predictions.csv
summary_metrics.csv
feature_manifest.csv
protocol_steps.csv
run_manifest.json
```

## Metrics

First-pass pooled OOS diagnostics:

```text
ROC-AUC
recovery PR-AUC
log loss
Brier score
accuracy at 0.5
mean predicted recovery probability
```

The individual episode probabilities are primary evidence because the OOS binary sample is small.

Uncertainty intervals and chronological resampling are intentionally deferred until the baseline ladder is inspected. The first question is whether any simple feature set clearly dominates the historical prior.
