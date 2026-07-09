# Primitive regime forecasting front

Last updated: 2026-07-08

## Immediate question

Before training ESN/QRC, establish which broad market states and 20-day transitions are already covered by primitive classical information.

This front is intentionally small and fast. It is a development instrument, not the final regime theory.

## Why next-regime prediction

Predicting the same-day regime from the variables that define it would be circular. The first forecast target is therefore:

```text
broad regime 20 trading rows ahead
```

The current broad regime remains available to the persistence control, but the target is future state.

## Provisional six-state map

The map formalizes the descriptive states already recorded in `regime_branching_analysis.md`:

```text
calm
mild_stress
deterioration
acute_crisis
volatile_rebound
easing_normalization
```

The map uses no VIX and no future target information.

Causal ingredients:

```text
RV5 / RV20 / RV60
expanding historical RV quantiles
5-day and 20-day returns
5-day RV changes
120-day drawdown
short/long RV ratios
```

Rule hierarchy:

1. acute crisis;
2. volatile rebound;
3. deterioration;
4. easing / normalization;
5. mild stress;
6. calm.

Implementation:

```text
src/qpitome_qrc/regimes/broad_state_map.py
```

Important: this is provisional. The old analysis explicitly showed that six clusters and `stress x direction` are not final truths. The first run is designed to expose whether this deterministic map is sensible enough to support the next step.

## Primitive controls

Runner:

```text
scripts/baselines/run_regime_front_baselines.py
```

Models:

### Persistence

```text
future regime = current regime
```

This measures transition inertia.

### HAR-state logistic regression

Inputs:

```text
log RV5
log RV20
log RV60
```

This asks how much of the future regime is already captured by the same multiscale volatility structure that makes HAR-like models effective.

### State logistic regression

Adds simple causal current-state variables:

```text
RV ratios
RV direction
5-day and 20-day returns
120-day drawdown
```

This is still intentionally primitive. No path morphology, sequence model, VIX, macro, ESN, or QRC.

## Evaluation

Uses the shared purged walk-forward geometry.

Metrics:

```text
accuracy
balanced accuracy
macro F1
multiclass log loss
confusion counts
```

The main outputs are not only aggregate scores. We need to inspect:

- regime frequencies;
- transition counts;
- confusion structure;
- OOF class probabilities;
- dates where persistence fails;
- dates where both primitive models fail;
- whether failures concentrate near the unstable branching state.

## Output contract

Default directory:

```text
results/baselines/regime_front_v1/
```

Files:

```text
oof_predictions.csv
fold_metrics.csv
confusion_counts.csv
summary.csv
regime_counts.csv
transition_counts.csv
run_manifest.json
```

## Development substrate

Use the smaller SPY/VIX dataset for the main loop.

Default resolution order:

1. `phase3_spy_vix_volatility_extended.csv` if present;
2. frozen `phase2_spy_vix_volatility.csv` otherwise.

The long `paper_monthly` history remains compatible future work and an occasional spare-machine robustness check. It is not on the current critical path.

## Stop rule for this first run

Do not add models before inspecting the output.

Possible outcomes:

1. persistence already dominates most states: inspect only true transition failures;
2. HAR-scale volatility explains most future states: broad regime forecasting is boring;
3. simple state variables close the remaining gap: let classical models own it;
4. specific transition classes remain confused: those become the next analysis target.

The unstable aftermath / recovery-versus-relapse branch should emerge as one candidate unresolved region, not be forced into the result.
