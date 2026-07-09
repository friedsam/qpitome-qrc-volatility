# Regime target hierarchy and decision process

Last updated: 2026-07-08

## Why this note exists

The project is deliberately trying a new Phase 3 formulation. The target should not be locked by intuition alone, but the workflow must remain stable enough that experiments accumulate rather than drift.

The governing principle remains:

> Broad volatility is likely classically easy. Use regime and transition analysis to locate the unresolved dynamical remainder before allocating ESN/QRC capacity.

## Target hierarchy

The project currently distinguishes four different tasks.

### 1. Broad continuous volatility forecasting

```text
current market information -> future realized volatility
```

Role:

- challenge-metric substrate;
- confirms that broad volatility is largely classical/persistent;
- provides the negative control showing that QRC should not be expected to dominate an easy low-dimensional problem.

This is not currently the main QRC target.

### 2. Future broad regime forecasting

```text
current state -> broad regime 20 trading days later
```

Current provisional states:

```text
calm
mild_stress
deterioration
acute_crisis
volatile_rebound
easing_normalization
```

Role:

- diagnostic only;
- measures broad state inertia and multiclass difficulty;
- identifies which future-state distinctions are already cheap.

Do not heavily optimize this task. Stable regimes dominate it.

### 3. Regime-change forecasting

```text
current state -> change / no change over 20 trading days
```

Role:

- removes the need to predict the exact future class initially;
- focuses directly on transition risk;
- produces OOF change probabilities;
- localizes errors by exact `current -> future` transition.

Runner:

```text
scripts/baselines/run_regime_change_front.py
```

This is the immediate next experiment.

### 4. Branch-resolution forecasting

```text
unstable stressed aftermath -> recovery / relapse / mixed
```

Role:

- primary candidate for the eventual classical -> ESN -> QRC residual ladder;
- should be defined only after the broad transition diagnostics are inspected;
- should recover the recurrent unstable-aftermath morphology if that phenomenon is real.

The branch state must be defined causally before future outcomes are labeled.

## Expected difficult region

The strongest prior is not `calm vs crisis`.

It is the region where:

```text
realized volatility remains high
longer-horizon damage remains visible
short-horizon volatility is decelerating
recent decline has slowed or partially reversed
```

Likely difficult transition families include:

```text
deterioration -> volatile rebound
versus continued/renewed deterioration

volatile rebound -> easing/normalization
versus volatile rebound -> deterioration

easing/normalization -> calm/mild stress
versus easing/normalization -> deterioration
```

The most important candidate is apparent recovery or normalization that can still resolve either into genuine recovery or relapse.

## Why broad volatility should favor classical models

Broad volatility forecasting is dominated by:

```text
persistence
current RV level
multiscale RV structure
VIX-like stress information
```

A constrained QRC is therefore being asked to spend capacity relearning low-dimensional structure that cheap classical models already represent well.

A more coherent Phase 3 test is:

```text
classical model owns broad volatility/state
+
ESN/QRC attacks unresolved transition structure
```

## Current execution order

1. Keep the completed six-class next-regime run as a diagnostic.
2. Ignore its current multiclass log-loss column until the class-order warning is corrected; accuracy, balanced accuracy, macro-F1, and confusion outputs remain usable.
3. Run the binary regime-change front.
4. Inspect:
   - overall transition rate;
   - HAR-scale change prediction;
   - broader state-model change prediction;
   - hardest exact transitions;
   - hardest current regimes.
5. Decide whether the unstable-aftermath branch appears naturally in the failure structure.
6. Only then lock the episode-level branch target.
7. Generate strong classical OOF probabilities.
8. Define residual targets for ESN/QRC.

## Stop rule

Do not add ESN or QRC merely because the transition problem is difficult.

Proceed only if the unresolved errors are:

- recurrent across folds/history;
- concentrated in stakeholder-relevant transition families;
- not eliminated by cheap classical controls;
- compatible with a causal episode definition.
