# Phase 3 executive compass

This is the short operational companion to `PHASE3_REQUIREMENTS_AND_QRC_GUIDANCE.md`.

## What signal structure are we supposed to connect to QRC?

Phase 2 explicitly frames the relevant dynamic-system structure as:

- **long memory**;
- **multi-scale behavior**;
- **regime switching**;
- **non-Gaussian behavior**;
- **chaotic and/or nonlinear dynamics**.

The project must not merely say that finance is nonlinear. We need to identify which of these properties is actually present in the selected task and connect it to a specific QRC mechanism.

Working principle:

> Match a demonstrated signal property to a demonstrated reservoir property.

Examples:

- long memory ↔ fading memory / recurrent state;
- multiscale behavior ↔ temporal multiplexing / multiple evolution times;
- regime switching ↔ nonlinear state-space separation;
- non-Gaussian structure ↔ higher-order observables / nonlinear feature maps;
- nonlinear or chaotic dynamics ↔ non-equilibrium quantum evolution and interaction structure.

## What counts as quantum advantage?

Phase 3 does **not** restrict quantum advantage to lower forecast error.

Allowed forms of benefit include:

- **predictive superiority**;
- **higher-order pattern extraction**;
- **insights difficult to obtain classically**;
- **computational benefit**;
- **robustness or noise-related benefit**.

Any claimed benefit must be quantitative and compared against a strong classical baseline on the same problem instance.

## What numbers must the judges see?

At minimum, the final evidence package must report:

- **qubits / atoms**;
- **circuit depth or analog evolution specification**;
- **shots**;
- **runtime**;
- **performance metrics**.

For Track A, performance reporting should include:

- RMSE;
- QLIKE;
- Mincer-Zarnowitz calibration;
- strong classical baseline values;
- seed variation where applicable;
- scaling, shot, and noise effects.

## Immediate scientific question

Before choosing the final target, determine where linear multiscale approximation is sufficient and where reservoir dynamics add value.

The proposed map is:

```text
natural daily aggregation   -> next daily-scale target
natural weekly aggregation  -> next weekly-scale target
natural monthly aggregation -> next monthly-scale target
```

without overlapping future labels.

For each scale:

```text
HAR performance
vs
ESN gain over HAR
```

Interpretation hypothesis:

```text
HAR strong, ESN similar -> little reservoir headroom
HAR weak, ESN strong    -> reservoir-relevant structure
HAR weak, ESN weak      -> likely noise, not useful nonlinearity
```

This is a diagnostic hypothesis, not a rule for constructing a favorable task.

## Non-negotiable project discipline

1. stakeholder value first;
2. empirical dynamical structure second;
3. strong ESN baseline before quantum claims;
4. exact simulator -> finite shots -> realistic noise -> hardware where feasible;
5. concrete numbers for every claimed advantage.

Progress dashboard: `PHASE3_PROGRESS.md`
