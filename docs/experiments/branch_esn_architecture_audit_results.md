# Branch ESN architecture audit results

## Scope

This note records the outcome of the bounded ESN repair, capacity-admission, and compact-readout experiments defined in `docs/experiments/branch_esn_architecture_audit.md`.

The goal was not broad ESN tuning. The goal was to determine whether the primitive negative result could be explained by an obvious architecture defect, insufficient reservoir capacity, or excessive readout dimensionality.

## Initial bounded architecture audit

The four fixed variants were:

1. primitive frozen reset ESN
2. fixed normalized inputs
3. fixed normalized inputs + constant bias channel
4. fixed normalized inputs + bias + mean reservoir-state summary

No hyperparameter search was performed.

### Long-history result

Dataset:

- 80 complete episodes
- 28 recovery
- 19 relapse
- 33 mixed
- 37 binary OOS predictions

Ensemble metrics:

| Variant | Period | ROC AUC | PR AUC | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| primitive | all | 0.348 | 0.528 | 0.913 | 0.329 |
| normalized | all | 0.342 | 0.514 | 0.897 | 0.326 |
| normalized + bias | all | 0.333 | 0.504 | 0.902 | 0.323 |
| normalized + bias + mean | all | 0.397 | 0.580 | 0.987 | 0.344 |
| primitive | 2000+ | 0.299 | 0.403 | 1.050 | 0.377 |
| normalized | 2000+ | 0.264 | 0.389 | 1.036 | 0.380 |
| normalized + bias | 2000+ | 0.271 | 0.392 | 0.994 | 0.357 |
| normalized + bias + mean | 2000+ | 0.292 | 0.401 | 1.150 | 0.401 |
| primitive | 2010+ | 0.389 | 0.694 | 0.751 | 0.274 |
| normalized | 2010+ | 0.278 | 0.591 | 0.775 | 0.284 |
| normalized + bias | 2010+ | 0.333 | 0.612 | 0.727 | 0.260 |
| normalized + bias + mean | 2010+ | 0.306 | 0.614 | 0.953 | 0.352 |

Interpretation:

- Fixed input normalization alone does not improve discrimination and only marginally improves full-history probability quality.
- Adding a bias improves Brier score and log loss in some historical slices but does not improve ROC AUC.
- Adding a mean reservoir-state summary improves full-history ranking from AUC 0.348 to 0.397 and PR AUC from 0.528 to 0.580, indicating that final-state-only compression discards information.
- The mean-state variant worsens calibration strongly and does not improve contemporary slices. The naive higher-dimensional trajectory summary is not usable.

### Modern result

Dataset:

- 48 complete episodes
- 17 recovery
- 13 relapse
- 18 mixed
- 17 binary OOS predictions

Ensemble metrics:

| Variant | Period | ROC AUC | PR AUC | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| primitive | all | 0.394 | 0.607 | 0.972 | 0.329 |
| normalized | all | 0.394 | 0.607 | 0.989 | 0.330 |
| normalized + bias | all | 0.394 | 0.597 | 1.020 | 0.338 |
| normalized + bias + mean | all | 0.364 | 0.594 | 1.103 | 0.371 |
| primitive | 2010+ | 0.333 | 0.707 | 1.034 | 0.338 |
| normalized | 2010+ | 0.333 | 0.707 | 1.065 | 0.345 |
| normalized + bias | 2010+ | 0.367 | 0.725 | 1.074 | 0.342 |
| normalized + bias + mean | 2010+ | 0.300 | 0.692 | 1.145 | 0.368 |

Interpretation:

- None of the bounded repairs improves the modern all-period result.
- Primitive and normalized models have identical AUC.
- Bias slightly improves 2010+ ranking but worsens overall probability quality.
- The mean-state trajectory summary is consistently worse on the modern sample.

## Historical-capacity admission test

The current branch ESN uses only 50 reservoir units, so a final capacity objection remained. Four frozen historical Phase 2/3 NumPy ESN configurations were tested without tuning:

- 300 units, spectral radius 0.70, input scale 0.30, leak 0.30
- 300 units, spectral radius 0.90, input scale 0.30, leak 0.30
- 500 units, spectral radius 0.70, input scale 0.20, leak 0.50
- 500 units, spectral radius 0.90, input scale 0.20, leak 0.50

The binary classification readout remained fixed so the test isolated reservoir capacity rather than readout search.

Full-history ensemble results:

| Reservoir | ROC AUC | Log loss | Brier |
|---|---:|---:|---:|
| 50-unit primitive | 0.348 | 0.913 | 0.329 |
| 300, sr 0.70 | 0.330 | 1.089 | 0.367 |
| 500, sr 0.70 | 0.321 | 1.189 | 0.367 |
| 300, sr 0.90 | 0.303 | 1.147 | 0.395 |
| 500, sr 0.90 | 0.306 | 1.263 | 0.408 |

Interpretation:

- Reservoir capacity is not the main failure.
- Larger historically successful reservoirs do not reveal hidden branch-resolution skill.
- Probability quality worsens as reservoir dimensionality increases.
- The 50-unit result is not an artifact of an obviously underpowered reservoir.

## Compact readout / PCA admission test

The next question was whether the reservoir contained useful temporal information but exposed too many coordinates to a tiny readout sample.

The 50-unit primitive reservoir was therefore compressed causally inside each prequential training step to 5 or 10 PCA components and added to the exact `state_plus_motion` baseline.

### Long history

| Model | Period | ROC AUC | PR AUC | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| state + motion | all | 0.388 | 0.567 | 0.771 | 0.284 |
| + 5 ESN PCs | all | 0.348 | 0.585 | 0.974 | 0.350 |
| + 10 ESN PCs | all | 0.382 | 0.638 | 1.053 | 0.388 |
| state + motion | 2010+ | 0.667 | 0.829 | 0.583 | 0.200 |
| + 5 ESN PCs | 2010+ | 0.500 | 0.761 | 0.639 | 0.228 |
| + 10 ESN PCs | 2010+ | 0.361 | 0.714 | 0.879 | 0.329 |

Five components explain roughly 94-96% of reservoir variance. Ten explain roughly 99.76-99.79%.

### Modern sample

| Model | ROC AUC | PR AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| state + motion | 0.621 | 0.748 | 0.762 | 0.275 |
| + 5 ESN PCs | 0.545 | 0.704 | 1.007 | 0.311 |
| + 10 ESN PCs | 0.561 | 0.762 | 0.878 | 0.309 |

Five components explain roughly 95-96% of reservoir variance. Ten explain roughly 99.83-99.91%.

Interpretation:

- Excessive nominal readout dimension is not the full explanation.
- The reservoir representation is already highly compressible.
- Preserving almost all reservoir variance does not preserve useful incremental transition information.
- The dominant reservoir directions appear to encode strong but task-misaligned structure such as stress magnitude, volatility level, path amplitude, or historical regime.
- Adding these compact generic directions actively harms the stronger task-specific `state_plus_motion` baseline.

## Methodological consequence for PCA and QRC

The compact-readout result makes a broader distinction explicit:

> Preserving variance is not the same as preserving information complementary to a classical model.

If a classical model performs nearly identically after PCA, that proves only that the retained components preserve what that classical model needs. It provides no evidence either for or against whether the same compression preserves information a quantum model could use to add value beyond the classical model.

Formally, the relevant project objective is closer to preserving information about the outcome conditional on the classical baseline than to preserving marginal variance:

```text
useful objective:       information(Z; Y | classical baseline)
PCA objective:          variance(Z)
```

This is not an argument that PCA is bad. It is an argument against treating classical-model equivalence after PCA as validation of quantum opportunity.

A constrained quantum model should also not be forced to reconstruct an easy component that a classical model already predicts well. Doing so can lower the total score even if the quantum model captures genuinely complementary structure in the hard remainder.

## Combined conclusion

The generic standalone ESN lane is now closed for the current binary branch target.

The negative result cannot be reduced to:

- too few reservoir units;
- one poor spectral radius or leak setting;
- unequal raw channel scale;
- lack of bias;
- final-state-only compression;
- excessive nominal readout dimension.

The strongest current interpretation is:

> The generic reservoir is learning structure, but not structure aligned with recovery-versus-relapse information missing from the classical transition baseline.

## Decision

For the current binary recovery-versus-relapse task:

- keep the primitive ESN as the frozen generic classical-reservoir benchmark;
- retain the architecture, capacity, and PCA experiments as negative/diagnostic evidence;
- do not continue broad standalone ESN tuning;
- do not treat high variance retention or classical equivalence after compression as evidence about quantum success;
- move to a task-aligned baseline-plus-correction architecture.

The next bounded experiment is:

```text
state_plus_motion baseline
+
1-3 residual-targeted PLS components of the frozen ESN
+
strongly regularized additive logit correction
```

The baseline is fitted first and frozen. PLS is supervised by the baseline training residual `y - p_baseline`. The correction has no intercept, so a zero correction exactly recovers the baseline. This directly tests whether the reservoir contains information about what the classical transition model missed rather than asking it to reproduce the easy classical component.
