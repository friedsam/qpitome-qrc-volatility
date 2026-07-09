# Branch residual-offset correction results

## Claim tested

A constrained reservoir model may underperform when forced to reproduce structure already handled well by the classical baseline. The appropriate test is therefore not whether a reservoir replaces the baseline, but whether it adds stable information about what the baseline misses.

The tested decomposition was:

```text
state_plus_motion baseline
        -> fit first
        -> freeze

50-unit reset ESN state
+
baseline training residual y - p_baseline
        -> residual-targeted PLS
        -> 1, 2, or 3 components
        -> strongly L2-regularized additive correction to baseline log-odds
```

The correction has no intercept. A zero correction therefore reproduces the original baseline exactly.

No ESN, PLS-dimension, or correction-penalty search was performed.

## Fixed experiment

Baseline:

```text
features:
  stress_ratio
  drawdown_120d
  rv_ratio_5_20_branch
  return_5d_branch
  rv_5d_change_5d_branch
  worst_return_5d_in_prior_window

StandardScaler
LogisticRegression(C=1.0)
```

Reservoir:

```text
50 units
spectral radius 0.9
input scale 0.3
leak 0.3
seeds 0, 1, 2
```

Correction:

```text
PLS dimensions: 1, 2, 3
PLS target: outer-training residual y - p_baseline
frozen baseline logit offset
no correction intercept
L2 penalty: 10
```

Evaluation:

```text
expanding episode-prequential protocol
train only on earlier episodes whose full outcome windows have completed
one OOS probability per eligible binary episode
```

Implementation:

```text
src/qpitome_qrc/baselines/branch_residual_correction.py
scripts/baselines/esn/run_branch_residual_offset_correction.py
tests/test_branch_residual_correction.py
```

## Long-history result

Dataset:

- 80 complete episodes
- 28 recovery
- 19 relapse
- 33 mixed
- 37 binary OOS episodes

Primary ensemble comparison:

| Model | Period | ROC AUC | PR AUC | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| state + motion | all | 0.388 | 0.567 | 0.771 | 0.284 |
| + residual PLS1 | all | 0.339 | 0.518 | 0.944 | 0.341 |
| + residual PLS2 | all | 0.309 | 0.517 | 1.001 | 0.362 |
| + residual PLS3 | all | 0.348 | 0.567 | 0.974 | 0.355 |
| state + motion | 2000+ | 0.479 | 0.552 | 0.765 | 0.283 |
| + residual PLS1 | 2000+ | 0.382 | 0.455 | 0.958 | 0.348 |
| + residual PLS2 | 2000+ | 0.333 | 0.436 | 1.028 | 0.374 |
| + residual PLS3 | 2000+ | 0.368 | 0.507 | 0.997 | 0.365 |
| state + motion | 2010+ | 0.667 | 0.829 | 0.583 | 0.200 |
| + residual PLS1 | 2010+ | 0.639 | 0.823 | 0.614 | 0.215 |
| + residual PLS2 | 2010+ | 0.556 | 0.793 | 0.635 | 0.226 |
| + residual PLS3 | 2010+ | 0.556 | 0.784 | 0.612 | 0.218 |

Mean correction coefficient norms were nonzero:

```text
PLS1: approximately 0.17-0.25
PLS2: approximately 0.25-0.29
PLS3: approximately 0.31-0.32
```

Interpretation:

- The correction machinery did not collapse to zero.
- PLS found reservoir directions correlated with baseline training residuals.
- Those directions did not generalize out of sample.
- The failure is therefore not explained by forcing the ESN to reproduce the baseline's easy component.

## Modern result

Dataset:

- 48 complete episodes
- 17 recovery
- 13 relapse
- 18 mixed
- 17 binary OOS episodes

Primary ensemble comparison:

| Model | ROC AUC | PR AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| state + motion | 0.621 | 0.748 | 0.762 | 0.275 |
| + residual PLS1 | 0.576 | 0.715 | 0.929 | 0.301 |
| + residual PLS2 | 0.606 | 0.760 | 0.892 | 0.293 |
| + residual PLS3 | 0.606 | 0.758 | 0.849 | 0.281 |

2010+ slice:

| Model | ROC AUC | PR AUC | Log loss | Brier |
|---|---:|---:|---:|---:|
| state + motion | 0.533 | 0.856 | 0.766 | 0.284 |
| + residual PLS1 | 0.500 | 0.803 | 0.943 | 0.312 |
| + residual PLS2 | 0.533 | 0.836 | 0.904 | 0.303 |
| + residual PLS3 | 0.533 | 0.834 | 0.844 | 0.286 |

Mean correction coefficient norms were again nonzero:

```text
PLS1: approximately 0.23-0.26
PLS2: approximately 0.26-0.29
PLS3: approximately 0.29-0.34
```

Interpretation:

- PLS2 and PLS3 show tiny PR-AUC improvements in the full modern sample.
- Neither improves ROC AUC, log loss, or Brier score.
- PLS3 comes close to baseline Brier but remains worse overall.
- With only 17 OOS episodes, these small PR-AUC differences are not persuasive evidence of incremental signal.

## Scientific conclusion

The residual architecture is not rejected. The tested ESN representation is rejected as a useful source of complementary transition information.

The distinction is:

```text
residual architecture: conceptually valid
current generic ESN state: empirically negative
```

The complete ESN investigation has now tested and failed to rescue the generic reservoir through:

- reset versus continuous state;
- 50 versus 300-500 units;
- spectral-radius and leak configurations inherited from historical work;
- fixed input normalization;
- explicit bias;
- final-state versus mean trajectory summary;
- 5- and 10-component PCA compression;
- addition to the exact state-plus-motion baseline;
- residual-targeted 1-3 component PLS compression;
- additive correction to a frozen classical baseline.

The generic ESN lane is therefore closed for this binary target.

## Consequence for QRC

Do not reproduce the failed premise by sending the same broad six-channel path into a more expensive quantum reservoir and hoping the internal dynamics discover recovery-versus-relapse structure automatically.

Retain the additive architecture:

```text
classical transition baseline
+
small QRC correction
```

But redesign the quantum input and observables around transition-specific temporal contrasts that are not already the main classical state variables.

The next design question is:

> What temporal structure could distinguish genuine recovery from relapse while remaining absent from the six-variable state-plus-motion baseline?

That question is addressed in `docs/experiments/task_aligned_qrc_transition_design.md`.
