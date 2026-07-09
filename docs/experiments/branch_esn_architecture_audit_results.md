# Branch ESN architecture audit results

## Scope

This note records the outcome of the bounded ESN repair experiment defined in `docs/experiments/branch_esn_architecture_audit.md`.

The four fixed variants were:

1. primitive frozen reset ESN
2. fixed normalized inputs
3. fixed normalized inputs + constant bias channel
4. fixed normalized inputs + bias + mean reservoir-state summary

No hyperparameter search was performed.

## Long-history result

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
- Adding a bias improves Brier score and log loss, especially in the 2000+ and 2010+ slices, but does not improve ROC AUC.
- Adding a mean reservoir-state summary improves full-history ranking from AUC 0.348 to 0.397 and PR AUC from 0.528 to 0.580, indicating that final-state-only compression discards information.
- However, the mean-state variant worsens calibration strongly and does not improve the contemporary slices. The naive higher-dimensional trajectory summary is therefore not usable.

## Modern result

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
- Bias slightly improves 2010+ ranking, but only with 13 OOS episodes and 3 relapses, while worsening overall probability quality.
- The mean-state trajectory summary is consistently worse on the modern sample.

## Combined conclusion

The bounded audit does not produce a better usable ESN.

The repairs reveal three separate facts:

1. **Input scaling is not the main failure.** Fixed normalization does not improve modern or long-history ranking.
2. **Bias can improve calibration in some historical slices.** This is a real but limited effect and does not generalize to the modern sample.
3. **Final-state-only compression does discard information.** The long-history mean-state variant improves rank metrics, but the naive higher-dimensional readout overfits and fails to generalize.

The primitive ESN result therefore remains negative, but the reason is not reducible to one obvious implementation defect. At the same time, the audit does not justify a broad architecture search.

## Decision

For the current binary recovery-versus-relapse task:

- keep the primitive ESN as the frozen classical reservoir benchmark;
- retain the architecture-audit variants as negative/diagnostic evidence;
- do not replace the benchmark with normalized, biased, or mean-state variants;
- do not continue tuning this binary task aggressively because the future stakeholder-facing target may become recovery/relapse/inconclusive.

One additional bounded admission test may still be justified before closing the ESN lane: evaluate one or more frozen historical Phase 2 ESN configurations without tuning, because the current branch model uses only 50 units and does not test the architecture scale that previously succeeded. This is a capacity admission test, not a hyperparameter search.
