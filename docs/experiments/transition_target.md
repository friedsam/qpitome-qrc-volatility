# Phase 3 volatility-transition target

## Primary target

The leading Phase 3 Track A target is

```text
rv_innovation_20d = log(future_rv_20d / rv_20d)
```

Interpretation:

- positive: realized volatility increases over the forecast horizon;
- negative: realized volatility decreases;
- zero: no change relative to the current 20-day realized-volatility level.

The target is defined in `qpitome_qrc.data.targets` and must be constructed before each runner applies its existing split-local missing-data policy.

## Protocol invariants

The target change does not authorize changes to:

- canonical dataset;
- five purged expanding walk-forward folds;
- feature availability;
- train-only scaling or PCA;
- sequence/window geometry;
- model mechanics;
- test-fold isolation.

For the first model adaptation, use the same canonical feature information as the full linear baseline. Do not recreate the previous information asymmetry where the strongest level predictor was exposed directly to one comparator but only indirectly or not at all to others.

## Primary metrics

Report transition-space metrics:

- innovation RMSE;
- innovation R-squared.

A zero-change forecast predicts innovation `0` at every test point and is the persistence reference.

## Secondary level reconstruction

For comparability with the historical level task, reconstruct

```text
future_rv_pred = rv_20d * exp(predicted_innovation)
```

and report the existing future-volatility RMSE, QLIKE, and Mincer-Zarnowitz diagnostics as secondary metrics.

## Evidence establishing the target candidate

On the canonical five-fold protocol:

- the historical `har_ridge` manifest uses `[rv_5d, rv_10d, rv_20d, rv_60d, vix_close]` with a raw target;
- a VIX-only ridge nearly reproduces its median level-target RMSE;
- the innovation target has near-zero skill for the zero-change baseline but positive out-of-sample structure for broader models.

The old `future_rv_20d` level task remains a contextual benchmark. It is not deleted or relabeled as invalid; its interpretation is narrower because it is dominated by volatility-level persistence and implied-volatility information.

## ESN adaptation contract

The first ESN experiment should:

1. preserve current canonical fold, feature, preprocessing, sequence, reservoir, and validation-selection mechanics;
2. replace the target/readout semantics with direct prediction of signed `rv_innovation_20d`;
3. compare against zero-change, HAR-RV, and full linear baselines on identical folds;
4. report transition-space metrics first and reconstructed future-RV metrics second;
5. avoid test-fold hyperparameter tuning.
