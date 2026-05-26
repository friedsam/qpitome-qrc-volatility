# May 23 Milestone 3 — Classical Baseline Floor

## Purpose

The goal of this milestone was to establish a credible classical comparison floor for the Phase 2 QRC volatility-regime forecasting project.

The baseline study was designed to support the Phase 2 design argument, not to complete a full benchmark campaign. The main requirement was to show that the proposed QRC architecture will be evaluated against meaningful classical alternatives rather than against a weak naive model.

The current project framing uses continuous realized-volatility forecasting as the quantitative substrate:

- `future_rv_5d`
- `future_rv_20d`

The primary evaluation metrics are:

- RMSE
- QLIKE
- Mincer-Zarnowitz regression statistics

A derived volatility-regime transition layer can later be built from the forecasted volatility path, but the main supervised target is continuous realized volatility rather than a sparse binary transition-event label.

## Baseline families

Three baseline families were evaluated or prepared for comparison.

### 1. Persistence baselines

Persistence baselines predict future realized volatility from current trailing realized volatility, for example:

- `rv_5d -> future_rv_5d`
- `rv_10d -> future_rv_5d`
- `rv_20d -> future_rv_20d`
- `rv_60d -> future_rv_20d`

These models provide the primitive volatility-clustering floor. They contain no fitted parameters, but they are meaningful because volatility is persistent and clustered.

### 2. HAR-like and regularized linear baselines

The next baseline family used low-cost classical regression models:

- HAR-like linear regression
- HAR-like ridge regression
- full-feature ridge regression
- full-feature ElasticNet regression

These models test whether the engineered volatility, return, range, drawdown, and VIX features contain exploitable predictive signal beyond naive persistence.

The HAR-like models are especially important because they are simple, interpretable, and strong for realized-volatility forecasting.

### 3. PCA-compressed ESN regression baseline

A classical Echo State Network regression baseline was then built using ReservoirPy.

This reused the strongest pieces of the earlier ESN prototype but adapted them to the current regression framing:

- binary stress classification was replaced by continuous realized-volatility regression;
- logistic readout was replaced by ridge regression;
- PR-AUC/F1 metrics were replaced by RMSE, QLIKE, and Mincer-Zarnowitz diagnostics;
- raw-volatility prediction was replaced by log-volatility prediction to stabilize QLIKE.

The ESN used PCA-compressed volatility/VIX inputs and a 40-day input sequence. The best-performing bounded configuration used:

```text
feature set: PCA-6 / PCA-8 / PCA-10
target: future_rv_20d
sequence length: 40 trading days
reservoir units: 300
spectral radius: 0.7
leak rate: 0.5
input scaling: 0.5
reservoir connectivity: 0.1
readout: ridge regression on log-volatility
ridge alpha: 10
seeds: 1–5
```

The log-target readout was important. Direct raw-volatility ESN prediction produced occasional near-zero volatility forecasts, which caused pathological QLIKE values even when RMSE looked acceptable. Predicting log-volatility and transforming back with `exp()` removed this failure mode and produced much more stable QLIKE behavior.

## PCA compression diagnostics

The PCA diagnostics for the current 27-feature Phase 2 regression feature set are different from the older compact/correlation-pruned classifier feature set. The older compact-feature analysis showed much higher cumulative variance at PCA-6. That number should not be used for the current full-feature regression pipeline.

Current train-only PCA explained variance for the 27-feature Phase 2 feature set:

| Component | Explained variance ratio | Cumulative explained variance |
|---:|---:|---:|
| 1 | 0.408573 | 0.408573 |
| 2 | 0.112421 | 0.520994 |
| 3 | 0.090849 | 0.611843 |
| 4 | 0.073259 | 0.685102 |
| 5 | 0.064206 | 0.749308 |
| 6 | 0.055584 | 0.804892 |
| 7 | 0.036496 | 0.841388 |
| 8 | 0.032709 | 0.874098 |
| 9 | 0.031574 | 0.905672 |
| 10 | 0.016945 | 0.922617 |

Interpretation:

```text
PCA-6:
  aggressive small-qubit fallback; captures ~80.5% of current full-feature variance

PCA-8:
  primary QRC input setting; captures ~87.4% of current full-feature variance

PCA-10:
  sensitivity / larger-input extension; captures ~92.3% of current full-feature variance
```

This does not invalidate the PCA-compressed ESN result or the QRC design. It clarifies the architecture ladder: PCA-6 is a deliberately compressed fallback, while PCA-8 and PCA-10 preserve more information for the primary and sensitivity QRC settings.

## Key results

The persistence baseline confirmed that volatility clustering provides useful signal, but it was clearly weaker than learned models.

For the 20-day target, the earlier persistence and HAR-like results established the following approximate comparison floor:

```text
future_rv_20d persistence:
  test RMSE ≈ 0.124

future_rv_20d HAR-like ridge / linear:
  test RMSE ≈ 0.101
  test MZ R² ≈ 0.354
```

The PCA-compressed ESN with 40-day memory materially improved the held-out 2020–2024 test performance:

```text
future_rv_20d PCA-compressed ESN:
  test RMSE ≈ 0.075–0.086 across strong seeds/settings
  test QLIKE ≈ -2.44 to -2.49
  test MZ R² ≈ 0.45–0.53
```

This is a substantial improvement over both persistence and the HAR-like linear baseline on the held-out test period.

The result should not be overstated as a final benchmark claim because the ESN was explored with a bounded but non-exhaustive tuning process. However, it is strong enough for Phase 2: the project now has a sophisticated classical reservoir comparator, not merely primitive baselines.

## Interpretation

The baseline study produced three important conclusions.

First, the regression framing is superior to the earlier sparse binary stress-classification framing. The earlier classifier task was difficult, imbalanced, and only indirectly aligned with Track A. The continuous realized-volatility target provides a cleaner quantitative substrate and supports the required volatility-forecasting metrics.

Second, the ESN result shows that nonlinear reservoir memory is useful for this task. A PCA-compressed ESN with 40-day memory and ridge-regularized log-volatility readout outperformed persistence and HAR-like linear baselines on the held-out test period. This suggests that market-state information is not fully captured by static HAR-style features alone.

Third, the ESN provides direct design information for QRC. The QRC prototype should be compared against a reservoir-style classical control, not only against persistence or linear regression. The ESN result suggests that the following ingredients matter:

- compressed volatility/VIX state representation;
- approximately 40 trading days of memory;
- positive/log-volatility output modeling;
- ridge-regularized readout;
- seed-level robustness checks;
- comparison against both full-feature and PCA-compressed classical models.

## Implication for QRC

The QRC design argument is now sharper.

A weak QRC prototype should not be presented as successful merely because it beats naive persistence. The meaningful comparison is against a compact reservoir baseline: PCA-compressed ESN regression.

The near-term QRC prototype may require PCA because only a limited number of qubits or input channels can be used. This is acceptable for Phase 2, because PCA-compressed ESN baselines already show that useful volatility information survives compression. In Phase 3, this bottleneck can be relaxed through larger quantum reservoirs, feature re-uploading, alternative encodings, or hybrid architectures.

The immediate QRC goal is therefore:

```text
Test whether a compact quantum reservoir can reproduce or improve useful aspects of the ESN-style nonlinear-memory baseline under comparable PCA-compressed inputs.
```

## Phase 3 extensions

The Phase 2 baseline study was intentionally bounded. Phase 3 should extend it through broader robustness checks and additional model families.

Planned Phase 3 extensions include:

- systematic ESN hyperparameter sweeps;
- larger ESN reservoirs and broader seed studies;
- GARCH-family volatility baselines;
- LSTM/GRU sequence-learning baselines;
- regime-specific evaluation across calm, turbulent, and transition periods;
- noise and resource-scaling studies for QRC;
- testing whether the ESN advantage persists under different market indices, volatility targets, and train/test windows.

These extensions are not required for the Phase 2 design argument, but they will be necessary before making strong benchmark or advantage claims.

## Milestone sign-off

May 23 is complete.

The project now has:

- naive persistence baselines;
- HAR-like and regularized linear regression baselines;
- PCA diagnostics;
- a non-toy ReservoirPy ESN regression baseline;
- evidence that PCA-compressed reservoir memory can outperform simpler classical baselines on the held-out 20-day realized-volatility target;
- a much stronger comparison floor for QRC design.

The key milestone result is that the project moved beyond primitive classification and established a sophisticated classical reservoir benchmark that directly informs the QRC architecture.
