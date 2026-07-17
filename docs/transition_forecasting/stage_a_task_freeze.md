# Stage A — Track A task freeze

## Scientific object
A volatility regime transition is treated operationally as a persistent change in the conditional volatility state, not as any isolated extreme observation and not as a latent label produced by the forecasting model.

The project does **not** claim that financial markets are deterministic chaotic systems. It tests the narrower claim that ordered pre-transition volatility trajectories contain nonlinear, path-dependent information beyond level, persistence, and static nonlinear summaries.

## Primary task
For index or global market state `X`, use only observations available through `t` to forecast:

1. the next 10-session volatility path; and
2. whether that path enters a persistent turbulent regime within the forecast horizon.

The model input is an ordered window of 20–40 sessions. Primary forecast leads are 1, 5, and 10 sessions before detected onset.

## Primary event definition
For each index, let `x_t = log(ParkinsonVol_t)`. Thresholds are estimated only on dates before 2016-01-01.

- turbulent threshold: training 80th percentile;
- onset candidate: first session crossing the turbulent threshold;
- persistence: at least 10 of the next 15 sessions are turbulent;
- prior state: at most 2 of the previous 10 sessions are turbulent;
- event separation: 60 sessions within an index.

This defines a **persistent transition**. A transient spike that does not satisfy persistence remains outside the primary positive class. February 2018 therefore stays out of the primary SPX class under this rule; it may be analyzed separately as a transient-shock class.

## Independence unit
Index-level onsets occurring within seven calendar days are grouped into one global episode. Train/test splitting, bootstrap uncertainty, permutation tests, and model comparisons must operate at the episode level. Different indices from the same global episode cannot occur in different folds.

## Negative controls
Controls must be drawn from the same index and same evaluation era as the event. They must:

- have no onset in the following forecast horizon;
- lie at least 60 sessions from any onset;
- be matched using only causal pre-origin summaries: level, short/medium means, slopes, recent variance, and recent maximum;
- not be matched on the full ordered trajectory, because that would remove the signal being tested.

## Required outputs of Stage B

- index-level transition catalogue;
- global episode catalogue;
- forecast-origin sample manifest at leads 1/5/10;
- matched-control manifest;
- matching diagnostics and standardized distances;
- data audit with exact date ranges and missingness.

## Nonlinearity gate before QRC
QRC/ESN modeling proceeds only if ordered input windows outperform at least one of:

- shuffled-time windows;
- reversed-time windows;
- static summary features;
- linear autoregressive baseline.

The principal reservoir comparison is QRC versus ESN on identical samples, splits, input channels, readout targets, and tuning budgets.

## Metrics
Continuous path forecasts: horizon-specific RMSE, QLIKE where target scaling is valid, integrated path loss, and Mincer–Zarnowitz diagnostics.

Transition event forecasts: PR-AUC, ROC-AUC, Brier score, calibration, and episode-level bootstrap intervals.

All metrics are reported separately for ordinary observations, matched controls, and transition windows. Unconditional metrics are controls, not the headline result.
