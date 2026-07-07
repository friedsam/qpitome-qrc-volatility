# Track A metrics notes

Date: May 20, 2026

## Correction

RMSE, QLIKE, and Mincer-Zarnowitz regression are not secondary metrics for Track A. They are the stated Track A evaluation metrics and must be treated as central.

Regime-transition forecasting remains the headline problem framing, but the challenge explicitly expects quantitative volatility-forecasting evaluation.

## Required Track A metrics

### RMSE

Root Mean Squared Error. Standard L2 regression error for realized-volatility or realized-variance forecasts.

### QLIKE

Quasi-likelihood loss. Volatility-specific loss recommended by Patton because RMSE-on-variance can be misleading. Use for realized-volatility / variance forecast comparison.

### Mincer-Zarnowitz regression

Regression of realized values on forecasts. Used to test forecast unbiasedness and efficiency, typically joint hypothesis: intercept = 0 and slope = 1.

## Modeling implication

The prototype should not be framed as pure classification.

Minimum aligned modeling stack:

1. Forecast realized volatility / variance over a stated horizon.
2. Evaluate with RMSE, QLIKE, and Mincer-Zarnowitz.
3. Use the volatility forecast path to derive regime-transition warnings.
4. Report regime-transition diagnostics as task-specific interpretation, not as replacements for Track A metrics.

## Submission framing

Correct framing:

- Headline problem: volatility-regime transition early warning.
- Quantitative substrate: realized-volatility / variance forecasting at fixed horizon(s).
- Required evaluation: RMSE, QLIKE, Mincer-Zarnowitz.
- Secondary interpretation: transition warning, lead time, false alarms, event-level behavior.

Incorrect framing:

- Treating PR-AUC/F1 from high-stress classification as the main Track A evaluation.
- Treating RMSE/QLIKE/Mincer-Zarnowitz as optional or secondary.
