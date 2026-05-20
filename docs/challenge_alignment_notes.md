# Challenge alignment notes

Date: May 20, 2026

## Challenge framing

Track A is financial volatility prediction. The challenge asks for a QRC system that uses public equity-market data to identify volatility regime shifts and forecast their transitions.

The strongest alignment is therefore not a binary high-stress classifier alone. The primary benchmark should be volatility forecasting, with regime/transition interpretation derived afterward.

## Recommended Track A metrics from the challenge brief

- RMSE: standard regression error.
- QLIKE: volatility-specific quasi-likelihood loss.
- Mincer-Zarnowitz regression: realized values regressed on forecasts; test intercept = 0 and slope = 1 for forecast unbiasedness/efficiency.

Classification metrics such as PR-AUC, ROC-AUC, F1, precision, and recall are useful secondary diagnostics for derived stress or regime-warning labels, but they are not the primary Track A metrics.

## Literature ballpark from Li et al. 2025 QRC realized-volatility paper

For S&P 500 realized-volatility forecasting:

### One-step ahead, S = 1

- QR2: MSE about 0.103, QLIKE about 1.4004.
- QR1: MSE about 0.105, QLIKE about 1.4427.
- Best classical RCX: MSE about 0.1089, QLIKE about 1.6480.
- LSTM: MSE about 0.1295.
- HAR: MSE about 0.1476.

### Five-step ahead, S = 5

- Best classical RC: MSE about 0.1528, QLIKE about 2.0551.
- QR1: MSE about 0.1556.
- QR2: MSE about 0.1663.
- LSTM: MSE about 0.1831.
- HAR: MSE about 0.2143.

Interpretation: expected QRC advantage is modest, not dramatic. Matching the strongest classical reservoir within error bars is already useful. A small improvement in RMSE/QLIKE, better robustness, or better scaling/noise behavior would be meaningful.

## Project implication

Current branch preserves the emergency/high-stress-label approach. The next alignment branch should pivot to:

1. continuous volatility forecasting: future_rv_5d / future_rv_20d or multi-horizon volatility vector;
2. Track A metrics: RMSE, QLIKE, Mincer-Zarnowitz;
3. derived regime/transition metrics as secondary interpretation;
4. QRC architecture justification tied to nonlinear multivariate temporal structure, memory, and reservoir expressivity;
5. qubit-count, encoding-density, shot-budget, and noise studies for Phase 3 planning.
