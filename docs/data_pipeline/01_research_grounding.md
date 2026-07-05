# Research grounding for the July 2026 reset

## Why the reset is necessary

The previous pipeline drifted toward a 20-day daily SPY/VIX task that strongly favored HAR-style multiscale realized-volatility averages. That task was permissible, but it was not derived backward from the challenge's paper stack. The result was a mismatch between the stated QRC goal and the actual experimental arena.

## Challenge requirements that govern the rebuild

The challenge asks for a QRC system on a public financial dataset, benchmarked rigorously against strong classical methods. For Track A, the recommended baselines are GARCH, ESN, and LSTM; the brief explicitly identifies ESN as the most important baseline because beating a classical reservoir is what would justify QRC. Primary metrics are RMSE, QLIKE, and Mincer-Zarnowitz calibration. The workflow must be reproducible and should characterize scaling, encoding density, shot budget, and noise.

The challenge also points to public OHLCV data, realized-volatility libraries, and FRED macroeconomic data, and cites neutral-atom QRC, few-atom QRC, chaotic forecasting, realized-volatility QRC, memory/nonlinearity tradeoffs, and Aquila hardware studies.

## What the finance anchor paper actually does

Li et al., *Quantum Reservoir Computing for Realized Volatility Forecasting* (arXiv:2505.13933):

- Target: monthly log realized volatility of the S&P 500.
- Construction: natural log of the square root of the sum of squared daily returns within each month.
- Frequency: monthly, non-overlapping observations.
- Span: February 1950 to December 2017, 815 observations.
- Out-of-sample period: 245 monthly forecasts from August 1997 to December 2017.
- Inputs: RV and quarterly/annual RV aggregates plus valuation, Fama-French, interest-rate, inflation, credit, and industrial-production variables.
- Forecast design: rolling monthly re-estimation, one-step and five-step ahead.
- Baselines: AR1, AR3, ARMAX, HAR, HARX, LSTM, LSTMX, RC, RCX.
- Evaluation: MSE, QLIKE, Model Confidence Set, and Diebold-Mariano tests.
- Classical-reservoir result: RC without exogenous variables is weak; RCX improves sharply, indicating that the richer state matters.
- QRC result: the quantum reservoirs are competitive with or better than the classical set on the authors' benchmark, with QR2 best at one step.

Important caveat: the paper selects among many reservoir instances/features and reports the best-performing configurations. We will not copy that practice uncritically; selection must be time-series-aware and separated from final testing.

## Lessons from the broader paper stack

### Large-scale analog neutral-atom QRC
Kornjaca et al. (arXiv:2407.02553) emphasize a three-stage pipeline: classical preprocessing, quantum reservoir encoding, and classical readout. They explicitly allow dimensional reduction and feature engineering before quantum encoding, and demonstrate several analog encoding channels. Scaling the physical reservoir and readout capacity is part of the scientific question.

### Few-atom QRC
Zhu et al. (arXiv:2405.04799) show that useful performance can arise from very small reservoirs when memory and nonlinear processing are measured directly. Small hardware does not remove the need to establish what information the reservoir retains and transforms.

### Chaotic forecasting and generalized synchronization
Ahmed et al. (arXiv:2506.22335) treat QRC as a driven dynamical system. Stability, synchronization, and faithful reproduction of invariant dynamics matter; point forecast error alone is not enough to understand whether the reservoir is working.

### Memory-nonlinearity tradeoff
The supplied 2026 memory/nonlinearity work reinforces that QRC design cannot maximize memory and nonlinearity independently. Input timescale, interaction strength, reservoir timescale, and readout must be matched to the task.

### Experimental spin QRC
Hou et al. (arXiv:2508.12383) use correlated spin dynamics and time-multiplexed readout. Their results reinforce that readout capacity and experimentally available observables are part of the architecture, not an afterthought.

### Financial lag-embedding study
Maheshwari et al. (arXiv:2605.02656) report that lag structure and multivariate correlated inputs strongly affect QRC/QLSTM performance. This supports treating input representation and lag design as first-class experimental variables.

### VOLARE
VOLARE provides standardized realized estimators from ultra-high-frequency data, including realized variance, bipower variation, semivariances, quarticity, realized kernels, and covariance measures. It is useful as a comparison dataset because it changes the measurement quality and information content of the volatility target, but access constraints make it unsuitable as the sole final reproducible dataset.

## Data-design conclusions

1. The primary task should begin from the anchor paper's monthly benchmark because it is challenge-grounded, public-data reproducible, and directly comparable to published QRC results.
2. VOLARE should be used as a controlled comparison of richer realized-volatility measurement, not as the sole final dataset.
3. The old daily SPY/VIX data should remain isolated as a legacy comparison dataset only.
4. The data pipeline must distinguish raw missingness, expected warm-up NaNs, future-target censoring, and diagnostic censoring. No silent dropping is allowed.
5. All model comparisons must use identical dates and a declared selection metric matched to the reported objective.
6. ESN is the primary classical baseline. The cheap baseline is only a sanity check. HAR is diagnostic, not the project target.
7. Rydberg work starts early, in parallel with classical refinement, because the scientific goal is quantum advantage that survives shots/noise and because hardware practice cannot be postponed.

## Sources consulted

- Challenge brief supplied with the project.
- Li et al., arXiv:2505.13933.
- Kornjaca et al., arXiv:2407.02553.
- Zhu et al., arXiv:2405.04799.
- Ahmed et al., arXiv:2506.22335.
- Hou et al., arXiv:2508.12383.
- Maheshwari et al., arXiv:2605.02656.
- VOLARE paper, arXiv:2602.19732.
- Official data sources planned: Yahoo public chart data, FRED, Kenneth French Data Library, and Shiller data after exact field-definition verification.
