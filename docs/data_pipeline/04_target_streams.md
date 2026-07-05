# Target streams: paper sanity benchmark versus challenge objective

## Rule

The anchor paper is a calibration experiment, not the project definition.

The repository keeps two distinct modeling streams:

1. `paper_monthly` — literature-grounded sanity benchmark;
2. `challenge_primary` — competition task, defined from the challenge objective and evaluated for volatility-regime transition performance.

Results, targets, feature decisions, and conclusions must not be silently transferred between the two streams.

## Paper sanity benchmark

Target:

```text
features available through month t -> log realized volatility in month t+1
```

Protocol:

- exact 815-month calendar slice, February 1950-December 2017;
- 245 monthly forecasts, August 1997-December 2017;
- rolling 569-month calendar window;
- common causal preprocessing rules.

Purpose:

- verify data and forecasting implementation;
- reproduce qualitative paper findings where possible;
- test whether the ESN implementation is fundamentally competent;
- obtain feature/encoding clues for QRC;
- establish a controlled benchmark before challenge-specific design.

The paper-informed compact seven-feature proxy is:

```text
vol_state_1m
vol_state_3m_mean
market_mkt_excess
market_str
credit_default_spread_baa_minus_aaa_level
macro_ip_growth_lag1
macro_inflation_growth_lag1
```

This set is not exact QR1 or QR2 because DP/EP and exact paper definitions remain unresolved. It is not the final challenge input set.

## Paper benchmark conclusion: closed

Key one-step results:

| Model | RMSE log RV | QLIKE |
|---|---:|---:|
| ESN compact7, 50 nodes, alpha 100, seed mean | 0.323634 | 0.277353 |
| HARX public available | 0.322857 | 0.275950 |
| Ridge compact 7 | 0.326019 | 0.285782 |
| Ridge all 24 | 0.326671 | 0.275446 |
| HAR | 0.339656 | 0.304060 |
| AR(1) | 0.360590 | 0.352500 |
| Persistence | 0.377906 | 0.360259 |
| Original fixed ESN all 24 | 0.420115 | 0.504364 |
| Original fixed ESN compact 7 | 0.488163 | 0.899482 |

The first arbitrary ESN failed because the reservoir/readout design was too aggressive for the sample size: 200 nodes, 225-dimensional skip-connected readout, and alpha 0.001.

A controlled autopsy showed:

- reducing the reservoir to 50 nodes recovered performance;
- skip-connected inputs plus states outperformed states-only readout;
- stronger regularization stabilized the skip-connected readout;
- the compact seven-feature inputs were not the cause of the original failure.

Frozen sanity configuration:

```text
compact7
50 reservoir nodes
spectral radius 0.9
input scale 0.5
leak 0.3
readout alpha 100
inputs + reservoir states
washout 24
```

Five-seed stability check, seeds 0-4:

```text
RMSE mean  0.323634
RMSE sd    0.001607
RMSE range 0.321868-0.325848
QLIKE mean 0.277353
```

Conclusion:

- the ESN implementation is functional;
- the recovered ESN is stable and reaches the HARX performance tier;
- it does not establish a meaningful advantage over HARX or compact Ridge;
- no further paper-target ESN optimization is justified now;
- the paper sanity branch is closed unless a later challenge result creates a specific reason to reopen it.

## Challenge-primary stream

This is now the active scientific stream.

The challenge target remains intentionally unresolved until challenge-specific target analysis is completed. The challenge framing is broader than next-month paper parity and emphasizes volatility-regime shifts and transition forecasting.

Current design direction:

- retain a continuous volatility forecast as the primary quantity;
- evaluate overall RMSE/QLIKE and calibration;
- add explicit transition-focused slices such as calm-to-turbulent and turbulent-to-calm periods;
- measure whether a model gains around regime transitions rather than only in persistent regimes;
- define horizon and regime labels before challenge-model optimization;
- carry forward paper lessons only as hypotheses: compact inputs, multiscale state, exogenous stress, and capacity control.

No paper-parity result alone is sufficient evidence for the final challenge model.
