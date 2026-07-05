# Target streams: paper sanity benchmark versus challenge objective

## Rule

The anchor paper is a calibration experiment, not the project definition.

The repository must keep two distinct modeling streams:

1. `paper_monthly` — literature-grounded sanity benchmark;
2. `challenge_primary` — competition task, to be defined from the challenge objective and evaluated for volatility-regime transition performance.

Results, targets, feature decisions, and conclusions must not be silently transferred between the two streams.

## Paper sanity benchmark

Current target:

```text
features available through month t -> log realized volatility in month t+1
```

Current one-step protocol:

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

Run compact linear sanity check through the existing baseline runner:

```bash
python scripts/models/run_sanity_baselines.py
```

Run compact ESN sanity check through the existing ESN runner:

```bash
python scripts/models/run_esn_fixed.py --feature-set compact7
```

The full 24-feature ESN remains available as:

```bash
python scripts/models/run_esn_fixed.py --feature-set all24
```

## Challenge-primary stream

The challenge target remains intentionally unresolved at this stage.

The challenge framing is broader than next-month paper parity and emphasizes volatility-regime shifts and transition forecasting. The final task should therefore be defined from challenge-specific analysis rather than inherited from the paper.

Current design direction:

- retain a continuous volatility forecast as the primary quantity;
- evaluate overall RMSE/QLIKE and calibration;
- add explicit transition-focused slices such as calm-to-turbulent and turbulent-to-calm periods;
- measure whether a model gains around regime transitions rather than only in persistent regimes;
- define horizon and regime labels before challenge-model optimization.

No paper-parity result alone is sufficient evidence for the final challenge model.

## Current paper benchmark status

Observed one-step results before the compact sanity check:

| Model | RMSE log RV | QLIKE |
|---|---:|---:|
| HARX public available | 0.322857 | 0.275950 |
| Ridge all 24 | 0.326671 | 0.275446 |
| HAR | 0.339656 | 0.304060 |
| AR(1) | 0.360590 | 0.352500 |
| Persistence | 0.377906 | 0.360259 |
| Fixed ESN all 24 | 0.420115 | 0.504364 |

Interpretation:

- multiscale volatility memory matters;
- resolved exogenous inputs add value;
- compact HARX and full 24-feature Ridge are in the same performance tier;
- the first arbitrary 200-node ESN configuration failed clearly;
- the compact seven-feature run is a sanity check on input burden and ESN implementation, not a final ESN search.
