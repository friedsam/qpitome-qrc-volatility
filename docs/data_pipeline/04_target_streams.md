# Target streams: paper sanity benchmark versus challenge objective

## Rule

The anchor paper is a calibration experiment, not the project definition.

The repository keeps two distinct modeling streams:

1. `paper_monthly` — literature-grounded sanity benchmark;
2. `challenge_primary` — competition task, now frozen at monthly next-period realized volatility with transition-focused secondary evaluation.

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

This set is not exact QR1 or QR2 because DP/EP and exact paper definitions remain unresolved. It is not automatically the final challenge input set.

## Paper benchmark conclusion: closed enough

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
- the paper sanity branch remains a controlled reference and may be reused for architecture comparisons.

## Temporal-scale diagnostic

A controlled daily/weekly/monthly comparison used natural non-overlapping next-period targets and gave HAR and ESN the same three volatility-state inputs.

Result:

```text
daily   HAR RMSE 1.242969   ESN RMSE 1.241045
weekly  HAR RMSE 0.488279   ESN RMSE 0.491261
monthly HAR RMSE 0.425101   ESN RMSE 0.427950
```

Target lag-1 autocorrelation increased strongly with aggregation:

```text
daily   0.113
weekly  0.544
monthly 0.658
```

Interpretation:

- daily absolute-return volatility is mostly noise;
- weekly and monthly aggregation reveal persistent state;
- changing temporal scale alone does not create classical reservoir headroom;
- univariate volatility memory is not the most promising source of reservoir advantage;
- the more plausible exploit is interaction among volatility state, market state, credit stress, and macro dynamics.

This closes broad temporal-scale search.

# Challenge-primary stream: target frozen

## Primary target

For prediction origin at the end of calendar month `t`, forecast realized volatility in calendar month `t+1`:

```text
y_(t+1) = log(sqrt(sum of squared daily log returns in month t+1))
```

Equivalent variance-space quantity for QLIKE:

```text
V_(t+1) = exp(2 * y_(t+1))
```

Rules:

- natural calendar-month aggregation;
- no overlapping future target windows;
- target constructed from the same public S&P 500 daily series used by the primary data pipeline;
- information available through the end of month `t` may be used;
- no data from month `t+1` may enter features, scaling, fitting, model selection, or trading-day estimation;
- the primary prediction remains continuous regression, not crisis classification.

The target is now frozen. It changes only if a concrete implementation bug is found.

## Why monthly

The decision is empirical and challenge-driven:

- monthly volatility has the strongest memory among the natural daily/weekly/monthly scales tested;
- daily targets are largely noise;
- weekly aggregation offers more data but no ESN gain over HAR;
- monthly scale aligns volatility memory with market, credit, and macro state variables;
- the paper provides a relevant controlled QRC reference on the same target family;
- the challenge allows a chosen realized-volatility horizon and emphasizes regime shifts rather than requiring crisis classification.

## Primary metrics

Every final model is evaluated on identical forecast dates with:

- RMSE in log-RV space;
- MAE in log-RV space;
- QLIKE in variance space;
- Mincer-Zarnowitz intercept, slope, and R²;
- runtime and resource usage where relevant.

## Development and final temporal holdout

To prevent target and architecture choices from being optimized on recent crises, the challenge stream uses two temporal stages.

### Development/comparison period

```text
forecast dates: 1997-08 through 2017-12
```

Purpose:

- direct comparison with the paper-era benchmark;
- target diagnostics;
- feature and architecture ablations;
- HAR/HARX, ESN, GARCH, LSTM, TFIM, and Rydberg development;
- Phase 2-style diagnostic plots.

The existing 245-date paper protocol is reused as the development comparison backbone where compatible.

### Final temporal holdout

```text
forecast dates: 2018-01 through the latest complete target month
currently expected through 2026-06, subject only to verified source completeness
```

Rules:

- no target, feature-family, architecture, or hyperparameter choice may be made because of final-holdout performance;
- the holdout is opened only after the development protocol and candidate models are frozen enough for final comparison;
- all models must use identical holdout forecast dates;
- unavailable inputs are failures or explicitly handled missingness cases, not silent date deletions.

This period includes materially different regimes and recent transition episodes, which strengthens stakeholder relevance.

## Training rule

The default benchmark uses causal walk-forward re-estimation.

For each forecast date:

- model information ends at the previous month-end prediction origin;
- all scalers, PCA, feature selection, clipping, thresholds, and readout fitting use training data only;
- the development backbone retains the existing rolling 569-month calendar window unless a model requires a different native history rule that is declared in advance;
- model-native exceptions, such as a daily-return GARCH history window, must be explicit and may not inspect future results for selection.

No model may silently use a different forecast-date set.

## Input families for the first serious reservoir comparison

The first controlled comparison must separate:

### Volatility-state only

```text
1-month state
3-month state
12-month state
```

### Multivariate external-state version

At minimum, resolved public variables representing:

```text
volatility state
market shock / reversal
credit stress
rates
macro dynamics
```

The exact final compact QRC input map remains to be selected causally, but the scientific comparison is fixed:

```text
volatility memory only
versus
volatility + external state
```

This directly tests the paper-like hypothesis that reservoir value comes from nonlinear interaction among external drivers and volatility state rather than from univariate volatility memory alone.

## Transition-focused secondary evaluation

Transition analysis is secondary to the continuous forecast and may not redefine the target.

Initial descriptive states:

```text
stable calm
calm -> elevated
stable elevated
elevated -> calm
largest upward transitions
```

Final thresholds must be estimated from training data only within each fold or from a predeclared development-period rule. Full-sample thresholds are prohibited for model evaluation.

Transition reporting should answer:

- where HAR/HARX fails;
- where ESN or QRC gains;
- whether gains occur before or during state changes;
- whether a gain is concentrated in one crisis or persists across episodes.

## Active next steps

1. generate Phase 2-style target and baseline diagnostic figures on the development period;
2. inspect where HARX and ESN differ through time and across transition states;
3. rebuild the frozen Phase 2 TFIM control against this monthly task;
4. add the same diagnostic figures with TFIM;
5. integrate corrected GARCH and LSTM baselines;
6. freeze compact multivariate input candidates for TFIM/Rydberg;
7. move to finite-shot, noise, size-scaling, and hardware studies.
