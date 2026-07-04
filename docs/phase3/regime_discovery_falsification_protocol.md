# Regime Discovery Falsification Protocol

Date: 2026-07-04
Status: preregistered diagnostic protocol before inspecting economic-separation results

## Purpose

The project must not invent market regimes from a small number of famous episodes or from labels chosen because they improve a model score. This audit asks a prior question:

> Does the existing SPY/VIX dataset contain persistent, statistically reproducible market states whose transitions are associated with materially different future gain/loss outcomes?

Only if the answer is yes should ESN/QRC transition forecasting be attempted.

## Stakeholder alignment

A regime is not defined as simply “high volatility” or “turbulent.” A state or transition is useful only if it changes an economically relevant decision for trading desks, market makers, portfolio managers, risk managers, derivatives desks, or risk-control teams.

The audit therefore evaluates future outcomes that map to gains and losses:

- forward SPY return over 5, 10, and 20 trading days;
- worst forward drawdown over the next 20 trading days;
- best forward upside over the next 20 trading days;
- future 20-day realized volatility as a risk descriptor, not as the sole regime definition.

## Non-negotiable anti-overfitting rules

1. Regime definitions use only information available at time t.
2. Unsupervised scalers and cluster models are fit on the original Phase 2 training period only (through 2014-12-31).
3. Future returns, drawdowns, and volatility are never used to construct state labels.
4. Candidate definitions are specified before inspecting outcome-separation results.
5. All tested candidate-definition × outcome × evaluation-period hypotheses are corrected with Benjamini–Hochberg FDR.
6. Daily serial dependence is handled with block permutation rather than iid label permutation.
7. A state is rejected as too small if its occupancy is below 5% in the full sample or if it disappears in a major evaluation period.
8. A “regime” must show persistence beyond daily noise. Report self-transition probability and run-length distribution.
9. Transition counts are episode counts, not raw adjacent-day label flips. Repeated flips inside a short cooldown window are not treated as independent evidence.
10. A candidate is not considered robust because of one crisis period. Results are reported separately for 1993–2004, 2005–2014, 2015–2019, and 2020–2024.

## Candidate state definitions

These are deliberately limited in number.

### A. Volatility level × acceleration grid

Inputs:

- current `rv_20d`;
- current `rv_ratio_5_20`.

Train-only quantile bins produce a small interpretable state space. This tests whether current volatility level and short-vs-medium acceleration define persistent states with different future outcomes.

### B. Four-dimensional market-risk clustering

Train-only standardized inputs:

- `rv_20d`;
- `vix_close`;
- `spy_drawdown_20d`;
- `rv_ratio_5_20`.

K-means is fit for k = 2, 3, 4, and 5. These models test whether a compact multivariate state description is more useful than a single volatility threshold.

### C. Eight-dimensional market-state clustering

Train-only standardized inputs:

- `rv_20d`;
- `rv_ratio_5_20`;
- `rv_ratio_20_60`;
- `spy_drawdown_20d`;
- `vix_close`;
- `vix_log_change`;
- `spy_log_hl_range`;
- `spy_log_volume_change`.

K-means is fit for k = 2, 3, 4, and 5. This tests whether richer joint geometry creates meaningful states, while explicitly penalizing unstable or tiny clusters.

## Regime-quality tests

For each candidate definition:

### Persistence

Report:

- state occupancy;
- one-day self-transition probability;
- median run length;
- 90th-percentile run length;
- number of distinct runs.

A useful regime should persist long enough to be more than day-to-day classification noise.

### Economic separation

For each future outcome and evaluation period, calculate an omnibus between-state separation statistic and a block-permutation p-value.

The permutation null shifts/shuffles contiguous state-label blocks rather than individual days, reducing false significance caused by serial dependence.

Report:

- observed between-state effect statistic;
- permutation p-value;
- FDR-adjusted q-value;
- state-wise means, medians, and sample counts.

### Temporal stability

A candidate must be examined separately in:

- 1993-04-27 to 2004-12-31;
- 2005-01-01 to 2014-12-31;
- 2015-01-01 to 2019-12-31;
- 2020-01-01 to 2024-03-15.

A state definition that is economically meaningful only during 2020 is not considered a general market regime.

### Transition feasibility

For every directed state transition A → B, report:

- raw transition-day count;
- episode count after a 5-trading-day cooldown;
- counts by evaluation period;
- subsequent return, drawdown, upside, and volatility distributions.

Directed transitions with too few independent episodes are rejected as modeling targets even if their apparent effect is large.

## Conservative decision criteria

A candidate state system is a plausible basis for transition forecasting only if all of the following hold:

1. no state has full-sample occupancy below 5%;
2. states show nontrivial persistence (self-transition probability and run lengths above daily-noise behavior);
3. at least one gain/loss-relevant future outcome survives FDR correction out of sample;
4. the economic ordering is not driven solely by one evaluation period;
5. at least one directed transition has enough independent episodes for modeling (target guideline: at least 50 episodes overall and meaningful representation across multiple periods).

If no candidate passes, the correct conclusion is that this dataset does not support a defensible regime-transition target and the project should not manufacture one.

## Outputs

The analysis script writes:

- `results/diagnostics/regime_discovery/state_persistence.csv`
- `results/diagnostics/regime_discovery/economic_separation_tests.csv`
- `results/diagnostics/regime_discovery/state_outcome_summaries.csv`
- `results/diagnostics/regime_discovery/transition_feasibility.csv`
- `results/diagnostics/regime_discovery/regime_discovery_report.md`

The report must include rejected candidates and reasons for rejection, not only the best-looking result.
