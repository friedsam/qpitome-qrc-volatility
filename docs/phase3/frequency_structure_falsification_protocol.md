# Frequency Structure Falsification Protocol

Date: 2026-07-04
Status: preregistered before inspecting spectral results

## Question

Do the SPY/VIX state variables or surviving regime-state sequences contain reproducible slow-moving temporal structure that is stronger than expected from generic persistence and autocorrelation?

This is not a search for a visually attractive market cycle. A spectral peak is not accepted unless it survives null tests and appears with compatible time scales across multiple historical periods.

## Series tested

Continuous observables:

- `rv_20d`
- `rv_ratio_5_20`
- `rv_ratio_20_60`
- `vix_close`
- `spy_drawdown_20d`

Derived channel added exactly from committed data:

- `vix_rv_spread = log(vix_close / 100) - log(rv_20d)`

Regime sequences are reconstructed exactly for the four candidates that passed the coarse regime screen:

- `state__risk4_k2`
- `state__risk4_k3`
- `state__state8_k2`
- `state__vol_level_x_acceleration_grid`

For multi-state regimes, each state is converted to a one-vs-rest binary occupancy sequence before spectral analysis. Integer cluster labels are never treated as metric-valued signals.

## Methods

### 1. Autocorrelation

For each continuous series and binary state occupancy sequence, calculate the autocorrelation function to 1,000 trading days where sample length permits.

Report characteristic decay lags:

- first lag below 1/e;
- first zero crossing;
- integrated positive autocorrelation time.

These quantify persistence without assuming periodicity.

### 2. Welch power spectral density

Use a detrended Welch periodogram rather than a raw whole-sample FFT.

- sampling frequency: 1 observation per trading day;
- segment length: up to 1,024 observations, reduced for shorter periods;
- 50% overlap;
- linear detrending;
- Hann window.

Search periods from 20 to 1,500 trading days. The low-frequency boundary avoids treating the total sample trend as a cycle; the high-frequency boundary avoids conflating ordinary short-horizon fluctuations with the slow-pattern hypothesis.

### 3. Peak prominence against block surrogates

For each series and historical period:

1. divide the observed sequence into contiguous 60-trading-day blocks;
2. randomly permute block order;
3. recompute the Welch PSD;
4. compare observed peak power and band-integrated power against surrogate distributions.

This null preserves much local serial structure while destroying long-range phase organization.

A candidate frequency is not accepted merely because ordinary PSD power is large. It must exceed the empirical surrogate distribution.

### 4. Cross-period stability

Run the analysis separately for:

- 1993–2004;
- 2005–2014;
- 2015–2019;
- 2020–2024;
- full sample.

A putative slow time scale is considered stable only if compatible peaks appear in at least two non-overlapping major periods and the full-sample result is not solely driven by 2020–2024.

### 5. Broad frequency bands

Aggregate power into predeclared bands:

- 20–60 trading days;
- 60–125 days;
- 125–250 days;
- 250–500 days;
- 500–1,000 days;
- 1,000–1,500 days.

Band-power surrogate tests are more robust than exact-bin peaks and are the primary evidence for slow structure.

## Conservative interpretation rules

A slow-moving pattern is considered potentially real only if:

1. the relevant band or peak exceeds the 95th percentile of block-surrogate power;
2. a compatible period/band recurs in at least two non-overlapping historical periods;
3. the result is not confined to one crisis episode;
4. the same time scale is visible in at least one economically interpretable observable or regime occupancy sequence;
5. the pattern is not merely the expected consequence of very high autocorrelation.

If these conditions fail, the correct conclusion is that the data show persistence but no defensible slow cycle.

## Outputs

The script writes to `results/diagnostics/frequency_structure/`:

- `acf_summary.csv`
- `spectral_peaks.csv`
- `band_power_tests.csv`
- `cross_period_stability.csv`
- `run_manifest.json`
- `frequency_structure_report.md`

The report must distinguish:

- persistence;
- broad low-frequency structure;
- reproducible quasi-periodicity;
- unsupported cycle claims.
