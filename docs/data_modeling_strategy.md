# Data Modeling Strategy

Date: May 21, 2026  
Branch: `phase2-volatility-regression-qrc`

## Dataset decision

Preferred datasets, pending access:

- VOLARE realized-volatility archive

Executable fallback dataset:

- `data/raw/kaggle_market_ohlcv/SPY.csv`
- `data/raw/vix/VIX_History.csv`

We will proceed with the fallback dataset now because access to realized-volatility archives is pending and Phase 2 must not depend on external approval.

## Fallback dataset

Instrument:

- SPY ETF daily OHLCV
- Date range: 1993-01-29 to 2024-03-15
- Frequency: daily trading days
- Primary price column: `adjusted_close`

Exogenous volatility signal:

- VIX daily OHLC
- Date range: 1990-01-02 to 2026-05-07
- Overlaps the full SPY period

## Target construction

Compute SPY daily log returns:

```python
r_t = log(adjusted_close_t / adjusted_close_{t-1})
```

Construct realized-volatility proxies from squared daily log returns:

```python
rv_5d_t  = sqrt(252 / 5  * sum_{i=t-4}^{t} r_i^2)
rv_20d_t = sqrt(252 / 20 * sum_{i=t-19}^{t} r_i^2)
```

Forecast targets:

```python
future_rv_5d_t  = sqrt(252 / 5  * sum_{i=t+1}^{t+5} r_i^2)
future_rv_20d_t = sqrt(252 / 20 * sum_{i=t+1}^{t+20} r_i^2)
```

## Initial features

SPY-derived features:

- log return
- absolute log return
- squared log return
- log high-low range
- log volume
- volume change
- rolling realized volatility: 5d, 10d, 20d, 60d

VIX-derived features:

- VIX close
- VIX log change
- VIX high-low range
- VIX rolling mean: 5d, 20d
- VIX rolling standard deviation: 5d, 20d

## Regime-transition layer

Regime-transition warning is derived from the realized-volatility path.

Initial transparent rule:

- calm now: current `rv_20d` below a training-set calm threshold
- turbulent future: future `rv_20d` above a training-set turbulent threshold
- transition event: calm now and turbulent within the forecast horizon

Thresholds must be estimated from training data only.

This regime-transition layer is the Track A problem framing. It does not replace the required volatility-forecasting metrics.

## Required Track A metrics

Primary metrics:

- RMSE
- QLIKE
- Mincer-Zarnowitz regression

Secondary interpretation:

- regime-transition warning examples
- event-level behavior
- possible lead-time / false-warning discussion

## Splits

Use chronological splits only.

Initial split:

- train: 1993-01-29 to 2014-12-31
- validation: 2015-01-01 to 2019-12-31
- test: 2020-01-01 to 2024-04-15

## Leakage rules

Allowed:

- future returns only for target construction
- training-set statistics for normalization
- training-set statistics for regime thresholds
- training-set fit for PCA or feature selection

Forbidden:

- random splits
- fitting scalers on validation or test data
- fitting PCA on validation or test data
- defining regime thresholds using the full dataset
- using future VIX or future volatility features as model inputs

## Phase 2 stance

This fallback dataset is acceptable for Phase 2 because it is public, reproducible, and uses challenge-compatible equity-market data. Its limitation is that realized volatility is reconstructed from daily prices rather than computed from high-frequency intraday data. If VOLARE access becomes available, the loader should be swapped while preserving the same target/metric interface.

## Split and transition check

Processed fallback dataset:

- file: `data/processed/phase2_spy_vix_volatility.csv`
- shape: 7775 rows × 41 columns
- usable date range: 1993-04-27 to 2024-03-15

Chronological split:

- train: 1993-04-27 to 2014-12-31, n = 5459
- validation: 2015-01-02 to 2019-12-31, n = 1258
- test: 2020-01-02 to 2024-03-15, n = 1058

Train-only regime thresholds:

- calm threshold: `rv_20d` q50 = 0.13897
- turbulent threshold: `future_rv_20d` q80 = 0.21256

Transition-event rates:

- train: 88 / 5459 = 1.6%
- validation: 57 / 1258 = 4.5%
- test: 28 / 1058 = 2.6%

Interpretation:

The transition label is useful as an event-warning interpretation layer, but it is too sparse to serve as the only Phase 2 supervised target. The main modeling target remains realized-volatility forecasting, evaluated with RMSE, QLIKE, and Mincer-Zarnowitz. Regime-transition warnings are derived from the forecast path using train-only thresholds.

## May 22 validation

The Phase 2 data/metrics infrastructure was validated using the processed fallback dataset:

- processed file: `data/processed/phase2_spy_vix_volatility.csv`
- loader: `load_phase2_volatility_data()`
- split utility: `chronological_tabular_split()`
- transition utility: `add_train_only_transition_flags()`
- metrics: RMSE, QLIKE, Mincer-Zarnowitz

A naive persistence forecast was evaluated as an initial baseline floor:

- `rv_5d` or `rv_10d` predicting `future_rv_5d`
- `rv_20d` or `rv_60d` predicting `future_rv_20d`

Best observed persistence results:

| target | split | best persistence predictor | RMSE | QLIKE | MZ R² |
|---|---:|---|---:|---:|---:|
| `future_rv_5d` | validation | `rv_10d` | 0.06846 | -2.79026 | 0.28200 |
| `future_rv_5d` | test | `rv_10d` | 0.10246 | -2.22157 | 0.50171 |
| `future_rv_20d` | validation | `rv_60d` | 0.06484 | -2.88638 | 0.09657 |
| `future_rv_20d` | test | `rv_20d` | 0.12431 | -2.00401 | 0.25156 |

Interpretation:

Persistence is a meaningful baseline because volatility clusters, but validation/test behavior differs substantially. This supports the challenge framing: volatility-regime forecasting is nonstationary and difficult. Future baselines and QRC prototypes should be compared against this persistence floor before making stronger claims.

### Model-ready feature and scaling validation

The Phase 2 SPY+VIX pipeline now produces leakage-safe model inputs for both tabular regression baselines and sequence-based reservoir models.

The finalized model interface contains 27 feature columns and two continuous realized-volatility targets:

- `future_rv_5d`
- `future_rv_20d`

Feature normalization is performed after chronological splitting. The scaler is fit on the training split only and then applied unchanged to validation and test splits. This preserves the leakage rule that validation and test statistics must not influence preprocessing, normalization, thresholding, or model fitting.

For target `future_rv_20d`, the validated tabular arrays are:

| Split | X shape | y shape |
|---|---:|---:|
| Train | `(5459, 27)` | `(5459,)` |
| Validation | `(1258, 27)` | `(1258,)` |
| Test | `(1058, 27)` | `(1058,)` |

The validated configuration is:

```text
target = future_rv_20d
feature_count = 27
scaler = standard
```

For sequence-based ESN/QRC-style models, a 20-trading-day lookback produces:

```text
X_train_sequence = (5440, 20, 27)
y_train_sequence = (5440,)
first_sequence_target_date = 1993-05-24
last_sequence_target_date = 2014-12-31
```

The sequence count is correct:

```text
5459 train rows - 20 lookback + 1 = 5440 sequences
```

This completes the May 22 data-pipeline infrastructure needed before May 23 classical baselines.