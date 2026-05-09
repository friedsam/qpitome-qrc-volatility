# Data Sources

## Purpose

This project uses public financial time-series data for volatility-regime / market-stress detection.

The v0 dataset uses:

- SPY daily OHLCV data
- VIX daily historical data

The goal is to build features such as SPY returns, realized volatility, intraday range, volume, dollar volume, drawdown, VIX level, and VIX change.

## Raw Data Files

Expected local files:

```text
data/raw/kaggle_market_ohlcv/SPY.csv
data/raw/vix/VIX_History.csv
```

Optional files downloaded with the same Kaggle dataset:

```text
data/raw/kaggle_market_ohlcv/SP500.csv
data/raw/kaggle_market_ohlcv/QQQ_raw.csv
data/raw/kaggle_market_ohlcv/QQQ_split_adj.csv
data/raw/kaggle_market_ohlcv/NASDAQ_100.csv
```

These optional files are not required for v0 but may support later cross-asset extensions.

## Source 1: SPY OHLCV

Source: Kaggle dataset  
Dataset: `guillemservera/sp500-nasdaq-spy-qqq-ohlcv-data`  
Contents: daily OHLCV data for SPY, QQQ, S&P 500, and NASDAQ 100  
Access method: manual Kaggle download  
Local path: `data/raw/kaggle_market_ohlcv/`

Download procedure:

1. Open Kaggle.
2. Search for `guillemservera/sp500-nasdaq-spy-qqq-ohlcv-data`.
3. Download and unzip the dataset.
4. Copy the files into `data/raw/kaggle_market_ohlcv/`.
5. Confirm that `SPY.csv` exists.

## Source 2: VIX

Source: Cboe VIX Historical Data  
Contents: VIX daily historical data  
Access method: manual CSV download  
Local path: `data/raw/vix/VIX_History.csv`

Download procedure:

1. Open Cboe VIX Historical Data.
2. Download `VIX Index data for 1990 to present`.
3. Save the CSV as `data/raw/vix/VIX_History.csv`.

## Versioning Policy

Raw data files are not committed to Git.

The repository commits:

- data-source documentation
- inspection scripts
- processing scripts
- generated feature definitions

The repository does not commit:

- downloaded raw CSV files
- processed data files
- large result files

## v0 Dataset Definition

Required inputs:

```text
SPY.csv
VIX_History.csv
```

Required processed output:

```text
data/processed/market_stress_v0.csv
```

Planned columns:

```text
date
spy_open
spy_high
spy_low
spy_close
spy_adj_close
spy_volume
spy_return
spy_abs_return
spy_range
spy_dollar_volume
spy_drawdown_20d
rv_5d
rv_10d
rv_20d
vix_close
vix_change
vix_ma_5d
future_rv_5d
future_high_vol_label
```

## Notes

The initial v0 dataset deliberately uses SPY and VIX only. This keeps the first benchmark aligned with the Phase 1 proposal while avoiding early scope creep.

Optional extensions may add QQQ, S&P 500 index data, NASDAQ 100 data, or cross-asset stress proxies after the v0 baseline works.

## Processed Dataset Creation

The processed v0 dataset is generated from:

```text
data/raw/kaggle_market_ohlcv/SPY.csv
data/raw/vix/VIX_History.csv
```

Run from the repository root:

```bash
python scripts/prepare_dataset.py
```

This creates:

```text
data/processed/market_stress_v0.csv
```

The processed file is not committed to Git.

### Current processed dataset

As of the first successful run:

```text
Rows: 7,830
Columns: 22
Date range: 1993-03-01 to 2024-04-08
High-volatility labels: 1,570
Non-high-volatility labels: 6,260
```

The label balance is approximately:

```text
high-volatility class: 20%
non-high-volatility class: 80%
```

This is expected because the provisional high-volatility label uses the 80th percentile of future 5-day realized volatility.

### Feature construction

The script computes:

```text
SPY adjusted-close returns
absolute returns
intraday range proxy: (high - low) / close
dollar volume: adjusted_close × volume
rolling realized volatility: 5d, 10d, 20d
20-day drawdown
VIX daily change
VIX percent change
VIX 5-day moving average
future 5-day realized volatility
binary future high-volatility label
```

### Label definition

The provisional target is:

```text
future_rv_5d = annualized realized volatility over the next 5 trading days
future_high_vol_label = 1 if future_rv_5d > 80th percentile threshold
```

Current provisional threshold:

```text
future_rv_5d 80th percentile = 0.20885475004200635
```

### Leakage warning

The current processed file uses a provisional threshold computed over all available rows. This is acceptable for raw exploration only.

For model training and final benchmarking, the threshold must be recomputed using the training split only, then applied to validation and test data. This avoids using future/test-distribution information when defining labels.