# Data retrieval and processing

This directory contains the raw inputs, frozen fallback copies, processed datasets, and manifests used by the QRC volatility project.

The data pipeline is designed so that a new checkout can repopulate the `data/` tree from the repository scripts. Retrieval is remote-first. When a remote source is unavailable, the downloader copies the corresponding frozen file from `data/fallback/` into `data/raw/`. Fallback files are never overwritten by the download scripts.

## Directory layout

```text
data/
├── fallback/   # Frozen known-good source files used only if retrieval fails
├── raw/        # Canonical downloaded or fallback-copied source files
├── processed/  # Analysis-ready datasets built from raw data
└── manifest.json
```

Important manifests include:

- `data/raw/yahoo_daily_history/manifest.json`: SPY and VIX retrieval status, date ranges, row counts, hashes, and fallback use.
- `data/raw/external_dataset_manifest.json`: retrieval status for macro, factor, index, and S&P 500 source files.
- `data/raw/raw_dataset_inventory.json`: inventory and validation summary for the expected raw datasets.
- processed-data manifests under each processed dataset directory.

## Data sources

### Daily US market and volatility data

The primary daily market dataset uses Yahoo Finance chart data:

- SPY: S&P 500 ETF market history.
- `^VIX`: CBOE Volatility Index history.

These files are written to:

```text
data/raw/yahoo_daily_history/historical_market_data.csv
data/raw/yahoo_daily_history/historical_volatility_data.csv
```

SPY and VIX are joined by trading date and used to build the main daily volatility dataset.

### Long-history S&P 500 data

Yahoo Finance `^GSPC` history is used to construct the monthly realized-volatility dataset beginning in 1950. It is stored as:

```text
data/raw/monthly_market_features/sp500_yahoo_raw.csv
```

The original research snapshot used a different S&P 500 source. The current Yahoo-based series reproduces the archived monthly processed data to floating-point precision over the fixed paper period, while avoiding dependence on the earlier external snapshot.

### FRED macroeconomic and credit series

The Federal Reserve Economic Data service supplies:

- `TB3MS`: 3-month Treasury bill rate.
- `CPIAUCSL`: Consumer Price Index.
- `INDPRO`: Industrial Production Index.
- `AAA`: Moody's seasoned AAA corporate bond yield.
- `BAA`: Moody's seasoned BAA corporate bond yield.
- `NIKKEI225`: Nikkei 225 index series.

Files are stored under:

```text
data/raw/macro_fred_monthly/
data/raw/international_equity_indices/
```

### Kenneth French Data Library

Monthly US equity factors are downloaded from the Kenneth French Data Library:

- Fama-French market, SMB, HML, and risk-free-rate series.
- Short-term reversal factor.

The original ZIP archives are retained under:

```text
data/raw/equity_factor_returns_monthly/
```

### Additional international and small-cap indices

Yahoo Finance supplies:

- `^RUT`: Russell 2000.
- `^FTSE`: FTSE 100.

These raw files are stored under:

```text
data/raw/international_equity_indices/
```

They are retained as canonical external inputs even when they are not required by the current Stage 1 processed datasets.

## Retrieval policy

The downloaders use the following policy:

1. Attempt the remote source.
2. Validate the downloaded file before accepting it.
3. If retrieval or validation fails, copy the matching frozen fallback file.
4. Validate the copied fallback file.
5. Record the source status and error details in a manifest.

A fallback file is therefore a read-only recovery source, not a cache. Running a downloader does not modify `data/fallback/`.

## Major processing steps

### SPY/VIX daily volatility dataset

The daily builder performs the following operations:

1. Parse and validate SPY and VIX dates and required columns.
2. Sort observations and align the two sources by trading date.
3. Standardize column names.
4. Compute SPY log returns, absolute returns, squared returns, high-low range, and open-close return.
5. Compute Parkinson and Garman-Klass range-based variance estimators.
6. Compute volume and dollar-volume transformations.
7. Compute rolling realized volatility over 5, 10, 20, and 60 trading days.
8. Compute future realized-volatility targets.
9. Compute volatility ratios, slopes, drawdown, and VIX transformations.
10. Write the processed table and manifest, then validate schema and content.

Outputs are under:

```text
data/processed/spy_vix_volatility/
```

The current Yahoo-based SPY history is expected to differ slightly from the older Kaggle-derived snapshot. Schema, processing definitions, and overlapping-date behavior remain consistent.

### Monthly market-feature dataset

The monthly builder performs the following operations:

1. Read daily `^GSPC` prices beginning in January 1950.
2. Compute daily log returns.
3. Aggregate monthly realized volatility as

   ```text
   sqrt(sum(daily_log_return^2))
   ```

4. Preserve close and adjusted-close variants when available.
5. Compute monthly log realized volatility and 3-month and 12-month volatility summaries.
6. Convert FRED observations to month-end dates and merge them one-to-one.
7. Extract monthly observations from the Kenneth French ZIP archives.
8. Compute CPI inflation and industrial-production growth.
9. Apply a one-month information-availability lag to macro growth variables.
10. Compute the transparent BAA-minus-AAA default-spread candidate.
11. Create one-month-ahead and five-month-ahead volatility targets.
12. Produce a fixed paper-period slice from February 1950 through December 2017.
13. Produce an extended dataset containing only completed calendar months.
14. Engineer 24 leakage-aware predictor features covering volatility state, volatility dynamics, market factors, credit/rate conditions, and lagged macro dynamics.

Outputs are under:

```text
data/processed/monthly_market_features/
```

Key files are:

```text
extended_complete_market_months.parquet
paper_parity_1950_02_to_2017_12.parquet
features/preliminary_features.parquet
features/feature_catalog.csv
features/feature_manifest.json
manifest.json
```

The fixed paper-period dataset contains 815 monthly observations and 38 columns. The engineered feature table contains 815 observations, 24 predictor features, two future targets, and the date column.

## Rebuilding the data directory

Run all commands from the repository root in the project environment.

### 1. Download SPY and VIX

```bash
python scripts/data/download_market_data.py
```

This populates `data/raw/yahoo_daily_history/`, using the corresponding frozen fallback files only when Yahoo retrieval fails.

### 2. Download the external datasets

```bash
python scripts/data/download_external_datasets.py
```

This retrieves the long-history S&P 500 series, FRED macro and credit series, Kenneth French factor archives, Nikkei 225, Russell 2000, and FTSE 100.

### 3. Inventory the raw inputs

```bash
python scripts/data/inventory_raw_datasets.py
```

This verifies that the expected canonical raw files exist and records the result in:

```text
data/raw/raw_dataset_inventory.json
```

### 4. Build the SPY/VIX processed dataset

```bash
python scripts/data/build_volatility_dataset.py
```

### 5. Validate the SPY/VIX processed dataset

```bash
python scripts/data/validate_volatility_dataset.py
```

### 6. Build monthly market and feature datasets

```bash
python scripts/data/build_monthly_market_features.py
```

A successful monthly build should report:

```text
paper_rows: 815
extended_rows: 918
feature_count: 24
```

The extended row count reflects the configured retrieval cutoff and can change if the source cutoff is deliberately updated. The paper-period row count and feature count are fixed.

## Complete rebuild sequence

```bash
python scripts/data/download_market_data.py
python scripts/data/download_external_datasets.py
python scripts/data/inventory_raw_datasets.py
python scripts/data/build_volatility_dataset.py
python scripts/data/validate_volatility_dataset.py
python scripts/data/build_monthly_market_features.py
```

After rebuilding, run the test suite:

```bash
pytest -q
```

## Reproducibility notes

Remote financial and macroeconomic providers can revise historical observations. Yahoo can adjust price history, and FRED can revise recent macroeconomic values. Therefore:

- fixed historical processed ranges should be compared numerically with tolerances rather than by Parquet file hash;
- tests should use exact schema, date, row-count, and missingness checks;
- numerical comparisons should normally use approximately `rtol=1e-10` and `atol=1e-12`;
- the newest source months should be treated separately because publication lags and revisions can change their values;
- the frozen fallback files provide a stable recovery path when a remote source is unavailable.

The reconstructed paper-period monthly dataset was checked against the archived processed reference. Shapes, dates, columns, missingness, macroeconomic series, factor series, credit series, and feature catalog matched. Numerical differences were limited to floating-point noise, with the largest paper-dataset difference below `4e-14` and engineered-feature differences below `2e-12`.

## Updating the retrieval window

Date cutoffs are intentionally explicit in the downloader scripts. When extending the data horizon:

1. update the relevant end date in the retrieval script;
2. rerun retrieval and processing;
3. inspect source manifests and processed manifests;
4. verify that only expected tail rows changed;
5. update tests or documented expected extended-row counts when appropriate.

Do not silently replace fallback files during a routine refresh. A fallback update should be deliberate, reviewed, and committed as a provenance change.
