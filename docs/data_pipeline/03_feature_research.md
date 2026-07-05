# Feature research for reservoir-based volatility forecasting

## Design principle

Features are not collected because they are available. They are chosen to test whether a dynamical reservoir can extract nonlinear temporal information beyond a cheap linear sanity baseline.

The feature set needs three layers:

1. **state** - current volatility/market condition;
2. **dynamics** - how the state is changing and at what timescale;
3. **drivers** - exogenous financial and macroeconomic variables that may alter future dynamics.

## Primary paper-grounded features

### Volatility state

- monthly log realized volatility;
- quarterly RV average;
- annual RV average.

### Valuation state

- dividend-price ratio;
- earnings-price ratio.

### Equity factors

- market excess return;
- HML;
- SMB;
- short-term reversal.

The anchor paper's large RC-to-RCX improvement makes multivariate correlated inputs a central hypothesis rather than optional decoration.

### Macro/credit state

- three-month T-bill rate;
- inflation;
- default spread;
- industrial production growth.

## Additional candidate features to test, not assume

### Rates and slopes

- first differences of log RV;
- short/medium and medium/long slopes;
- acceleration of volatility change;
- rate of change of exogenous variables.

### Relative-scale features

- short/long volatility ratios;
- deviation from trailing robust median;
- standardized surprise relative to a rolling robust scale.

### Frequency-domain summaries

Potentially useful only after strict falsification:

- low/medium/high band power on non-overlapping or carefully defined series;
- spectral entropy;
- dominant-timescale proxies.

Previous work showed how overlapping rolling constructions can manufacture apparent frequency structure. Any frequency feature must be tested against a through-pipeline null before promotion.

### VOLARE-only realized measures

- realized variance;
- bipower variation;
- upside/downside semivariance;
- realized quarticity;
- realized kernel;
- jump proxies derived from RV versus bipower variation.

These can separate continuous volatility, jumps, asymmetry, and measurement uncertainty - information unavailable in close-to-close daily RV.

## Compression and encoding

Compression is a model-design question, not a preprocessing default.

Candidate strategies:

- no compression for the cheap baseline and ESN when dimension is manageable;
- PCA fitted on training data only as one explicit ablation;
- supervised feature selection confined to training/validation;
- physically motivated grouped channels for Rydberg encoding;
- time multiplexing and/or local/global controls when hardware allows.

The previous fixed PCA-6 bottleneck is not carried forward as a default.

## Promotion rules

A feature or feature family is promoted only if it:

1. has clear information timing and provenance;
2. survives missingness and leakage audit;
3. improves validation performance under the metric used for model selection;
4. is not redundant with a simpler feature family without measurable benefit;
5. preserves or improves performance under finite-shot/noise tests for QRC;
6. can be reproduced on the primary public dataset.
