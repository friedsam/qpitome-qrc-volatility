# Transition-Forecasting Data Pipeline: Design and Rationale

## 1. Purpose of the pipeline

The data pipeline converts raw daily stock-index price histories into a leakage-controlled forecasting dataset for volatility-regime transitions.

The prediction problem is deliberately causal. Each model input ends at a forecast origin. The label indicates whether that origin precedes a previously detected transition event by a specified lead time. Future observations are retained only as targets and are never used to construct the input features presented to the model.

The pipeline was designed around four requirements:

1. preserve the historical data lineage used in the earlier project stages;
2. remove demonstrably invalid observations without smoothing or inventing market data;
3. define transition events consistently across many markets;
4. prevent chronological and sample-selection leakage during model development.

The completed pipeline produces a one-channel volatility representation, a deterministic three-channel representation, a full candidate-control pool, and three purged walk-forward development folds while reserving the newest portion of the data as an untouched test set.

---

## 2. Market-data source and frozen source inventory

The raw source is the Kaggle dataset `guillemservera/global-stock-indices-historical-data`, distributed under CC-BY-NC-4.0. The dataset contains daily OHLCV histories assembled by the dataset author from Yahoo Finance. It includes a consolidated panel and individual CSV files for more than 30 global stock indices.

The acquisition stage supports three modes:

- use an existing validated local snapshot;
- download the current Kaggle archive through the Kaggle CLI;
- use a locally retained fallback snapshot when live retrieval is unavailable.

Before installation, the acquisition code verifies that:

- the combined panel exists;
- the individual-index directory exists;
- the required date, open, high, low, and close columns are present;
- at least 30 individual index files are available;
- every retained file can be inventoried.

For a fallback copy, every file is checked against a stored SHA-256 manifest before use. For any accepted source snapshot, the pipeline writes a raw acquisition manifest containing the source identity, license, retrieval mode, timestamps, file count, sizes, and SHA-256 hashes.

A second frozen inventory records which files were eligible for the historical GPT-2 pipeline and the expected hash of every source file. During processed-dataset construction, the current raw files are checked against those frozen hashes. A mismatch causes the build to stop rather than silently creating a dataset from a changed source snapshot.

This distinction is important: the acquisition manifest documents what was obtained, while the frozen modeling inventory guarantees that the downstream analysis uses the intended historical source version.

---

## 3. Parsing and normalization of individual index files

Each individual index CSV is parsed separately. Column names are stripped and converted to lowercase. Dates are parsed to a timezone-free daily timestamp, and commas are removed from numeric fields before converting open, high, low, and close values to numbers.

Rows are rejected when any of the following conditions holds:

- the date is invalid;
- an OHLC value is missing or nonnumeric;
- an OHLC value is zero or negative;
- the reported high is below the reported low.

The surviving observations are sorted by date. When more than one row has the same date, the final source row is retained and the earlier duplicate is recorded as removed.

Every rejected row is written to a correction ledger with its original source row, reason, and action. The pipeline therefore does not merely output a cleaned table; it preserves an auditable account of every basic removal.

No interpolation, forward filling, winsorization, arbitrary clipping, or synthetic dates are used. Missing or invalid market observations are removed rather than replaced with invented values.

---

## 4. Structural bad-print detection and removal

Basic validity checks do not detect every erroneous price observation. A single extreme intraday high or low can be numerically valid while still being an obvious data-provider bad print. Such a row is especially damaging here because the volatility measure depends directly on the high-low range. One bad wick can create an artificial volatility spike, a false transition event, and many contaminated samples.

The structural bad-print policy therefore looks for a specific pattern: an abnormally large intraday range caused by a long wick while the open-to-close body remains comparatively small.

For each valid row, the pipeline calculates:

- log intraday range: `log(high / low)`;
- absolute log body move: `abs(log(close / open))`;
- upper-wick excursion above the larger of open and close;
- lower-wick excursion below the smaller of open and close.

The range is compared only with prior observations. The baseline is a trailing 252-session median and median absolute deviation, with at least 60 prior sessions required. A row is flagged only when all of the following are true:

1. its log range exceeds 0.20;
2. its log range exceeds the prior rolling median by more than 12 robust standard deviations, using the 1.4826 MAD consistency factor;
3. its open-to-close body move is below `log(1.05)`;
4. at least one wick excursion exceeds `log(1.20)`.

The rolling statistics are shifted by one session, so the row being evaluated cannot influence its own threshold. This makes the rule causal and prevents an extreme observation from diluting the benchmark used to judge itself.

The frozen policy identifies exactly 12 rows across two indices. The production pipeline removes those rows before calculating volatility. Their dates, index names, diagnostic quantities, and removal actions are retained in `row_corrections.csv`.

A separate parity-control mode can retain the flagged rows while recording them. That mode exists only to reproduce and compare the historical GPT-2 artifacts; it is not the corrected production dataset.

---

## 5. Excluding unusable early history

Some source files contain long early periods in which the reported daily high equals the reported daily low. For a range-based volatility estimator this gives a zero range. Taking the logarithm of zero volatility is undefined, so simply retaining those rows would either create non-finite values or silently discard large, irregular sections of history after transformation.

The project therefore performed a separate range-quality audit for each index. The audit records:

- nominal first and last dates;
- first date with a nonzero high-low range;
- annual range coverage;
- total zero-range and nonzero-range counts;
- fraction of observations with nonzero range;
- longest consecutive zero-range run;
- the first qualifying year and recommended effective start.

The recommended effective start is frozen in the range-quality contract and applied consistently in transition detection, daily-volatility construction, and sample generation.

This matters for several major indices. For example:

- the S&P 500 file begins in 1927, but its usable range history begins in 1962;
- the Nasdaq Composite file begins in 1971, but its effective start is 1985;
- the Nikkei 225 file begins in 1965, but its effective start is 1989;
- the NYSE Composite file begins in 1965, but its effective start is 2003;
- the Australian S&P/ASX 200 file begins in 1992, but its effective start is 2001.

The pipeline does not delete these old rows from the raw archive. It excludes them from range-based modeling by applying the frozen effective start for each index. This preserves the source while preventing artificial zero-volatility histories from influencing thresholds, transition counts, matching features, or model inputs.

---

## 6. Daily volatility construction

For each eligible index and each date at or after its effective start, the pipeline calculates Parkinson range volatility:

`abs(log(high / low)) / sqrt(4 log 2)`

It then takes the natural logarithm of that volatility. Exact zero ranges are converted to missing values before the logarithm and omitted.

The final daily series is therefore `log_parkinson_volatility`, not raw price and not close-to-close return volatility.

Parkinson volatility was chosen because it uses the full intraday high-low range and is more informative about daily dispersion than a close-to-close return alone. The logarithm makes multiplicative changes in volatility additive and reduces the extreme scale imbalance between calm and turbulent periods.

The resulting series are checked for finite values, sorted by index and date, and rejected if duplicate index-date rows remain.

The pipeline also consolidates the cleaned OHLC files into a single ordered table for auditability. The model tensors themselves are built from the log Parkinson volatility series.

---

## 7. Transition-candidate detection

A transition is defined as the onset of a persistent high-volatility regime, not merely a single volatile day.

For each index, the turbulent threshold is estimated from that index's pre-2016 history as the 80th percentile of log Parkinson volatility. At least 250 pre-2016 sessions are required after applying the effective-start filter. Indices without sufficient training history are skipped.

A day is considered an onset candidate when:

- its volatility is at or above the index-specific 80th-percentile threshold;
- at least 10 of the next 15 sessions are also above that threshold;
- no more than 2 of the preceding 10 sessions were above the threshold;
- it is at least 60 sessions after the previously accepted onset for that index.

These rules distinguish a regime transition from an isolated spike. The forward persistence criterion confirms that the market actually entered a sustained turbulent state, while the prior-window condition requires the onset to emerge from a relatively calm state. The 60-session separation prevents one extended crisis from being counted repeatedly as many separate onsets.

The detection step uses future sessions to define the historical event label, but those future sessions are not included in a model input. Forecast origins are later placed before the onset, and all model features are calculated using observations available at the origin.

For each detected event, the catalogue records event anatomy such as the pre-onset level, onset level, jump in log volatility, 60-session peak, persistence, and recovery time. These fields describe the event and support audits; they are not used as pre-origin model inputs.

---

## 8. Clustering nearby detections into global market episodes

The same global shock can trigger transition detections on slightly different trading days across countries because of time zones, holidays, and market-specific response timing. Treating each date as an independent event would split one crisis into several pseudo-independent observations.

The pipeline therefore clusters unique onset dates into global episodes. Dates are sorted chronologically. A new episode begins when an onset is more than seven calendar days after the first date of the current cluster; otherwise it joins that cluster.

The fixed-start window is important. The cluster does not expand indefinitely through a chain of adjacent dates. Every date must remain within seven days of the cluster's anchor date. This prevents a long sequence of loosely connected events from collapsing into one oversized episode.

Every clustered event receives a global episode identifier. Later chronological splitting treats all positive samples from the same episode as one chronology group, so a single global shock cannot appear in both training and validation data.

---

## 9. Choosing representative market events

Several indices can represent the same underlying national market. For example, the United States includes the S&P 500, NYSE Composite, Dow Jones Industrial Average, Nasdaq Composite, Russell 2000, and NYSE American Composite. Australia and mainland China also have multiple index files.

Keeping every highly correlated domestic index event would overweight those countries and create near-duplicate samples. The pipeline maps indices to market groups and keeps at most one representative index event per market group within each global episode.

For multi-index markets, a fixed priority list determines the representative. The United States prioritizes the S&P 500, followed by the NYSE Composite, Dow, Nasdaq, Russell 2000, and NYSE American Composite. Australia prioritizes the All Ordinaries over the ASX 200, and mainland China prioritizes the Shanghai Composite over the Shenzhen Component. For a market represented by a single index, that index is retained automatically.

The consolidated all-index panel is excluded because it duplicates the individual files. VIX is excluded because it is a volatility index rather than an equity-market price index.

The result is a representative event catalogue that preserves cross-market breadth without counting multiple proxies for the same market shock as independent evidence.

---

## 10. Constructing positive forecasting samples

Each representative transition event generates up to three positive samples, corresponding to forecast leads of 1, 5, and 10 trading sessions.

For a transition onset at position `t`, the forecast origin is:

- `t - 1` for lead 1;
- `t - 5` for lead 5;
- `t - 10` for lead 10.

Every input contains the 40 consecutive volatility observations ending at the origin. The target contains the following 10 sessions after the origin. Exact dates are stored for:

- the beginning of the input window;
- the forecast origin;
- the end of the target window.

A sample is omitted when there are not 40 complete pre-origin observations or 10 complete post-origin target observations.

At the origin, the pipeline also computes seven causal matching features from present and prior volatility only:

- current level;
- five-session mean;
- twenty-session mean;
- five-session slope;
- twenty-session slope;
- twenty-session standard deviation;
- twenty-session maximum.

These features are used to select comparable controls. They do not include the future target or event-anatomy variables.

The positive label therefore means: given the 40-session history available at this origin, a persistent volatility transition begins after the specified lead.

---

## 11. Initial control construction

A useful control should resemble a positive sample before the forecast origin but should not be located near another known transition.

For each index, lead, and original chronological split, the initial Stage D builder considers every origin with a complete 40-session input and 10-session target. An origin is excluded when it is within 60 sessions of any detected transition on that index.

Candidates are compared with positives using the seven causal matching features. Features are standardized within the relevant index-lead-split candidate pool, and Euclidean distance is calculated in that standardized feature space. The three nearest unused candidates are selected for each positive.

A control origin cannot be reused within the matching stratum. Each positive must receive exactly three controls, or validation fails.

This initial dataset establishes the canonical sample inventory and allows broad quality checks. However, these controls cannot simply be carried unchanged into every rolling development fold. A control chosen using a broad historical split could be influenced by candidate availability outside the narrower train or validation partition used in a particular fold. That would make the model-development data subtly dependent on future chronology.

For that reason, the final evaluation pipeline rebuilds the candidate pool and rematches controls separately within every fold and partition.

---

## 12. One-channel and three-channel model representations

### One-channel representation

The canonical one-channel tensor contains the 40-session sequence of log Parkinson volatility levels. Its shape is:

`(samples, 40, 1)`

This representation preserves the original corrected Stage D signal and provides the direct parity link to the historical pipeline.

### Three-channel representation

The three-channel tensor is a deterministic transform of the one-channel sequence:

1. log-volatility level;
2. first difference of log volatility, with the first value set to zero;
3. normalized sequence time from zero to one.

Its shape is:

`(samples, 40, 3)`

The second channel exposes local volatility movement explicitly rather than requiring every model to reconstruct it from adjacent levels. The third channel gives classical sequence models an explicit within-window position coordinate. For the physical Rydberg reservoir, time is already represented by sequential evolution, so the planned physical input uses level and difference while treating time implicitly.

The transformation does not change sample identity, order, labels, folds, or matching. The one-channel and three-channel datasets therefore support a controlled representation comparison rather than two independently constructed datasets.

---

## 13. Chronological train, validation, and fixed-test design

The final evaluation does not use the earlier fixed 2013/2016 train-validation-test boundaries as its main development protocol. Instead, it constructs three expanding walk-forward folds and reserves the newest 17% of chronology groups as a fixed test set.

Chronology groups are defined differently for positives and controls:

- all positive samples from one global episode form one group;
- each control is grouped by its own actual origin.

This keeps every forecast lead from the same positive episode together. It also prevents an old episode from being split across train and validation merely because its participating markets have slightly different onset dates.

The remaining pre-test chronology is divided into four ordered blocks. The three development folds are:

- fold 1: block 1 for training, block 2 for validation;
- fold 2: blocks 1-2 for training, block 3 for validation;
- fold 3: blocks 1-3 for training, block 4 for validation.

The newest 17% remains assigned to test in every fold. It is materialized for reproducibility but is not evaluated during development.

This design gives three forward-looking validation periods while allowing the training history to expand in the same direction it would in real deployment.

---

## 14. Interval purging and the embargo

A sample occupies more time than its single origin date. Its input begins 39 sessions earlier, and its target extends 10 sessions later. Two samples with different origins can therefore share input observations, target observations, or both.

The fold builder uses the exact stored `input_start_date` and `target_end_date` to define each sample's full interval. It does not approximate the interval from calendar-day offsets when exact trading-session dates are available.

Two leakage controls are then applied.

### Boundary embargo

Around each train-validation boundary, a 10-calendar-day embargo is applied. Any train or validation sample whose full interval intersects the boundary plus or minus the embargo is removed from active use.

This creates a buffer against near-boundary dependence and reduces the chance that adjacent market observations influence both sides of the split.

### Same-index interval-overlap purge

Within each index, the pipeline compares complete sample intervals across train, validation, and test partitions. When intervals from different partitions overlap, the chronologically later sample is purged.

The purge is index-specific because observations from the same index are direct duplicates or transformations of the same underlying time series. Cross-market co-movement is part of the forecasting problem and is not treated as literal row overlap.

After purging, the maximum training interval end must be earlier than the minimum validation interval start. A fold that is not strictly chronological is rejected.

Together, exact-interval purging and the embargo protect against leakage that an origin-date-only split would miss.

---

## 15. Fold-local control rematching and origin reuse

After fold assignment and purging, controls are selected again from a full candidate pool. Matching occurs separately within each:

- fold;
- fold split (`train`, `val`, or reserved `test`);
- index;
- forecast lead.

The candidate features are standardized within the corresponding partition and lead. Positives are processed in deterministic order, and candidates are ranked by standardized Euclidean distance. Ties are resolved deterministically by origin position and date.

The lead is a candidate filter because a control must have the same forecast configuration as its matched positive. However, lead does not reset the origin-uniqueness rule. Within a fold, split, and index, a market origin can be used only once across all positives and across leads 1, 5, and 10.

This prevents the same 40-session sequence from appearing multiple times under different lead labels or being reused as the control for several positive events.

The rematching audit records, for every positive:

- total available candidates;
- available unused candidates at selection time;
- number of controls selected;
- whether the required three-control match was complete;
- maximum selected distance.

Any reused origin or incomplete positive-control set causes validation to fail.

Fold-local rematching is one of the central leakage protections in the pipeline. The positive event catalogue is fixed, but the negative examples are selected only from information legally available inside the partition in which they will be used.

---

## 16. Validation and frozen outputs

The corrected canonical dataset contains:

- 286,015 cleaned OHLC rows;
- 232,630 daily volatility rows;
- 383 representative transition events;
- 1,149 positive samples;
- 3,447 controls;
- 4,596 canonical samples across 35 indices and 198 global episodes.

The final purged walk-forward artifacts contain 10,272 fold rows. The one-channel tensor has shape `(10272, 40, 1)`, and the three-channel tensor has shape `(10272, 40, 3)`.

Validation checks include:

- required files and schemas;
- source hash agreement with the frozen inventory;
- expected structural-bad-print count and affected-index count;
- finite volatility and tensor values;
- unique index-date observations;
- exact three-to-one control ratio;
- complete 40-session inputs and 10-session targets;
- episode integrity across partitions;
- strict post-purge chronology;
- absence of same-index cross-partition interval overlap;
- absence of control-origin reuse;
- identical sample and fold assignments in the one- and three-channel versions;
- deterministic three-channel transformation;
- pickle-free loading of all NPZ artifacts.

The organized one-channel fold pipeline was also compared against an independent rerun of the historical corrected fold implementation. The selected controls, sample order, sample identifiers, fold assignments, and tensors were identical. The only differences were approximately `1e-15` variations in stored floating-point matching distances, which did not change any ranking or selection.

A SHA-256 freeze report was generated for 20 core dataset, candidate-pool, and fold artifacts. This report serves as the reference fingerprint for the selective port to `stage1-dev` and for clean-environment reproduction on a second Mac.

---

## 17. What remained untouched during development

The newest fixed test partition was not used to choose representations, models, hyperparameters, or reservoir settings. The test assignments were generated so that the complete pipeline can be reproduced, but test metrics remained unevaluated.

The pipeline also preserves the following boundaries:

- the raw archive is retained unchanged;
- excluded early history remains in the raw files and is filtered only for range-based modeling;
- removed observations are documented rather than overwritten silently;
- no missing price values are synthesized;
- future targets and event-anatomy fields are excluded from model inputs and matching features;
- historical exploratory outputs are not treated as final submission dependencies.

The final test set should be evaluated only after the classical and quantum model-development decisions are frozen.

---

## 18. End-to-end flow

In operational order, the completed pipeline is:

1. acquire or verify the global-index source snapshot;
2. validate schemas and freeze file hashes;
3. parse and normalize every individual index file;
4. remove invalid rows and duplicate dates while recording a ledger;
5. detect and remove 12 causal structural bad prints;
6. apply frozen per-index effective starts from the zero-range quality audit;
7. calculate finite daily log Parkinson volatility;
8. estimate pre-2016 index-specific turbulent thresholds;
9. detect persistent transition onsets;
10. cluster nearby onset dates into anchored seven-day global episodes;
11. choose one representative index per market and episode;
12. create 1-, 5-, and 10-session-ahead positive origins with 40-session inputs and 10-session targets;
13. construct the initial matched-control dataset for canonical lineage and audit;
14. create aligned one-channel and deterministic three-channel tensors;
15. reserve the newest 17% of chronology groups as fixed test;
16. construct three expanding train-next-validation folds;
17. purge embargoed and overlapping full sample intervals;
18. rebuild candidate pools and rematch three controls per positive inside each fold and partition;
19. validate chronology, matching, serialization, representation alignment, and historical parity;
20. freeze the final artifacts with SHA-256 checksums.

This sequence converts a heterogeneous historical market archive into a causal, auditable, and reproducible dataset suitable for comparing classical forecasting models with the proposed quantum reservoir architecture.
