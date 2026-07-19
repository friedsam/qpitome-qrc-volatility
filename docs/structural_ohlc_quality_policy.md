# Structural OHLC bad-print quality policy

## Purpose

The daily transition target is log Parkinson volatility,

`log(abs(log(high / low)) / sqrt(4 log 2))`.

This target is highly sensitive to isolated malformed highs or lows. The original range audit checked only mathematical validity (`high >= low`, positive prices, valid dates). It did not detect economically implausible single-day wicks where open and close remain close but one reported extreme is far away.

During Stage E diagnostics, two such `^XAX_data` lows on 2008-06-02 and 2008-06-03 created target values near -1.29 and -1.25 while neighboring observations were near -4.4 to -5.5. These two records dominated fold-6 QLIKE for HAR and reservoir models. The quality rule below was frozen before recomputing rankings.

## Frozen rule

A row is flagged only when all conditions hold:

1. `log(high / low) > 0.20`.
2. The range exceeds the preceding 252-row median by more than `12 * 1.4826 * MAD`, using at least 60 prior observations.
3. `abs(log(close / open)) < log(1.05)`.
4. Either the high exceeds both open and close by more than `log(1.20)`, or the low falls below both open and close by more than `log(1.20)`.

The rule is causal: its robust historical threshold uses only preceding rows. It does not inspect model predictions, validation losses, or future prices.

With the current raw-data lineage, the rule must produce exactly 12 flagged rows across two indices. The audit script fails loudly if the count changes, forcing inspection of source-data drift rather than silently changing the analytical sample.

## Sample policy

Raw OHLC files are never edited or interpolated. Both lineages are retained:

- `raw`: all mathematically valid observations;
- `quality_gated`: exclude a sample when its input interval or target interval intersects a flagged date.

Input interval: `input_start_date <= flagged_date <= origin_date`.

Target interval: `origin_date < flagged_date <= target_end_date`.

The current Stage E manifest contains 19 unique contaminated sample IDs. Raw and quality-gated results must be reported side by side.

## Reproduction

From the repository root, with the project environment active:

```bash
python scripts/transition_forecasting/quality/run_structural_bad_print_audit.py \
  --raw-root data/raw/transition_forecasting/global_stock_indices_historical_data/individual_indices_data \
  --manifest results/transition_forecasting/modeling/build_stage_e_chronological_dataset/chronology_rematch_8fold_002/rematched_rolling_manifest.csv \
  --expected-flagged 12
```

The command creates an immutable UTC-timestamped directory under:

`results/transition_forecasting/quality/structural_bad_prints/`

Outputs:

- `daily_ohlc_quality_audit.csv.gz`: daily diagnostics and flags;
- `frozen_structural_bad_prints.csv`: the exact flagged rows;
- `annotated_sample_manifest.csv.gz`: original manifest plus contamination flags;
- `contaminated_samples.csv`: all contaminated manifest rows;
- `clean_sample_ids.csv`: IDs eligible for the quality-gated analysis;
- `summary.json`: counts, lineage, policy statement, and test-row usage;
- `params.json`: exact numerical policy parameters.

## Required downstream behavior

Model scripts should accept `quality_policy=raw|exclude_structural_bad_prints`. For the gated setting, join predictions to `clean_sample_ids.csv` by `sample_id`, `fold`, and `fold_split` before calculating metrics. Model fitting must also exclude contaminated training samples for a complete rebuilt lineage. Existing frozen predictions may be filtered only for a sensitivity analysis; they are not a substitute for retraining when contaminated rows occur in training folds.

## Verification checklist

A reproducible run should verify:

- 12 daily rows flagged;
- 2 indices affected;
- 19 unique contaminated Stage E sample IDs;
- June 2 and June 3, 2008 in `^XAX_data` are flagged;
- broad VIX crisis observations are not removed merely because their ranges are large;
- no test labels or model outputs enter the policy;
- raw files remain unchanged;
- raw and gated metrics are both retained.

## Interpretation

The gate removes structurally inconsistent source observations, not difficult market regimes. A flagged record has a large isolated wick relative to both its candle body and its own preceding range distribution. Large genuine moves with correspondingly large open-to-close changes are retained.
