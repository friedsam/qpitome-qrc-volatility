# ESN Baseline v0

## Purpose

First Echo State Network baseline for the market-stress v0 dataset.

This is a transparent scaffold implementation, not the final production-grade ESN benchmark. It verifies that the sequence/reservoir baseline runs end-to-end before switching to a library-backed or cleaned benchmark implementation.

## Dataset

Input file:

```text
data/processed/market_stress_v0.csv
```

Sequence construction:

```text
X[t] = previous 20 trading days of engineered SPY/VIX features
y[t] = future_high_vol_label at date t
```

Chronological split:

```text
train: before 2016-01-01
validation: 2016-01-01 to 2019-12-31
test: 2020-01-01 onward
```

## Model

Echo State Network:

- fixed random recurrent reservoir
- reservoir size: 300
- spectral radius: 0.9
- input scale: 0.5
- leak rate: 0.5
- trained readout: logistic regression with class balancing

## Results

Validation:

```text
balanced_accuracy: 0.680
roc_auc: 0.849
pr_auc: 0.458
stress-class precision: 0.512
stress-class recall: 0.407
stress-class F1: 0.454
```

Test:

```text
balanced_accuracy: 0.725
roc_auc: 0.796
pr_auc: 0.588
stress-class precision: 0.496
stress-class recall: 0.678
stress-class F1: 0.573
```

## Interpretation

The first ESN pass detects a substantial fraction of high-volatility test periods but produces many false alarms.

This is usable as an initial matched-reservoir baseline scaffold, but not yet final.

## Known Limitations

- Current label threshold was created during dataset preparation using the full dataset.
- Final modeling code should recompute the high-volatility threshold using the training split only.
- Decision threshold is fixed at 0.5 rather than tuned on validation.
- ESN hyperparameters are arbitrary and not tuned.
- Implementation is custom and intended for transparency, not final benchmarking.

## Next Steps

1. Recompute label threshold from training split only.
2. Tune decision threshold on validation.
3. Replace or validate custom ESN against a library implementation.
4. Use the finalized ESN as the matched classical reservoir baseline for QRC.