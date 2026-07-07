# LSTM baseline

## Role

LSTM is the learned recurrent Track A comparator. It tests whether a compact,
trainable sequence model can exploit the same temporal information used by the
ESN and QRC families.

The Phase 3 implementation is split into:

- `src/qpitome_qrc/baselines/lstm.py`: reusable sequence/model mechanics;
- `scripts/baselines/run_lstm.py`: canonical purged walk-forward runner;
- `tests/baselines/test_lstm.py`: sequence-alignment and determinism tests.

## Model

Default configuration:

- one LSTM layer;
- hidden size 32;
- dropout 0.1 on the final recurrent state;
- scalar linear readout;
- MSE loss;
- Adam optimizer;
- learning rate 1e-3;
- weight decay 1e-4;
- 40 epochs;
- batch size 32;
- seed 42;
- CPU single-thread execution for reproducibility.

The default Phase 3 lookback is 40 rows, matching the current reservoir window.

## Reset-branch code retained and rejected

Retained:

- compact one-layer architecture;
- deterministic cold restart per fold;
- train-only feature scaling;
- fixed training schedule;
- fold-level training diagnostics.

Rejected:

- monthly sequence semantics;
- reset-branch `compact7` / `paper_rcx_proxy` feature catalogs;
- 24-month lookback;
- 245-fold monthly protocol;
- duplicated scoring code;
- reset-branch output layout.

The Phase 3 runner uses `qpitome_qrc.evaluation.walkforward` and
`qpitome_qrc.evaluation.metrics`.

## Feature policy

For headline comparisons, pass an explicit feature list shared with the model
being compared:

```bash
python scripts/baselines/run_lstm.py \
  --data <canonical-table.parquet> \
  --target future_rv_20d \
  --feature-cols feature_a,feature_b,feature_c
```

Leaving `--feature-cols` unset automatically selects numeric columns other than
the date, target, and return column. That mode is deliberately labeled a
diagnostic convenience in the manifest and should not be used for a headline
comparison.

The runner refuses missing feature or target rows. Missing-data policy belongs
in the dataset-preparation layer.

## Fold behavior

For every purged walk-forward fold:

1. fit the feature scaler on the training block only;
2. build local 40-row training sequences;
3. cold-start and train one LSTM;
4. build validation and test sequences independently inside each split;
5. evaluate with the shared volatility metrics.

Independent split-local sequence construction drops the first `lookback - 1`
rows of validation and test. This matches the repository's common-date alignment
convention for rolling-window models.

## Outputs

`results/baselines/lstm/` contains:

- `predictions.csv`;
- `metrics_by_fold.csv`;
- `training_diagnostics.csv`;
- `run_manifest.json`.

## Current caveats

The architecture is intentionally fixed and small; it is a baseline, not a
neural architecture search. Before comparative claims, run multiple seeds and
report dispersion or demonstrate that conclusions are seed-stable.
