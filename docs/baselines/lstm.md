# LSTM baseline

## Role

LSTM is the learned recurrent Track A comparator. It tests whether a compact,
trainable sequence model can exploit the same temporal information used by the
ESN and QRC families.

The Phase 3 implementation is split into:

- `src/qpitome_qrc/baselines/lstm.py`: reusable sequence/model mechanics;
- `scripts/baselines/lstm/run_phase3_lstm_walkforward.py`: Phase 3 runner;
- `tests/baselines/test_lstm.py`: sequence-alignment and determinism tests.

## Model

Default configuration:

- one LSTM layer;
- hidden size 32;
- dropout 0.1 on the final recurrent state;
- scalar linear readout;
- MSE loss on log realized volatility;
- Adam optimizer;
- learning rate 1e-3;
- weight decay 1e-4;
- 40 epochs;
- batch size 32;
- seed 42;
- CPU single-thread execution for reproducibility.

The default lookback is 40 rows, matching the current reservoir window.

## Phase 3 preprocessing

The standalone runner deliberately follows the established ESN preprocessing
path:

1. canonical `FEATURE_COLUMNS` from `qpitome_qrc.data.features`;
2. split-local explicit removal of non-finite modeling rows;
3. `StandardScaler` fit on training rows only;
4. PCA fit on scaled training rows only;
5. six PCA components by default;
6. split-local 40-row sequences;
7. train on `log(future_rv_20d)`;
8. exponentiate LSTM scores before Track A metrics.

This makes the first LSTM comparison interpretable against the current PCA6 ESN
path rather than importing the reset branch's unrelated monthly feature catalog.

## Reset-branch code retained and rejected

Retained:

- compact one-layer architecture;
- deterministic cold restart per fold;
- fixed training schedule;
- fold-level training diagnostics.

Rejected:

- monthly sequence semantics;
- reset-branch `compact7` / `paper_rcx_proxy` feature catalogs;
- 24-month lookback;
- 245-fold monthly protocol;
- duplicated scoring code;
- reset-branch output layout.

The runner uses shared data, walk-forward, sequence, and metric utilities.

## Run

Install optional dependencies:

```bash
python -m pip install -e ".[baselines,test]"
```

Smoke one fold before any full run:

```bash
python scripts/baselines/lstm/run_phase3_lstm_walkforward.py \
  --only-folds 1 \
  --epochs 2 \
  --tag smoke_fold1
```

The two-epoch command is a plumbing test only, not a baseline result.

Outputs stay in `scratch/lstm_walkforward/` until the implementation is
validated and intentionally promoted into the canonical comparison.

## Outputs

The runner writes tagged artifacts:

- `lstm_metrics_<tag>.csv`;
- `lstm_predictions_<tag>.csv`;
- `lstm_training_diagnostics_<tag>.csv`;
- `lstm_manifest_<tag>.json`.

## Current caveats

- The architecture is fixed and small; it is a baseline, not an architecture
  search.
- One seed is insufficient for a stability claim.
- PCA6 is a matched preprocessing choice, not evidence that six components are
  optimal for LSTM.
- The standalone runner is not yet wired into the canonical master comparison;
  promotion should happen only after unit tests and fold-level smoke runs pass.
