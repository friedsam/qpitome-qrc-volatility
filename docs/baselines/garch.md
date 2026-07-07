# GARCH baseline

## Role

GARCH is the econometric Track A comparator. It is not a reservoir model and it
does not inherit the reset branch's monthly paper protocol.

The Phase 3 implementation is split into:

- `src/qpitome_qrc/baselines/garch.py`: reusable model mechanics;
- `scripts/baselines/garch/run_phase3_garch_walkforward.py`: Phase 3 runner;
- `tests/baselines/test_garch.py`: unit tests for validation and unit conversion.

## Model

Default specification:

- GARCH(1,1);
- zero conditional mean;
- Student-t innovations;
- returns scaled by 100 during fitting for numerical stability;
- analytic multi-step conditional-variance forecast;
- 20-step aggregate volatility forecast.

The model module returns the full conditional-variance path. The runner converts
that path to the target convention

`future_rv = sqrt(sum(future_return**2))`

by undoing the return scaling, summing forecast variances, and taking the square
root.

## Phase 3 protocol

Default inputs match the canonical branch:

- data: `data/processed/phase2_spy_vix_volatility.csv`;
- return column: `spy_log_return`;
- target: `future_rv_20d`;
- fold geometry: 5 folds, minimum train 2500, validation 504, purge 60.

For each test origin, GARCH is refit using return history available through that
origin and then produces a 20-step variance path. This is a sequential
rolling-origin econometric baseline. It is deliberately disclosed as a different
training policy from fixed-readout reservoir models.

## Reset-branch code retained and rejected

Retained:

- canonical GARCH(1,1)-t specification;
- convergence checks;
- analytic multi-step variance path;
- explicit failure propagation rather than silent substitution.

Rejected:

- monthly target and feature table;
- NYSE forecast-month calendar aggregation;
- 245-fold monthly protocol;
- reset-branch output layout;
- duplicated scoring code.

The runner uses `qpitome_qrc.evaluation.walkforward` and
`qpitome_qrc.evaluation.metrics`.

## Run

Install optional dependencies:

```bash
python -m pip install -e ".[baselines,test]"
```

Smoke one fold before any full run:

```bash
python scripts/baselines/garch/run_phase3_garch_walkforward.py \
  --only-folds 1 \
  --tag smoke_fold1
```

Outputs stay in `scratch/garch_walkforward/` until the implementation is
validated and intentionally promoted into the canonical comparison.

## Outputs

The runner writes tagged artifacts:

- `garch_metrics_<tag>.csv`;
- `garch_predictions_<tag>.csv`;
- `garch_manifest_<tag>.json`.

## Current caveats

- Sequential refitting is not the same training policy as a fixed reservoir
  readout. Comparative claims must state this explicitly.
- No GJR/EGARCH leverage extension or hyperparameter search is performed.
- The 20-step aggregation assumes the canonical target is
  `sqrt(sum(future daily return**2))`; this mapping must be checked against the
  dataset-construction code during validation, not inferred from the column name
  alone.
