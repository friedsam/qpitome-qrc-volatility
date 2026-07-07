# GARCH baseline

## Role

GARCH is the econometric Track A comparator. It is not a reservoir model and it
must not inherit the reset branch's monthly paper protocol.

The Phase 3 implementation is split into:

- `src/qpitome_qrc/baselines/garch.py`: reusable model mechanics;
- `scripts/baselines/run_garch.py`: canonical walk-forward experiment runner;
- `tests/baselines/test_garch.py`: unit tests for unit conversion and validation.

## Model

Default specification:

- GARCH(1,1);
- zero conditional mean;
- Student-t innovations;
- returns scaled by 100 during fitting for numerical stability;
- analytic multi-step conditional-variance forecast;
- 20-step aggregate volatility forecast by default.

The model module returns the full conditional-variance path. The runner converts
that path to the target convention

`future_rv = sqrt(sum(future_return**2))`

by undoing the return scaling, summing forecast variances, and taking the square
root.

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

The Phase 3 runner uses `qpitome_qrc.evaluation.walkforward` and
`qpitome_qrc.evaluation.metrics` instead.

## Run

Install optional dependencies:

```bash
python -m pip install -e ".[baselines,test]"
```

Run against an explicit canonical benchmark table:

```bash
python scripts/baselines/run_garch.py \
  --data <canonical-table.parquet> \
  --return-column log_return \
  --target future_rv_20d
```

The runner refuses missing return or target rows. Missing-data policy belongs in
the dataset-preparation layer, not inside the model.

## Outputs

`results/baselines/garch/` contains:

- `predictions.csv`;
- `metrics_by_fold.csv`;
- `run_manifest.json`.

## Current caveats

The runner currently refits at each test origin using the latest available
history. This is an explicit sequential rolling-origin baseline, not the same
training policy as a fixed readout reservoir. Comparative claims must therefore
state that policy clearly. A fixed-parameter filtered GARCH variant can be added
later if needed for a stricter training-policy match.
