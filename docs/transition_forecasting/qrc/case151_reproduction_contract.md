# Case151 QRC reproduction contract

## Canonical scientific identity

This contract governs the first financial-QRC migration from `qrc-freeze-comparison-work`.
It is based on source commit `40ec805cc2b4efe416c0a57f1c599cca6def92c3`
and source run `palindrome_real_task_002`.

The canonical simulator is:

- folds 4–8, lead 5, no test rows;
- 40-step `level_instability` input with train-only robust scaling;
- six-atom staggered Rydberg ladder;
- interaction scale 1.25 and 0.02 μs per observation;
- A/4 → B/2 → A/4 palindrome;
- probes at steps 10, 20, and 40;
- `occupation_pair_raw` readout: six occupations plus fifteen pairs at each probe, 63 features total;
- `StandardScaler` plus multi-output `Ridge(fit_intercept=False)` on causal HAR residuals;
- alpha and correction lambda selected separately inside each fold by chronological inner validation;
- frozen alpha grid `[0.1, 1.0, 10.0, 100.0, 1000.0]`;
- frozen correction-lambda grid `[0.0, 0.25, 0.5, 1.0]`.

Case `P_GE151_^MERV_data_L5` is the fold-8 hardware-story example on the historical
fold lineage. Its selected readout is alpha 0.1 and lambda 0.25. It is an
intentionally selected development example, not a representative aggregate forecast
or a fold-invariant model parameter.

## Historical oracle and current-pipeline integration

`config/case151/expected_metrics.json` records exact historical pooled, transition,
interaction-off, Case151, Mincer–Zarnowitz, and Aquila-observable reference values.
Those values and the historical fold-8 selection are tied to canonical commit
`40ec805c...`, source run `palindrome_real_task_002`, and that run's fold composition.
They remain acceptance assertions for `historical-oracle` mode and are never tuning
targets.

The current aggregate financial pipeline regenerates its own fold tensors. The same
frozen Case151 model and chronological selection procedure are executed on those
tensors in `current-pipeline` mode. That mode verifies:

- the exact six-atom/palindrome/readout identity;
- `occupation_pair_raw` width 63;
- the unchanged alpha and lambda search grids;
- the observed fold-8 alpha and lambda belong to those grids;
- intercept-free residual head;
- zero test rows;
- all immutable files under `reference/case151/freeze_001/` against fixed SHA-256 values;
- output completeness.

It records current observed metrics, observed selected hyperparameters, and explicit
deltas versus the historical oracle. It does not require a different fold composition
to reproduce either the historical aggregate numbers or the historical selected grid
point, and it does not relabel current results as historical reproduction.

`historical-oracle` mode remains strict: it requires exact historical metrics and
fold-8 alpha 0.1 / lambda 0.25.

## Retry and preservation policy

A current-pipeline retry uses:

```bash
python scripts/reproduction/run_case151_simulation.py \
  --fold-dir results/runs/<RUN_ID>/files/data/processed/global_transition_dataset_1d/purged_walk_forward_folds \
  --output-root results/runs/<RUN_ID>/files/qrc/simulation/run \
  --run-id <RUN_ID> \
  --verification-mode current-pipeline \
  --archive-existing-failed
```

If an output directory exists without a verified audit, it is moved under
`files/qrc/simulation/run/failed_attempts/` before recomputation. A verified output is
never overwritten.

## Hardware boundary

The three recorded Aquila jobs already exist. Reproduction may inspect status and
retrieve their results only in a separate retrieval-only task; the judge-facing Agent
workflow must not create, query, select, retrieve, package, or submit hardware jobs.
`scripts/hardware/aquila_case151_support.py` contains no submission entry point.

Aquila evidence supports observable transfer under the accepted hardware-native
schedule. It is not an end-to-end hardware volatility forecast.

## Aggregate packaging interface

```text
results/runs/<RUN_ID>/files/qrc/
    simulation/run/<RUN_ID>/
    simulation/run/failed_attempts/
    hardware/<HARDWARE_RUN_ID>/
    comparisons/
```

`<RUN_ID>` identifies the local aggregate reproduction. Existing hardware jobs retain
their own `<HARDWARE_RUN_ID>` identities and must not be relabeled as newly generated.
The later six-mode compressed readout remains distinct from the canonical 63-feature
Case151 model.
