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
- alpha and correction lambda selected separately inside each fold by chronological inner validation.

Case `P_GE151_^MERV_data_L5` is the fold-8 hardware-story example. Its selected
readout is alpha 0.1 and lambda 0.25. It is an intentionally selected development
example, not a representative aggregate forecast.

## Fixed oracle

`config/case151/expected_metrics.json` records the exact pooled, transition,
interaction-off, Case151, Mincer–Zarnowitz, and Aquila-observable reference values.
The values are acceptance assertions, not tuning targets.

## Hardware boundary

The three recorded Aquila jobs already exist. Reproduction may inspect status and
retrieve their results, but the judge-facing workflow must not create or submit a
new hardware task. `scripts/hardware/aquila_case151_support.py` contains only
retrieval and local simulator helpers; it has no submission entry point.

Aquila evidence supports observable transfer under the accepted hardware-native
schedule. It is not an end-to-end hardware volatility forecast.

## Submission packaging interface

The submission maintainer may wire this stage into the aggregate runner using:

```text
results/runs/<RUN_ID>/files/qrc/
    simulation/run/<RUN_ID>/
    hardware/<HARDWARE_RUN_ID>/
    comparisons/
```

`<RUN_ID>` identifies the local aggregate reproduction. Existing hardware jobs retain
their own `<HARDWARE_RUN_ID>` identities and must not be relabeled as newly generated.

## Deliberate exclusions

This selective port does not modify `scripts/runs/run_submission.py`, the qBraid Skill,
or AGENTS/maintenance infrastructure. It does not substitute the later six-mode
compressed readout for the 63-feature Case151 model.
