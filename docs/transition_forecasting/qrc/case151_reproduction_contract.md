# Case151 QRC reproduction contract

## Canonical identity

- Source commit: `40ec805cc2b4efe416c0a57f1c599cca6def92c3`
- Source run: `palindrome_real_task_002`
- Selected model: `palindrome_ordered_on`
- Representation: `level_instability`
- Six-atom staggered ladder; `0.02 us` per observation
- Palindrome: `A/4 -> B/2 -> A/4`
- Probe steps: `10`, `20`, `40`
- Readout bank: `occupation_pair_raw`, 63 features
- Fold-local chronological selection over alpha `(0.1, 1, 10, 100, 1000)` and correction lambda `(0, 0.25, 0.5, 1)`
- No readout intercept; no reserved test rows
- Case `P_GE151_^MERV_data_L5` is fold 8 with alpha `0.1`, lambda `0.25`

The interaction-off matched reservoir performs better. This package does not claim
interaction-specific or generic quantum advantage.

## Submission-run placement

The standalone producer accepts explicit paths. The judge-facing orchestrator should
place one aggregate execution under:

```text
results/runs/<RUN_ID>/files/qrc/
    simulation/run/<RUN_ID>/
    hardware/<HARDWARE_RUN_ID>/
    comparisons/
```

The preserved Aquila jobs retain their external hardware IDs. They must not be
renamed to the aggregate run ID.

## Hardware boundary

The submitted workflow may inspect and collect the three existing Aquila jobs. Fresh
hardware submission is intentionally excluded from the submission repository.
