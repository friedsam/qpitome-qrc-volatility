# Phase 3 repository organization

## Principle

Organize by purpose, not model family. This keeps canonical evidence, searches, exploratory work, diagnostics, and hardware outputs distinct as the project grows.

## Scripts

### Active root entry points

Keep current operational runners at `scripts/` root until superseded by the canonical master runner:

- `run_phase3_ap_push.py`
- `run_phase3_ap_push_bitstring_shots.py`
- `run_phase3_esn_ridge_walkforward.py`
- `run_phase3_aquila_transfer_test.py`
- `run_phase3_aquila_transfer_test_explicit_key.py`
- `prepare_phase3_aquila_hardware_mvp.py`
- `qbraid_ahs_readiness_probe.py`

### `scripts/reference/`

Historical/reference runs retained for comparison:

- Phase 2 classical baselines
- Phase 2 persistence baselines
- Phase 2 feedback-TFIM reference
- Phase 2 prediction exports

### `scripts/exploratory/`

Model-development experiments that are not canonical benchmark entry points:

- RF-QRC tail probe
- RF-QRC time multiplexing
- structured level-rate RF-QRC
- one-QRC/two-head readout
- Rydberg market scalar probe
- Rydberg temporal probe
- Rydberg PCA2 probe
- early Rydberg temporal reservoir / robustness variants

### `scripts/diagnostics/`

Mechanism and ablation studies:

- dual-chain diagnostic
- temporal-memory capacity diagnostic
- qubit-scaling diagnostic
- TFIM anchor-order diagnostic
- TFIM level/rate-order diagnostic
- TFIM N/H scaling
- Rydberg-TFIM response diagnostics

## Results

### `results/canonical/`

Standardized comparison outputs only. Future master runner target.

Expected subareas:

- `metrics/`
- `predictions/`
- `summaries/`
- `fold_characterization/`
- `costs/`

### `results/reference/`

Historical benchmark evidence used for comparison, including retained Phase 2 tables.

### `results/tuning/`

Parameter-search trials, selected configurations, and search summaries. No canonical promotion without a separate comparison run.

### `results/experimental/`

Exploratory forecasting analyses and notebook-generated outputs.

### `results/diagnostics/`

Mechanism, geometry, memory, scaling, and ablation outputs.

### `results/hardware/`

Finite-shot studies, readiness probes, simulator-to-hardware transfer checks, and actual QPU results.

### `results/archive/`

Superseded files retained for provenance.

## Migration rule

Do not mass-delete historical files. Move only when classification is unambiguous. When an old script is still actively referenced in docs or commands, either update those references in the same change or leave the entry point in place until the replacement runner exists.

## Future output rule

New scripts must write to a purpose-specific result area:

- canonical comparison -> `results/canonical/`
- tuning/search -> `results/tuning/<search_name>/`
- exploratory notebook/script -> `results/experimental/<experiment_name>/`
- diagnostics -> `results/diagnostics/<diagnostic_name>/`
- finite-shot/hardware -> `results/hardware/<study_name>/`

`scratch/` remains disposable and git-ignored.
