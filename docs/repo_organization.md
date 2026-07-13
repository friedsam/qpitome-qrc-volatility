# Phase 3 repository organization

## Principle

Organize by scientific purpose and experimental role. A broad area may keep its own top-level directory, but no directory should become an undifferentiated stream of scripts.

The `scripts/modeling/` directory is the modeling workbench. It remains appropriate for modeling scripts, but it must be subdivided so a reader can understand where a script belongs, whether it is active, and what evidence it produces.

## Modeling scripts

### `scripts/modeling/regimes/`

Weekly regime construction, HMM fitting, predictive-density evaluation, and state interpretation.

Typical contents:

- weekly HMM baseline runners;
- filtered-state analysis;
- predictive-score diagnostics;
- regime-state stability checks.

### `scripts/modeling/branching_day5/`

Historical day-5 recovery-versus-relapse work, including the retired barrier-defined target and the falsification tests that exposed its construction artifact.

Typical contents:

- D0/D1/D2 classical baselines;
- barrier and first-passage controls;
- static and temporal day-5 reservoir probes;
- residual, protected-offset, spatial, nonlinear, PCA, and higher-order tests;
- exact D1 rank and first-passage-null audits.

This folder is retained for provenance. New primary work should not extend the retired barrier target.

### `scripts/modeling/track_a_onset/`

Current transition-onset and early-warning task.

Typical contents:

- onset-label audits;
- warning-score triviality checks;
- event-time trajectory analyses;
- false-alarm and episode-detection diagnostics.

### `scripts/modeling/track_b_destination/`

Current primary destination-prediction task.

Typical contents:

- target construction and robustness audits;
- class-drift, origin-regime, and temporal-clustering diagnostics;
- classical destination baselines;
- causal path preparation;
- matched ESN and Rydberg experiments;
- paired incremental-loss and threshold diagnostics.

### `scripts/modeling/shared/`

Reusable modeling utilities that are not part of `src/qpitome_qrc/` and are shared by more than one experiment family. Prefer promoting stable reusable code into `src/qpitome_qrc/` rather than allowing this folder to grow indefinitely.

### `scripts/modeling/archive/`

Superseded modeling runners that remain necessary for provenance but are no longer active entry points. Move a script here only when its replacement and historical role are documented.

## Other script areas

### Active root entry points

Keep current operational Phase 3 runners at `scripts/` root until superseded by a canonical master runner:

- `run_phase3_ap_push.py`
- `run_phase3_ap_push_bitstring_shots.py`
- `run_phase3_esn_ridge_walkforward.py`
- `run_phase3_aquila_transfer_test.py`
- `run_phase3_aquila_transfer_test_explicit_key.py`
- `prepare_phase3_aquila_hardware_mvp.py`
- `qbraid_ahs_readiness_probe.py`

### `scripts/reference/`

Historical/reference runs retained for comparison:

- Phase 2 classical baselines;
- Phase 2 persistence baselines;
- Phase 2 feedback-TFIM reference;
- Phase 2 prediction exports.

### `scripts/exploratory/`

Non-modeling Phase 3 development experiments that do not belong to the structured `scripts/modeling/` workbench or canonical root entry points.

### `scripts/diagnostics/`

Cross-cutting hardware, geometry, scaling, and mechanism diagnostics that are not specific to one modeling task.

## Documentation rule

Every nontrivial modeling script must be discoverable from `scripts/modeling/README.md` and from the relevant protocol or experiment note.

For each script or tightly coupled script group, documentation must identify:

1. the scientific question;
2. the relevant script path or `src/` implementation;
3. required inputs and preprocessing;
4. output/result directory;
5. evaluation protocol and leakage controls;
6. principal result or current status;
7. whether the script is active, exploratory, diagnostic, retired, or archived.

A script is not considered integrated merely because it exists in Git history.

## Results

### `results/canonical/`

Standardized comparison outputs only. Future master-runner target.

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

Do not mass-delete historical files. Move only when classification is unambiguous. When a script is referenced in documentation, commands, tests, or manifests, update those references in the same cleanup commit.

The cleanup commit should contain only:

- directory creation;
- `git mv` operations;
- reference repairs;
- `scripts/modeling/README.md` and relevant documentation updates.

Do not add a permanent cleanup utility.

## Future output rule

New scripts must write to a purpose-specific result area:

- canonical comparison -> `results/canonical/`
- tuning/search -> `results/tuning/<search_name>/`
- exploratory modeling -> `results/experimental/<experiment_name>/`
- diagnostics -> `results/diagnostics/<diagnostic_name>/`
- finite-shot/hardware -> `results/hardware/<study_name>/`

`scratch/` and `/tmp` remain disposable and are not durable evidence.
