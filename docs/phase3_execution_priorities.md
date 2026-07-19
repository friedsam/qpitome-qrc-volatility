# Phase 3 execution priorities

## Purpose

This document records the current execution order so work can resume without reopening settled analysis or diverting time from the Rydberg implementation.

## Priority 1 — Rydberg QRC progress

This is the main concern until the Sunday deadline.

Immediate objective: establish a working, testable Rydberg QRC path with enough progress to support simulator, finite-shot/noise, and hardware execution requirements.

Current design direction:

- HAR remains the persistence baseline.
- QRC is initially a residual correction layer: `prediction = HAR + QRC residual`.
- Main architecture is a hybrid spatial-temporal Rydberg reservoir:
  - local detuning encodes slow context/channel identity;
  - a global time-dependent waveform injects the recent ordered volatility path;
  - persistent evolution provides fading memory;
  - intermediate interactions provide nonlinear trajectory mixing;
  - local Z and selected ZZ/connected correlations are measured at two or three probe times.
- Required controls:
  - ordered input;
  - shuffled input;
  - reversed input where useful;
  - reset/static ablation;
  - interaction-off or weakened-interaction control;
  - matched classical baselines.
- Noise and hardware execution are mandatory challenge deliverables, not optional follow-up work.

Do not delay this work for broad classical exploration or full submission-pipeline refactoring.

## Priority 2 — minimal corrected classical confirmation on a second Mac

Purpose: confirm that the classical conclusions survive a clean rerun with the frozen structural OHLC quality policy.

This should be a narrow confirmation run, not another search or model-selection exercise.

Retained models:

- HAR;
- PLS5, shrinkage 0.5, ordered and shuffled — RMSE-oriented frozen candidate;
- PLS10, shrinkage 0.75, ordered and shuffled — order-sensitive frozen candidate.

Requirements:

- use the same corrected chronological folds and frozen parameters;
- apply the structural bad-print policy reproducibly;
- exclude contaminated training samples when rebuilding the quality-gated lineage;
- retain raw and quality-gated outputs side by side;
- use no test rows;
- regenerate point predictions and summary metrics;
- rerun the episode-cluster bootstrap for the ordered-versus-shuffled comparison;
- record environment, commit, source hashes, seeds, and immutable result paths.

This run can proceed on another Mac with minimal disruption while Rydberg development continues here.

## Priority 3 — complete submission-grade data and agent pipeline

This work is deferred until one of the following:

- meaningful Rydberg progress has been established;
- a long simulator/hardware job is running;
- access or hardware latency creates otherwise idle time.

The final goal is one clean reproducible pipeline that supports both the earlier Stage 1 project and the current transition-forecasting/QRC project, and that can be used to validate:

- the qBraid Run button;
- the qBraid Skill;
- environment creation and dependency installation;
- deterministic dataset construction;
- provenance and required-output validation.

Current status:

Completed:

- reusable structural OHLC bad-print policy;
- causal frozen rule;
- immutable audit output;
- sample-level contamination annotation;
- documentation;
- expected integrity counts: 12 flagged rows, 2 indices, 19 contaminated sample IDs.

Still required:

1. Integrate the structural audit into the canonical data workflow.
2. Make the relevant dataset builders consume the quality metadata.
3. Emit explicit raw and quality-gated dataset lineages.
4. Propagate the quality-policy version and source hashes into manifests.
5. Add required-output and integrity checks to `scripts/runs/run_submission.py`.
6. Preserve support for the existing older data workflow.
7. Add the new transition-forecasting dataset workflow without mixing it with exploratory model analysis.
8. Verify a clean-environment run through the same interface intended for qBraid.
9. Only after final model/QRC commands are frozen, append them to the submission workflow.

The agent run is an end-product reproduction interface. It should not encode exploratory analysis, cross-machine interpretation, or open-ended model selection.

## Structural OHLC policy status

The current audit command is:

```bash
python scripts/transition_forecasting/quality/run_structural_bad_print_audit.py \
  --raw-root data/raw/transition_forecasting/global_stock_indices_historical_data/individual_indices_data \
  --manifest results/transition_forecasting/modeling/build_stage_e_chronological_dataset/chronology_rematch_8fold_002/rematched_rolling_manifest.csv \
  --expected-flagged 12
```

Expected counts:

- `flagged_rows = 12`;
- `affected_indices = 2`;
- `manifest_rows = 22784`;
- `contaminated_rows = 29`;
- `contaminated_sample_ids = 19`;
- `input_contaminated_rows = 24`;
- `target_contaminated_rows = 12`;
- `raw_data_modified = false`;
- `test_rows_used = 0`.

The existing filtered frozen-prediction analysis is a sensitivity analysis. The definitive corrected confirmation requires rebuilding and retraining with contaminated training samples excluded.

## Work-order rule

Unless blocked by compute, access, or hardware latency:

1. Rydberg implementation and mechanism assays.
2. Minimal second-Mac corrected classical confirmation.
3. Submission-grade data/agent pipeline completion.

Do not allow Priority 3 infrastructure work to consume the remaining Rydberg development window.
