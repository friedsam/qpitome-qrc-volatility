# Data pipeline completion plan

## Scope

This document records only the remaining work needed to make the repository data pipeline reproducible, submission-grade, and usable through the qBraid Run button / Skill interface.

It deliberately excludes exploratory analysis, model selection, Rydberg design, and cross-machine interpretation.

## Current state

Already implemented:

- causal structural OHLC bad-print policy in `src/transition_forecasting/quality/structural_bad_prints.py`;
- CLI audit in `scripts/transition_forecasting/quality/run_structural_bad_print_audit.py`;
- documentation of the frozen policy in `docs/structural_ohlc_quality_policy.md`;
- immutable audit outputs;
- sample-level input/target contamination annotation;
- expected integrity counts for the current source-data lineage:
  - 12 flagged OHLC rows;
  - 2 affected indices;
  - 19 contaminated sample IDs;
  - 29 contaminated manifest rows.

What is not yet complete:

- the canonical data workflow does not call the structural audit;
- Stage E builders do not yet consume the quality flags during dataset construction;
- raw and quality-gated datasets are not emitted as first-class parallel lineages;
- the run interface does not validate the new outputs or integrity counts;
- the current audit command requires an already-built manifest, so it is not yet positioned correctly in a clean end-to-end build.

## Target pipeline

The finished transition-forecasting data path should be:

```text
raw OHLC files
  -> raw-file inventory and schema/range validation
  -> structural OHLC bad-print audit
  -> canonical daily volatility series with quality metadata
  -> transition catalogue
  -> candidate pool
  -> chronological fold/rematching builder
  -> sample manifest annotated with input/target contamination
  -> raw dataset lineage
  -> quality-gated dataset lineage
  -> validation of counts, leakage rules, hashes, and required outputs
```

Raw source files must never be edited, interpolated, or overwritten.

## Implementation steps

### 1. Make the structural audit independent of a pre-existing Stage E manifest

The first audit stage should require only the raw OHLC root and an output directory.

Create or extend a command that writes:

- `daily_ohlc_quality_audit.csv.gz`;
- `frozen_structural_bad_prints.csv`;
- `summary.json`;
- `params.json`;
- source-file hashes.

The current command may retain optional manifest annotation, but manifest input must not be required for the raw-data audit stage.

Required guard:

- fail if the current source-data lineage does not yield exactly 12 flagged rows and 2 affected indices, unless an explicit maintenance override is provided.

### 2. Add a canonical daily-series builder with quality metadata

The daily Parkinson-volatility construction should emit, per index/date:

- raw OHLC lineage fields or source reference;
- `log_parkinson_volatility`;
- `suspected_bad_print`;
- quality-policy version/hash;
- source-file SHA-256;
- build timestamp and code commit.

The quality flag should be joined by exact `(index, date)` identity.

Do not replace or impute the volatility value. Preserve the raw value and attach the flag.

### 3. Make catalogue construction quality-aware but non-destructive

Transition-catalogue construction should accept a quality policy:

- `raw`;
- `exclude_structural_bad_prints`.

For `raw`, preserve current behavior.

For `exclude_structural_bad_prints`, exclude flagged daily observations before computing catalogue-dependent quantities only where this is methodologically intended and explicitly recorded. Avoid silently changing event definitions without documenting the consequence.

Preferred implementation:

- build the raw catalogue once;
- propagate daily quality flags through the event/candidate lineage;
- create a separate gated catalogue or gated eligibility column;
- preserve both outputs.

### 4. Integrate contamination logic into the chronological dataset builder

The Stage E builder should annotate each sample using exact trading-row information intervals:

- input contamination when `input_start_date <= flagged_date <= origin_date`;
- target contamination when `origin_date < flagged_date <= target_end_date`;
- `bad_print_any = bad_print_in_input OR bad_print_in_target`.

Emit the bad-print dates and boolean flags in the canonical sample manifest.

The builder must create two explicit sample sets:

- raw: all otherwise-valid samples;
- quality-gated: samples with `bad_print_any == False`.

Contaminated training samples must be excluded from quality-gated model fitting. Filtering only validation predictions is a sensitivity analysis, not a complete rebuilt lineage.

### 5. Preserve chronological and leakage guarantees in both lineages

For raw and quality-gated outputs, validate:

- eight chronological rolling-origin folds;
- exact information intervals;
- 10-calendar-day embargo;
- no episode crossing split boundaries;
- no test rows used during development/confirmation;
- no cross-lead control-origin reuse;
- strict partition-local control matching;
- no duplicate sample identities;
- expected positive/control completeness after gating, or explicit documented reasons when gating removes a matched sample.

Important design decision to implement explicitly:

- when one member of a matched positive/control set is contaminated, decide whether to remove only that sample or remove/rebuild the matched set.
- the chosen rule must preserve balance and be recorded in the manifest.
- do not silently leave incomplete matched groups.

### 6. Emit first-class immutable data artifacts

Each build should write to a new unique result directory and include:

- raw daily series;
- structural quality audit;
- raw transition catalogue;
- quality-aware/gated catalogue or eligibility table;
- raw chronological manifest;
- quality-gated chronological manifest;
- fold tensors/data files for both lineages where required;
- contamination summary;
- balance diagnostics;
- leakage/integrity report;
- `params.json`;
- `summary.json`;
- source and output hashes;
- environment and Git commit metadata.

No undocumented moves, overwrites, or deletion of prior result directories.

### 7. Update `scripts/runs/run_submission.py`

Do not add exploratory model analysis.

Add a dedicated data workflow for transition forecasting, for example:

```bash
python scripts/runs/run_submission.py transition-data
```

The workflow should call only stable data-building and validation commands.

It should:

1. inventory/validate raw OHLC inputs;
2. run the structural audit;
3. build the canonical quality-annotated daily series;
4. build catalogue/candidate data;
5. build raw and gated chronological datasets;
6. validate required outputs and integrity counts;
7. write a top-level run manifest with command logs, hashes, environment, branch, and commit.

Preserve the existing `data` and `validate-data` workflows for the earlier project unless they are intentionally migrated with backward-compatible output paths.

### 8. Add required-output and semantic validation

The runner should not merely check that files exist.

Add semantic checks for at least:

- 12 flagged OHLC rows;
- 2 affected indices;
- 19 contaminated sample IDs for the currently frozen source lineage;
- zero raw-file modification;
- zero test-row use;
- expected fold count;
- no leakage violations;
- both raw and quality-gated manifests present;
- non-empty tensors/data outputs;
- hashes recorded for every required artifact.

If the raw-data hashes change, fail with a clear source-lineage drift message instead of assuming the old expected counts remain valid.

### 9. Add tests

Unit tests:

- June 2 and June 3, 2008 `^XAX_data` rows are flagged;
- ordinary large but coherent crisis moves are not flagged solely because the range is large;
- rolling threshold uses preceding observations only;
- input and target interval boundaries are correct;
- raw rows remain unchanged;
- manifest annotation is deterministic.

Integration test:

- run the transition data workflow on the retained repository data;
- assert expected counts and required artifacts;
- verify rerun in a fresh directory produces identical content hashes for deterministic outputs.

### 10. Make the qBraid interface use the same workflow

The qBraid Run button and Skill should invoke `scripts/runs/run_submission.py`, not separate hidden logic.

Environment setup must install all required dependencies and make repository imports work from a clean instance.

The qBraid-facing wrapper should expose:

- workflow name;
- optional output root/run ID;
- clear failure logs;
- final run-manifest path.

The same command must work locally and on qBraid.

## Open implementation decisions

Resolve these before coding the final builder integration:

1. Whether quality gating occurs before event detection or only at sample eligibility.
2. Whether contamination of one matched sample triggers rematching/removal of the complete matched group.
3. Canonical storage location and format for daily quality-annotated volatility series.
4. Whether the old and new projects share one top-level data workflow or retain separate named workflows under the same runner.
5. Which deterministic artifacts are committed versus generated during a run.

Record each decision in `params.json` and in the relevant builder documentation.

## Completion criteria

The data pipeline is complete when a clean clone/environment can run one documented command and produce:

- all required datasets for the older project;
- raw and quality-gated transition-forecasting datasets;
- identical deterministic hashes across machines using identical raw inputs and software versions;
- complete provenance and validation reports;
- no exploratory analysis or manual intermediate steps;
- an interface suitable for the qBraid Run button and qBraid Skill.
