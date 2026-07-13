# Results layout and provenance

Last updated: 2026-07-13

## Purpose

This document records the repository result-layout cleanup, the distinction between historical outputs and newly assigned reproducible destinations, and known missing artifacts. It should be consulted before moving, renaming, regenerating, or interpreting result files.

## Naming convention

For actively maintained modeling areas, outputs are stored directly in the corresponding parent directory. The former child experiment directory becomes a filename prefix:

```text
results/modeling/<area>/<experiment>__<filename>
```

Nested result directories are encoded with an additional prefix component:

```text
results/modeling/<area>/<experiment>__<nested_group>__<filename>
```

Example:

```text
results/modeling/day5_branching/baseline/
  cross_market_day5_direction_v1__day5_landmark_frame.csv
  cross_market_day5_direction_v1__robustness__metrics_by_market.csv
```

This convention prevents repeated generic filenames such as `summary_metrics.csv`, `predictions.csv`, and `manifest.json` from becoming ambiguous after directories are flattened.

## Areas already flattened

The following modeling areas have been flattened and their scripts updated on branch `phase3-refactor`:

- `results/modeling/weekly_regimes/`
- `results/modeling/transition_onset/`
- `results/modeling/transition_destination/`
- `results/modeling/day5_branching/baseline/`

Downstream consumers of the flattened Day 5 landmark frame were also updated.

## Day 5 branching structure

Scripts are organized by experiment family:

```text
scripts/modeling/day5_branching/
  baseline/
  falsification/
  residual_confirmation/
  spatial_rydberg/
  static_rydberg/
  temporal_controls/
```

At the time of the 2026-07-13 inventory, only these Day 5 result families were present in the current checkout:

```text
results/modeling/day5_branching/baseline/
results/modeling/day5_branching/static_rydberg/
```

The absence of the other result families from this checkout does not prove that their experiments were never run. Several historical scripts accepted a user-supplied `--outdir`, so the actual location of a particular run depended on the command used at execution time.

## Historical provenance limitation

The `results/modeling/` hierarchy did not exist when some early Day 5 experiments were run. Later refactoring assigned default destinations under `results/modeling/day5_branching/...` for future reproducibility. Those later defaults must not be treated as evidence of the original storage location of historical files.

In particular, the original spatial Rydberg shard and merge scripts required an explicit `--outdir`. The historical run used local shard directories across three Macs, followed by collection and merging on Mac 1. The exact supplied directory is not recoverable from the current scripts or shell history.

## Known missing Day 5 artifacts

The following outputs are known or strongly believed to have existed, but were not found under the current repository `results/` or `scratch/` trees during the 2026-07-13 inventory.

### Spatial Rydberg assay

Expected shard artifacts:

```text
predictions_shard_<n>.csv
feature_diagnostics_shard_<n>.csv
manifest_shard_<n>.json
```

Expected merged artifacts:

```text
predictions.csv
summary_metrics.csv
paired_score_deltas.csv
cluster_weighted_metrics.csv
all_market_model_ranking.csv
feature_diagnostics.csv
feature_diagnostics_summary.csv
merge_manifest.json
```

The shard computations were expensive and were distributed across three Macs. The files were collected and merged on Mac 1, but the collected shard directory and merged outputs are currently missing.

### Other Day 5 families not found in the current checkout

No outputs were found for the current script-defined destinations of:

```text
falsification/day5_first_passage_sanity
falsification/day5_input_audit
residual_confirmation/day5_residualized_rydberg
residual_confirmation/day5_residualized_rydberg_crossfit
residual_confirmation/day5_higher_order_output
residual_confirmation/day5_nonlinear_input_residual
residual_confirmation/day5_protected_input_residual
residual_confirmation/day5_standard_feature_assay
spatial_rydberg/day5_rydberg_pca_reconstruction
temporal_controls/cross_market_day5_fixed_esn_v1
```

These should be treated as **missing or unverified**, not as experiments that definitely were never run. Their historical outputs may have been written to explicit local `--outdir` locations before the current directory scheme was introduced.

## Existing Day 5 results preserved

The following result groups were present during the inventory and should be preserved:

### Baseline

```text
cross_market_day5_direction_v1
cross_market_day5_direction_v1/robustness
cross_market_day5_confirmatory_v1
cross_market_day5_regularized_controls_v1
```

These have been flattened into `results/modeling/day5_branching/baseline/` using filename prefixes.

### Static Rydberg

```text
cross_market_day5_fixed_quantum_feature_v1
cross_market_day5_fixed_rydberg_feature_v1
differential_local_rydberg_probe
static_rydberg_probe
```

These remained present at the end of the 2026-07-13 work session. Their downstream landmark input paths were updated to the flattened baseline location. Their own result directories were not yet flattened.

## Scratch results

Large collections under directories such as the following are separate Phase 3 Rydberg experiments and are not substitutes for the missing Day 5 spatial assay:

```text
scratch/rydberg_walkforward/
scratch/rydberg_temporal_reservoir/
scratch/canonical_cache/
```

Scratch outputs should not be moved or promoted solely based on broad filename matches such as `rydberg`, `spatial`, or `residual`. Promotion requires a clear producing script, configuration, and experiment identity.

## Cleanup rules

1. Do not infer historical provenance from a path introduced during later refactoring.
2. Before flattening a result family, map every producing script and downstream consumer.
3. Prefix filenames with the complete former relative directory identity.
4. Search the repository for old path literals after each update.
5. Do not silently regenerate expensive missing results and present them as historical originals.
6. If missing outputs are rerun, mark them as regenerated and record the code revision, command, machine, parameters, and date.
7. Preserve manifests and shard metadata whenever available.
8. Do not delete unknown files merely because they do not fit the current hierarchy.

## Recommended provenance for future runs

Every expensive or distributed run should record:

```text
script path
Git commit SHA
full command
explicit output directory
machine identifier
shard index and total shard count
start and completion time
input dataset versions
random seed
merge command
```

For distributed runs, the merge manifest should enumerate every input shard and include checksums or at least file sizes and row counts.
