# Project milestones

Last updated: 2026-07-13

## Current phase

The repository is in Phase 3 development and consolidation for the qBraid / MITRE / JonesTrading Global Industry Challenge.

Current priorities:

1. maintain a reproducible classical and QRC modeling pipeline;
2. preserve clear separation between canonical, experimental, diagnostic, and scratch outputs;
3. consolidate result paths without losing experiment identity;
4. document provenance and known missing artifacts;
5. retain bounded claims that distinguish mechanism evidence from predictive advantage;
6. prepare the Phase 3 submission and supporting reproducibility materials.

## Result-layout consolidation completed

The following result areas have been flattened and their scripts updated:

```text
results/modeling/weekly_regimes/
results/modeling/transition_onset/
results/modeling/transition_destination/
results/modeling/day5_branching/baseline/
```

The former child-directory name is now encoded as a filename prefix.

## Day 5 branching status

Present and preserved:

```text
baseline
static_rydberg
```

The baseline outputs have been flattened. The static Rydberg outputs remain present but have not yet been flattened.

Known missing or unverified historical output families:

```text
falsification
residual_confirmation
spatial_rydberg
temporal_controls
```

The distributed Day 5 spatial Rydberg shard and merge outputs are specifically known to be missing from the current checkout. See `docs/results_layout_and_provenance.md` for the detailed inventory and provenance limitations.

## Next repository-maintenance steps

1. Flatten `results/modeling/day5_branching/static_rydberg/` and update all producing scripts.
2. Audit documentation and scripts for remaining obsolete nested result paths.
3. Add run manifests and explicit provenance requirements to expensive distributed workflows.
4. Separate retained canonical evidence from exploratory or obsolete D1 experiments.
5. Avoid rerunning missing expensive experiments unless they remain necessary for the final Phase 3 claim set.
