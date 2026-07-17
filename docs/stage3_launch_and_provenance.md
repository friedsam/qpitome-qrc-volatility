# Stage 3 launch and provenance note

This note freezes the agreed Stage 3 scope before scientific code is ported or changed.

## Branch lineage

- Target branch: `stage3-dev`
- Commit 0: `7d0f7acecc62c13e52518a719eea2e1015b39bd6`
- Commit 0 purpose: synchronize the canonical data layer from `stage1-dev`
- Scientific source branch: `phase3-refactor`
- Frozen source commit: `a69e256634ef1a9819bc93614fefd15c711b3c9e`
- Machine-readable scope: `docs/stage3_port_manifest.json`

This commit changes documentation and provenance only. It does not port or modify scientific code.

## Stage 3 task

Stage 3 forecasts the destination of causal uncertainty episodes: positive versus negative future cumulative weekly return, with the primary frozen specification using an eight-week horizon and a one-percentage-point neutral zone.

The protected classical baseline is 13-week momentum. Rydberg is the primary experimental reservoir. ESN and TFIM are matched reservoir controls.

## Frozen starting evidence

The existing committed Stage 3 evaluation contains 48 mature post-split episodes: 35 positive and 13 negative.

- Momentum is the strongest current baseline.
- ESN direct and momentum-plus-ESN correction worsen proper scores.
- Rydberg return-plus-uncertainty slightly raises ROC AUC relative to momentum but worsens log loss and Brier score.
- Paired episode loss favors Rydberg in 22 episodes and momentum in 26; the mean Rydberg loss increment is positive, therefore worse.

This negative/inconclusive result is the required starting point. It must be reproduced before architecture or evaluation changes are introduced.

## Port policy

### Must port

1. The Stage 3 target audit, causal episode generation, path preparation, classical baselines, ESN, Rydberg simulator, and paired increment analysis.
2. The reusable evaluation machinery for historical splitting, outcome maturity, cross-fitting, residualization, protected offsets, proper scores, cluster weighting, reconstruction, and feature-rank diagnostics.
3. The systematic reservoir-interrogation methods developed during the Day-5/D1 work:
   - input and feature assays;
   - structural null and sanity checks;
   - raw and residual augmentation;
   - nonlinear residual maps;
   - matched Gaussian controls;
   - PCA and reconstruction analysis;
   - higher-order observable assays;
   - regularization controls.
4. Rydberg, ESN, and TFIM reservoir code needed for matched Stage 3 comparisons.
5. qBraid-managed Aquila readiness, local-detuning controls, guarded submission, job retrieval, and hardware reporting.

### Preserve but do not activate initially

- expensive higher-order and spatial sweeps;
- TFIM disorder/topology investigations;
- virtual-node scaling;
- hardware-native bitstring reconstruction;
- cross-reservoir representation studies.

### Exclude

- direct AWS/AwsDevice authentication and submission;
- unapplied repository-reorganization destination paths;
- the D1 target as an active Stage 3 target;
- missing historical outputs represented as present;
- uncertain or undocumented artifacts without clear provenance;
- duplicate implementations where shared utilities already exist.

## Experimental development sequence

Every reservoir model should be examined in the same order:

1. input information;
2. encoding behavior;
3. reservoir-state variability, rank, memory, and shift;
4. output-feature information and redundancy;
5. classical readout and regularization;
6. incremental value beyond the protected baseline;
7. finite-shot, noise, and hardware transfer.

The core question is not whether a reservoir has high-dimensional features. It is whether those features contain stable residual forecasting information beyond momentum and outperform matched classical and random-feature controls.

## Planned commits

### Commit 2 — Frozen Stage 3 reproduction

Port the current Stage 3 task and reproduce the committed 48-episode metrics without scientific changes.

### Commit 3 — General evaluation layer

Generalize D1-specific evaluation utilities while retaining compatibility wrappers and historical interpretability.

### Commit 4 — Reservoir abstraction

Provide common feature-generation contracts for ESN, TFIM, and Rydberg without forcing the reservoirs into one internal design.

### Commit 5 — Stage 3 diagnostic harness

Unify baseline-only, direct reservoir, protected-offset, residualized, Gaussian, shuffled, and regularization-path experiments.

### Commit 6 — Local/spatial Rydberg upgrade

Introduce channel-specific physical control after the frozen implementation is reproduced and diagnosed.

### Commit 7 — qBraid Aquila integration

Port the qBraid-only hardware path after simulator evidence supports a transfer test.

## Commit 2 acceptance criteria

- 48 evaluation episodes;
- 35 positive and 13 negative outcomes;
- identical target and outcome-maturity logic;
- identical paths and frozen standardization;
- identical momentum, ESN, and Rydberg metrics;
- identical paired-loss summary;
- no immature-outcome leakage;
- fixed seeds and machine-readable manifests.

## Provenance rules

Code and committed results take precedence over recollection and stale documentation. Historical output locations must not be inferred from current script defaults. Missing or uncertain artifacts must remain explicitly marked as missing or uncertain. Path changes and result migrations require a recorded provenance map before execution.
