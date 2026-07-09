# Episode-level walk-forward protocol

Last updated: 2026-07-08

## Purpose

The branching task is defined on sparse historical episodes, not daily rows. Evaluation must therefore use episodes as the unit of prediction and avoid leakage from overlapping future outcome windows.

Shared implementation:

```text
src/qpitome_qrc/evaluation/episode_walkforward.py
```

Audit runner:

```text
scripts/regimes/audit_episode_walkforward.py
```

## Current geometry

Working configuration:

```text
5 chronological folds
minimum 18 training episodes
6 validation episodes
minimum 4 test episodes per fold
40-row outcome horizon
0 additional embargo rows
```

The first test block begins only after the initial training-plus-validation episode budget. Remaining episodes are divided into five chronological, non-overlapping test blocks.

## Leakage rule

For a test block beginning at raw data row `test_branch_idx`, a prior episode is eligible for training or validation only when:

```text
episode.branch_idx + outcome_horizon < test_branch_idx - embargo_rows
```

Therefore every training/validation outcome is fully observable before the first branch point in the test block.

This is stricter than merely requiring the branch date itself to precede the test date.

## Train and validation assignment

For each fold:

```text
all leakage-safe prior episodes
-> most recent 6 become validation
-> all earlier eligible episodes become expanding training history
```

The validation block is chronological and remains prior to the test block.

## Test assignment

Test episodes are:

```text
chronological
non-overlapping across folds
never reused in another test fold
```

The test unit is an episode. Daily rows inside the same episode are never treated as independent predictions.

## Mixed outcomes

Fold geometry is constructed on all complete episodes:

```text
recovery
relapse
mixed
```

The protocol also reports the effective sample sizes for the recovery-versus-relapse binary task after mixed episodes are excluded.

This preserves one common chronological benchmark while allowing later comparison of:

```text
binary recovery vs relapse
three-class recovery vs relapse vs mixed
other probabilistic formulations
```

## Current status

This protocol is being audited before model fitting.

The audit must verify:

1. every fold has enough leakage-safe training episodes;
2. recovery and relapse are represented sufficiently in train, validation, and test;
3. test episodes cover multiple historical eras;
4. no training/validation outcome window crosses into the test start;
5. the resulting sample size is realistic for the intended boring baseline.

Only after this geometry passes inspection should the first episode-level probabilistic baseline be fit and its OOS predictions saved.
