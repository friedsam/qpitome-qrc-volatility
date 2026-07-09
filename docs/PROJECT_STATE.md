# Project state and submission lineage

Last updated: 2026-07-09

This is the authoritative project-level record. It is intentionally shorter than the research history. Detailed experiment notes remain in topic documentation and Git history.

## Current scientific thesis

Broad 20-day realized-volatility forecasting is strongly explained by cheap classical information, especially VIX/HAR-like level structure. The more interesting Track A problem is a recurrent causal **unstable-aftermath branching state**: markets remain stressed and damaged but appear to stabilize, then resolve into recovery, relapse, or mixed outcomes. The current project question is whether temporal models, especially reservoir models, can add predictive information where cheap classical volatility forecasting and static branch-state descriptions stop being decisive.

No quantum advantage has been demonstrated.

## Current story

```text
broad volatility forecasting
    -> cheap classical models dominate
    -> inspect where this stops answering the regime-transition question
    -> identify recurrent stressed/stabilizing branching state
    -> verify state recurrence and heterogeneous outcomes
    -> test whether HAR/VIX remains useful but ceases to determine outcome
    -> define the unresolved forecasting target
    -> ESN as closest classical reservoir control
    -> QRC only after the target and classical residual are fixed
```

## Proven or strongly supported findings

### 1. Broad volatility level is classically easy relative to the QRC task

Status: **KEEP**

The historical canonical HAR result used five features:

```text
rv_5d, rv_10d, rv_20d, rv_60d, vix_close
-> StandardScaler
-> Ridge(alpha=1.0)
-> future_rv_20d
```

Independent audit showed VIX alone is nearly as strong as the headline HAR+VIX model on the old level target. This changes the interpretation: the dominant baseline is largely a forward-looking implied-volatility information effect, not evidence that generic linear dynamics exhaust every transition problem.

Relevant files:

```text
src/qpitome_qrc/baselines/branch_har.py
scripts/canonical/run_canonical_har.py
```

### 2. The unstable-aftermath branching state is real enough to study

Status: **KEEP**

The state is defined causally from high realized volatility, persistent drawdown, short-horizon volatility deceleration, partial stabilization, and evidence that a decline actually occurred. VIX and future targets do not enter state detection.

The phenomenon survived nearby threshold definitions and independent long-history reconstruction. Exact episode counts and hard labels remain formulation-dependent; the existence of the recurrent branching morphology is much stronger than any single classifier score.

Canonical implementation:

```text
src/qpitome_qrc/regimes/branching_state.py
```

Primary provenance:

```text
docs/experiments/regime_branching_analysis.md
docs/experiments/branching_extractor_formalization.md
```

Primary audit outputs:

```text
results/regimes/branching_extractor_audit_v1/
results/regimes/branch_outcome_label_audit_v1/
```

### 3. Episode prediction requires leakage-safe chronological evaluation

Status: **KEEP**

The accepted current geometry is expanding prequential evaluation: before predicting one episode, use only earlier episodes whose full future outcome windows have already completed. Each eligible episode receives one out-of-sample probability.

Canonical implementation:

```text
src/qpitome_qrc/evaluation/episode_prequential.py
```

The earlier five-fold episode walk-forward protocol is useful history but is no longer the primary evaluation dependency.

### 4. Episode-level power is a hard limitation

Status: **KEEP AS LIMITATION**

Even long-history reconstruction yields only tens of independent recovery/relapse episodes. Small apparent differences between ESN and QRC cannot support strong superiority claims by episode-level AUC alone. Continuous forecasting metrics, probability calibration, and dependence-aware inference remain necessary.

### 5. Current QRC evidence is mechanistic, not advantageous

Status: **KEEP ELSEWHERE; NOT PART OF THIS JULY-8 AUDIT**

Current temporal Rydberg work established real nonlinear temporal memory but is beaten by an information-matched primitive classical control. Structured spatial results moved to approximate parity after fair preprocessing. The QRC architecture must not be redesigned until the forecasting target is fixed.

## Current unresolved scientific question

The immediate question is not yet "which QRC architecture wins?"

It is:

> **What exactly changes about classical volatility forecasting inside the branching state, and is that change linked to recovery versus relapse?**

The next accepted experiment must establish one of three cases:

1. HAR/VIX does not fail differently in branch states -> the proposed residual story is wrong.
2. HAR/VIX behaves differently in branch states, but forecast errors are unrelated to branch outcome -> volatility forecasting and branch resolution are separate tasks.
3. HAR/VIX forecast behavior or residuals differ systematically by eventual outcome -> a residual/temporal reservoir target is scientifically justified.

## Rework boundary

The current audit places the rework boundary here:

```text
branch discovery
-> extractor
-> labels
-> prequential protocol
-> cheap probabilistic controls
-> causal HAR inside branch states
================ REWORK / UNDERSTAND HERE ================
-> path ESN
-> HAR-residual ESN
-> input-family variants
-> long-history residual variants
```

The work after the line is not discarded, but it is **not an accepted dependency** until the HAR/branch relationship is understood deeply enough to define the target.

## Active artifact map

### KEEP — active dependencies

```text
docs/PROJECT_STATE.md

docs/experiments/regime_branching_analysis.md
docs/experiments/branching_extractor_formalization.md

src/qpitome_qrc/regimes/branching_state.py
src/qpitome_qrc/evaluation/episode_prequential.py
src/qpitome_qrc/baselines/branch_probabilistic.py
src/qpitome_qrc/baselines/branch_har.py

scripts/regimes/audit_branching_extractor.py
scripts/regimes/audit_branch_outcome_labels.py
scripts/regimes/audit_episode_prequential.py
scripts/baselines/comparison/run_branch_probabilistic_front.py
scripts/baselines/comparison/run_branch_har_front.py
scripts/diagnostics/har/audit_branch_har_continuous.py

results/regimes/branching_extractor_audit_v1/
results/regimes/branch_outcome_label_audit_v1/
results/regimes/episode_prequential_audit_v1/
results/baselines/branch_probabilistic_front_v1/
```

### MAYBE — diagnostic evidence only

```text
src/qpitome_qrc/baselines/branch_path_reservoir.py
scripts/baselines/esn/run_branch_path_reservoir_front.py
docs/experiments/branch_path_reservoir_front.md

scripts/baselines/esn/run_long_history_branch_resolution_replication.py
scripts/diagnostics/regime/audit_long_history_reconstruction_discrepancy.py
```

These may survive as one primitive temporal-control experiment and one forensic data-validation record. They are not yet part of the submission lineage.

### ARCHIVE CANDIDATES — scientifically superseded or premature fronts

```text
docs/experiments/regime_front.md
docs/experiments/regime_target_decision.md
docs/experiments/residual_front.md

docs/experiments/episode_walkforward_protocol.md
src/qpitome_qrc/evaluation/episode_walkforward.py
scripts/regimes/audit_episode_walkforward.py

scripts/baselines/comparison/run_regime_front_baselines.py
scripts/baselines/comparison/run_regime_change_front.py
scripts/baselines/comparison/run_residual_front_baselines.py

scripts/baselines/esn/run_branch_har_residual_path_front.py
scripts/baselines/esn/run_branch_har_residual_input_families.py
scripts/baselines/esn/run_branch_har_residual_input_families_v2.py
scripts/baselines/esn/run_long_history_har_residual_replication.py
scripts/baselines/esn/run_long_history_har_residual_replication_canonical.py
```

Associated result trees should move with their runners when the archive pass is executed. Nothing should be deleted before exact provenance and reproducibility are checked.

### REJECTED / INVALID

The earlier long-history reconstruction based on rolling standard deviation rather than the project RMS realized-volatility convention is invalid. Active-tree artifacts were already removed; Git history is sufficient provenance.

## Repository invariants from now on

1. **One scientific question -> one canonical runner.**
2. Shared behavior belongs in `src/`; scripts are thin orchestration.
3. No permanent `_v2`, `_v3`, `_final`, `_new`, or `_fixed2` filename chains.
4. Topic taxonomy must align across `scripts/`, `src/`, `results/`, and `docs/`.
5. No refactor is accepted without reproduction against an independent oracle or historical artifact.
6. Generated row-level output is not tracked unless it is irreplaceable evidence.
7. Every experiment must state before execution:
   - exact claim tested;
   - falsifying result;
   - existing artifact extended or replaced;
   - output location;
   - plausible paper role.
8. An artifact remains active only if deleting it would make a final claim unverifiable or remove the only evidence for an important rejected hypothesis.

## Time control

Working assumption: approximately 15 days remain.

```text
Days 1-2: audit, triage, reconstruct HAR/branch finding
Days 3-7: scientific convergence on target and baseline
Days 8-10: freeze final model/ablation chain
Days 11-15: no architecture wandering; reproduce, clean, archive, write, figures, final qBraid path
```

After day 10, new science requires explicit justification against submission risk.

## Immediate next action

Reconstruct the causal HAR behavior inside branch states from code and outputs, specifically:

```text
global HAR performance
vs branch-state HAR performance
vs matched stressed non-branch controls
vs recovery/relapse outcome-conditioned behavior
```

Do not run another reservoir experiment until this relationship is understood.
