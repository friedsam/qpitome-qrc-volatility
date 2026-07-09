# Project state and submission lineage

Last updated: 2026-07-09

This is the authoritative project-level record. Detailed experiment history remains in topic documentation and Git history.

## Current scientific thesis

The challenge is fundamentally about **forecasting regime changes**, not maximizing generic volatility-forecast accuracy.

Broad 20-day realized-volatility forecasting is strongly explained by cheap classical information, especially VIX/HAR-like level structure. The more interesting Track A problem is a recurrent causal **stressed-but-stabilizing branching state**: markets remain stressed and damaged, appear to stabilize, then resolve into recovery, relapse, or mixed outcomes.

The key empirical distinction is:

```text
volatility forecast skill
!=
regime-transition forecast skill
```

The current target is therefore:

```text
P(recovery versus relapse | information available at branch date)
```

No quantum advantage has been demonstrated.

## Current story

```text
broad volatility forecasting
    -> cheap classical models dominate
    -> challenge asks for regime-transition forecasting
    -> identify recurrent stressed/stabilizing branching state
    -> verify heterogeneous future resolutions
    -> VIX and static state do not resolve recovery versus relapse well
    -> state + motion provides weak contemporary transition ranking signal
    -> HAR volatility forecast remains strong almost everywhere except relapse branches
    -> generic ESN representations fail
    -> generic PCA compression preserves variance but not incremental transition value
    -> residual-targeted ESN correction also fails
    -> retain baseline-plus-correction architecture
    -> redesign reservoir input around transition-specific path ordering
    -> test matched compact classical temporal control before QRC
```

## Proven or strongly supported findings

### 1. Broad volatility level is classically easy relative to the transition task

Status: **KEEP**

Historical canonical HAR:

```text
rv_5d, rv_10d, rv_20d, rv_60d, vix_close
-> StandardScaler
-> Ridge(alpha=1.0)
-> future_rv_20d
```

VIX alone is nearly as strong as the headline HAR+VIX model on the broad level target. The dominant baseline is largely a forward-looking implied-volatility information effect, not evidence that generic linear dynamics solve regime transitions.

Relevant files:

```text
src/qpitome_qrc/baselines/branch_har.py
scripts/canonical/run_canonical_har.py
```

### 2. The stressed-but-stabilizing branching state is real enough to study

Status: **KEEP**

The state is defined causally from high realized volatility, persistent drawdown, short-horizon volatility deceleration, partial stabilization, and evidence that a decline actually occurred. VIX and future targets do not enter detection.

The phenomenon survived nearby threshold definitions and independent long-history reconstruction. Exact counts and hard labels remain formulation-dependent; the recurrent morphology is stronger evidence than any single classifier score.

Canonical implementation:

```text
src/qpitome_qrc/regimes/branching_state.py
```

Primary documentation:

```text
docs/experiments/branching_extractor_formalization.md
docs/experiments/branching_workstream_map.md
```

### 3. Episode prediction requires leakage-safe chronological evaluation

Status: **KEEP**

Accepted geometry: expanding prequential evaluation. Before predicting one episode, train only on earlier episodes whose full future outcome windows have completed.

Canonical implementation:

```text
src/qpitome_qrc/evaluation/episode_prequential.py
```

### 4. Episode-level power is a hard limitation

Status: **KEEP AS LIMITATION**

Modern canonical sample:

```text
48 complete episodes
17 recovery
13 relapse
18 mixed
17 binary OOS predictions
```

Long-history reconstruction:

```text
80 complete episodes
28 recovery
19 relapse
33 mixed
37 binary OOS predictions
```

Small AUC differences cannot support strong superiority claims alone. Probability quality, temporal stability, ablations, and matched controls are mandatory.

### 5. Cheap direct transition baselines are weak but informative

Status: **KEEP**

Modern direct binary results:

```text
historical class rate      AUC 0.492
VIX only                   AUC 0.386
current state              AUC 0.500
state + motion             AUC 0.621
```

Interpretation:

- VIX does not resolve the transition.
- Static state does not resolve the transition.
- A small amount of recent motion adds weak ranking information.
- Sample size is too small for a strong performance claim.

Canonical features:

```text
stress_ratio
drawdown_120d
rv_ratio_5_20_branch
return_5d_branch
rv_5d_change_5d_branch
worst_return_5d_in_prior_window
```

### 6. HAR adds no useful direct branch-transition signal

Status: **KEEP AS SUPPORTING NEGATIVE RESULT**

Direct HAR-derived branch classifiers do not improve on `state_plus_motion`.

The more important continuous result is different:

```text
HAR volatility forecasting works well in recovery branches
HAR volatility forecasting works very well in mixed branches
HAR volatility forecasting fails specifically in relapse branches
```

Canonical reproduced innovation R2:

```text
all complete episodes          +0.315
binary recovery/relapse        +0.035
mixed                          +0.783
recovery                       +0.623
relapse                        -0.263
```

This supports the project thesis that broad volatility predictability and transition predictability are different objects.

### 7. Generic ESN reservoir representations are closed as a primary lane

Status: **CLOSED / KEEP AS DIAGNOSTIC EVIDENCE**

The generic ESN investigation tested:

- reset and continuous state;
- 50, 300, and 500 units;
- historical spectral-radius and leak configurations;
- fixed input normalization;
- explicit bias;
- final-state and mean-state trajectory summaries;
- 5- and 10-component PCA compression;
- incremental addition to `state_plus_motion`;
- residual-targeted 1-3 component PLS compression;
- additive correction to a frozen classical baseline.

No usable improvement emerged.

The strongest interpretation is:

> The generic reservoir is learning structure, but not structure aligned with recovery-versus-relapse information missing from the classical transition baseline.

Detailed records:

```text
docs/experiments/branch_esn_architecture_audit.md
docs/experiments/branch_esn_architecture_audit_results.md
docs/experiments/branch_residual_offset_correction_results.md
```

### 8. High variance retention does not validate quantum opportunity

Status: **KEEP AS METHODOLOGICAL PRINCIPLE**

Compact ESN results showed:

```text
5 PCs preserve roughly 95% of reservoir variance
10 PCs preserve roughly 99.8-99.9%
```

Yet adding those components worsened incremental transition prediction.

Therefore:

```text
preserving variance
!=
preserving information complementary to the classical baseline
```

If a classical model performs nearly identically after PCA, that establishes only that PCA preserved what that classical model needs. It provides no evidence either for or against whether the same compression preserves information a quantum model could exploit.

This is not an argument against PCA. It is an argument against using classical equivalence after PCA as validation of quantum opportunity.

## Current unresolved scientific question

The immediate question is now:

> **Among episodes with similar branch-point state and recent motion, does temporal ordering within the preceding path contain stable information about recovery versus relapse?**

The next model should not be asked to reconstruct information already handled by the classical baseline.

Retained architecture:

```text
frozen state_plus_motion baseline
+
small temporal correction
```

The correction is the only object under comparison.

## QRC design direction

Do not send the previous generic six-channel path directly into a quantum reservoir and hope the circuit discovers the transition structure automatically.

The first task-aligned design focuses on four causal path contrasts:

```text
1. shock recurrence after apparent stabilization
2. rebound efficiency after damage
3. volatility-relaxation smoothness versus re-acceleration
4. return-volatility phase relation
```

Preferred minimal sequence:

```text
u1(t): signed return / local volatility
u2(t): downside-shock recurrence signal
u3(t): change in log(RV5 / RV20)
u4(t): drawdown-repair increment
```

The first QRC candidate is deliberately compact:

```text
4 qubits
40 sequential time steps
fixed recurrent entangling layer
8 observables total:
  4 single-qubit Z
  4 nearest-neighbor ZZ
```

Detailed design:

```text
docs/experiments/task_aligned_qrc_transition_design.md
```

## Required admission gate before QRC

The task-specific sequence must first pass:

```text
1. endpoint redundancy audit
2. temporal-order destruction control
3. matched compact classical temporal control
4. baseline-plus-correction OOS evaluation
```

The purpose is not to require classical success before quantum success. The purpose is to verify that the proposed channels actually represent the intended missing-information hypothesis and are not trivial rewrites of the baseline.

## Active artifact map

### KEEP — active dependencies

```text
docs/PROJECT_STATE.md

docs/experiments/branching_extractor_formalization.md
docs/experiments/branching_workstream_map.md
docs/experiments/branch_esn_architecture_audit_results.md
docs/experiments/branch_residual_offset_correction_results.md
docs/experiments/task_aligned_qrc_transition_design.md

src/qpitome_qrc/regimes/branching_state.py
src/qpitome_qrc/evaluation/episode_prequential.py
src/qpitome_qrc/baselines/branch_probabilistic.py
src/qpitome_qrc/baselines/branch_har.py

scripts/baselines/comparison/run_branch_probabilistic_front.py
scripts/baselines/comparison/run_branch_har_front.py
scripts/diagnostics/har/audit_branch_har_continuous.py

results/regimes/branching_extractor_audit_v1/
results/regimes/branch_outcome_label_audit_v1/
results/regimes/episode_prequential_audit_v1/
results/baselines/branch_probabilistic_front_v1/
```

### KEEP — important rejected-hypothesis evidence

```text
src/qpitome_qrc/baselines/branch_path_reservoir.py
src/qpitome_qrc/baselines/branch_esn_audit.py
src/qpitome_qrc/baselines/branch_residual_correction.py

scripts/baselines/esn/run_branch_path_reservoir_front.py
scripts/baselines/esn/run_branch_esn_architecture_audit.py
scripts/baselines/esn/run_branch_historical_esn_capacity_admission.py
scripts/baselines/esn/run_branch_compact_reservoir_incremental.py
scripts/baselines/esn/run_branch_residual_offset_correction.py
```

These remain active because deleting them would remove the evidence that the generic ESN failure is not explained by one obvious architecture defect, readout dimension, or forced reconstruction of the classical component.

### ARCHIVE CANDIDATES — superseded or premature fronts

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

Nothing should be deleted before exact provenance and reproducibility are checked.

## Repository invariants

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
9. **Do not let volatility forecasting become the project objective by default.**
10. **Do not use variance retention or classical equivalence after compression as evidence about quantum opportunity.**
11. **Do not force a constrained reservoir to reproduce information already handled well by the frozen classical baseline when the scientific question is incremental value.**

## Immediate next action

Implement and audit the four task-specific causal path channels before building another reservoir.

Required order:

```text
1. implement causal task-specific path channels in src/
2. add unit tests for causality, clipping, and endpoint invariance
3. run endpoint redundancy audit
4. run temporal-order destruction control
5. freeze the four-channel sequence
6. build matched compact classical temporal control
7. only then implement the 4-qubit QRC
8. evaluate both as additive corrections to the same frozen baseline
```

No broad reservoir or circuit architecture search is justified at this stage.
