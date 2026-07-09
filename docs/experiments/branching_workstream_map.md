# Branching-state workstream map

Last updated: 2026-07-09

## Purpose

This file is the traceability map for the branching-state line of work. It is not a second experiment registry or a replacement for the detailed experiment notes. Its job is to make the active chain recoverable without browsing the repository.

The controlling scientific design remains:

```text
docs/experiments/regime_branching_analysis.md
```

The formal detector and label history remain:

```text
docs/experiments/branching_extractor_formalization.md
```

## Current chain

| Stage | Scientific question | Active runner(s) | Main result location | Main documentation | Status |
|---|---|---|---|---|---|
| Problem discovery | Is there a recurrent nontrivial market state after cheap broad-stress structure is removed? | exploratory work; no single preserved runner | historical exploratory artifacts | `docs/experiments/regime_branching_analysis.md` | controlling design |
| Formal extraction | Can the unstable-aftermath state be defined causally and reproducibly? | `scripts/regimes/audit_branching_extractor.py`; `scripts/regimes/audit_branch_outcome_labels.py` | `results/regimes/branching_extractor_audit_v1/`; `results/regimes/branch_outcome_label_audit_v1/` | `docs/experiments/branching_extractor_formalization.md` | active benchmark construction |
| Episode protocol | Can every eligible episode receive one leakage-safe OOS prediction? | `scripts/regimes/audit_episode_prequential.py` | `results/regimes/episode_prequential_audit_v1/` | `docs/experiments/episode_walkforward_protocol.md` | canonical evaluation protocol |
| Cheap branch probability | Do class priors, VIX, current state, or simple motion solve recovery versus relapse? | `scripts/baselines/comparison/run_branch_probabilistic_front.py` | `results/baselines/branch_probabilistic_front_v1/` | `docs/experiments/branch_probabilistic_front.md` | not trivially solved |
| HAR branch probability | Does a causal HAR normalization forecast resolve branch outcomes? | `scripts/baselines/comparison/run_branch_har_front.py` | `results/baselines/branch_har_front_v1/` | `docs/experiments/branch_har_front.md` | weak branch discriminator |
| HAR continuous audit | Is HAR still useful as a volatility forecaster inside branch states? | `scripts/diagnostics/har/audit_branch_har_continuous.py` | `results/baselines/branch_har_continuous_audit_v1/` | `docs/experiments/branch_har_front.md` plus this map | diagnostic; HAR remains useful overall |
| Ordered path classification | Does a 40-day ordered path improve recovery-versus-relapse prediction? | `scripts/baselines/esn/run_branch_path_reservoir_front.py` | `results/baselines/branch_path_reservoir_front_v1/` | `docs/experiments/branch_path_reservoir_front.md` | direct classifier negative for current architecture |
| Modern HAR residual | Can path representations predict HAR forecast error across branch episodes? | `scripts/baselines/esn/run_branch_har_residual_path_front.py` | `results/baselines/branch_har_residual_path_front_v1/` | this map; result manifests | overall negative; outcome heterogeneity observed |
| Residual input families | Do volatility-failure or HAR-aware channels improve the residual model? | `scripts/baselines/esn/run_branch_har_residual_input_families_v2.py` | `results/baselines/branch_har_residual_input_families_v1/` | this map; result manifests | exploratory; no overall win |
| Long-history preprocessing audit | Why did the first long reconstruction disagree with the project RV convention and earlier episode count? | `scripts/diagnostics/regime/audit_long_history_reconstruction_discrepancy.py` | `results/regimes/long_history_reconstruction_discrepancy_v1/` | this map | valid forensic audit |
| Canonical long-history replication | Does the modern branch/HAR-residual experiment survive on 1950–2026 data without retuning? | `scripts/baselines/esn/run_long_history_har_residual_replication_canonical.py`; engine: `scripts/baselines/esn/run_long_history_har_residual_replication.py` | `results/regimes/long_history_branch_reconstruction_v2/`; `results/baselines/long_history_har_residual_replication_v2/` | this map | current canonical long-history run |

## Canonical long-history reconstruction

The supported long-history entry point is:

```text
scripts/baselines/esn/run_long_history_har_residual_replication_canonical.py
```

It must pass all of the following before model fitting:

```text
modern RV parity: RMS realized volatility
rows: 19,246
range: 1950-01-03 through 2026-07-02
complete episodes: 80
modern frozen branch thresholds: unchanged
```

Current canonical outcome counts:

```text
mixed:    33
recovery: 28
relapse:  19
```

The canonical runner uses the exact modern project RV convention:

```text
sqrt(rolling_mean(log_return^2) * 252)
```

## Invalid long-history run removed

The earlier 74-episode long-history reconstruction used rolling standard deviation instead of the project's RMS realized-volatility definition. That changed state detection, outcome scaling, HAR inputs, and reservoir inputs.

The following artifacts were therefore deleted from the active working tree:

```text
results/regimes/long_history_branch_reconstruction_v1/
results/baselines/long_history_har_residual_replication_v1/
```

The obsolete v2/v3 wrappers were removed. Git history preserves the mistake; the active result tree does not.

The forensic audit remains because it established the bug and verified exact RMS parity against the modern processed dataset.

## Current evidence boundary

The current evidence supports these statements:

- broad stress and generic regime change contain substantial cheap classical structure;
- the branching state recurs and has heterogeneous future outcomes;
- naive path summaries do not exhaust temporal information;
- one primitive ESN architecture is not enough to characterize ESN capacity;
- unconditional HAR-residual correction does not improve overall branch forecasting;
- outcome-conditioned differences are diagnostically interesting but do not by themselves define a deployable target;
- the target space and ESN architecture space remain insufficiently explored to choose a final QRC endpoint.

The evidence does **not** yet justify:

- treating relapse as the final QRC target;
- claiming ESN failure from one small architecture;
- claiming a reservoir advantage;
- claiming that volatility forecasting is the only relevant endpoint.

## Next step

No new runner should be added until it is placed in the existing taxonomy and added to this map or an existing experiment note.

The next scientific step is to map temporal value across multiple continuous targets on the canonical long-history episodes before committing to a QRC endpoint. That experiment has not yet been implemented.

## Submission extraction rule

For this workstream, final submission selection should start from this file, not from repository browsing. A candidate artifact should be included only if it appears in the current chain above and is marked as scientifically active or submission-relevant in the final project review.
