# Modeling script index

This directory is the modeling workbench. Scripts are grouped by scientific purpose so a reader can understand what each experiment is for, what evidence it produced, and whether it remains active.

## Folder map

| Folder | Purpose | Status |
|---|---|---|
| `regimes/` | Weekly regime construction, HMM scoring, and state interpretation | Active infrastructure |
| `branching_day5/` | Historical day-5 barrier-defined recovery/relapse program and falsification record | Retired primary target; retained for provenance |
| `track_a_onset/` | Transition-onset and early-warning task | Secondary work in progress |
| `track_b_destination/` | Conditional positive/negative destination prediction | Primary current task |
| `shared/` | Reusable preparation or evaluation code shared across tasks | Keep small; promote stable code into `src/` |
| `archive/` | Superseded runners retained for provenance | Inactive |

## Regime infrastructure

### `regimes/run_weekly_regime_baselines.py`

**Question:** Do four-state weekly Gaussian HMMs provide better causal predictive density and more useful transition context than HMM2 and IID Gaussian baselines?

**Core implementation:** `src/qpitome_qrc/baselines/gaussian_hmm.py`

**Inputs:** long-history daily or weekly market data; the current S&P run uses `results/regimes/long_history_branch_reconstruction_v2/gspc_daily_engineered_1950_2026.csv`.

**Outputs:** weekly returns, one-step predictions, predictive scores, fit history, and manifest.

**Safeguards:** expanding causal history, filtered states only, annual refits, no smoothed future states in scoring.

**Documentation:**

- `docs/experiments/weekly_regime_transition_redesign.md`
- `docs/protocols/track_b_destination_protocol.md`
- `docs/PROJECT_STATE.md`

**Status:** active infrastructure. The implementation is a masked Baum-Welch approximation, not a faithful Bayesian replication of Maheu-McCurdy-Song.

### `regimes/analyze_weekly_regime_baselines.py`

**Question:** Where do restricted and unrestricted HMM4 differ by decade, return tail, and crisis window?

**Outputs:** score reconstruction checks and conditional score summaries.

**Status:** diagnostic supporting the conclusion that four states matter materially while the exact restricted topology contributes a smaller, tail-concentrated gain.

### `regimes/analyze_weekly_regime_states.py`

**Question:** Do the filtered HMM4 states have stable and economically interpretable aggregate profiles?

**Outputs:** probability checks, state profiles, persistence, and dominant-state transition summaries.

**Status:** active diagnostic. Aggregate semantics are credible; annual-refit identity stability remains incompletely audited.

## Historical day-5 branch program

The scripts in `branching_day5/` reconstruct the full recovery-versus-relapse program: classical D0/D1/D2 baselines, static and temporal reservoir probes, spatial Rydberg assays, protected-offset and residual tests, nonlinear/PCA/higher-order controls, and the exact falsification that retired D1.

**Primary documentation:**

- `docs/experiments/branching_workstream_map.md`
- `docs/experiments/branch_path_reservoir_front.md`
- `docs/experiments/day5_spatial_rydberg_assay_matrix.md`
- `docs/experiments/day5_spatial_rydberg_results.md`
- `docs/experiments/branch_residual_offset_correction_results.md`
- `docs/protocols/day5_static_rydberg_probe.md`
- `docs/protocols/day5_differential_local_rydberg_diagnostic.md`
- `docs/results/day5_static_nonlinear_probe_20260711.md`
- `docs/PROJECT_STATE.md`

**Fundamental result:** D1 was invalidated as a headline because its four nominal features reduce to rank two and a zero-parameter first-passage corridor-position null reproduced approximately 94.6% of its apparent log-loss improvement.

**Status:** provenance and falsification record. Do not extend the barrier-defined destination target as the primary task.

## Track A: onset detection

### `track_a_onset/audit_branch_onset_triviality.py`

**Question:** Is the transition-onset label trivially predicted by volatility, recent return, or HMM confidence?

**Inputs:** causal restricted-HMM filtered probabilities and weekly returns.

**Outputs:** chronological train/holdout score audit.

**Result:** long-regime uncertainty and one-week probability motion contain meaningful signal; realized-volatility features are close to trivial.

**Documentation:** `docs/experiments/weekly_regime_transition_redesign.md` and `docs/PROJECT_STATE.md`.

**Status:** active diagnostic.

### `track_a_onset/analyze_branch_onset_trajectory.py`

**Question:** Does the warning score rise early enough before persistent broad-regime transitions to be operationally meaningful?

**Outputs:** frozen thresholds, episode detection rates, conditional non-event false alarms, and event-time trajectories.

**Result:** long-regime uncertainty warned on 45 of 58 holdout events during weeks -4 through -1, with median first warning three weeks early.

**Status:** credible work in progress; Track A remains secondary to Track B.

## Track B: destination prediction

### `track_b_destination/audit_branch_destination_targets.py`

**Question:** Which destination horizon and neutral zone produce a nontrivial, adequately populated task after causal uncertainty admission?

**Outputs:** episode table, target counts, class balance, and univariate predictability audit.

**Result:** provisional primary target is eight-week observed cumulative return with a +/-1% neutral zone.

**Documentation:** `docs/protocols/track_b_destination_protocol.md`.

**Status:** active target audit.

### `track_b_destination/analyze_branch_destination_structure.py`

**Question:** Is the provisional target coherent across train/holdout, origin regime, decade, and temporal clusters?

**Outputs:** origin/partition summary, prior baselines, feature AUC by origin, decade summary, and episode clusters.

**Result:** positive prevalence rises from 0.602 in training to 0.729 in holdout; bear-side and bull-side episodes differ; temporal clustering is substantial.

**Status:** active structural diagnostic.

### `track_b_destination/run_branch_destination_classical_baselines.py`

**Question:** What is the smallest credible classical baseline ladder for Track B under class drift?

**Models:** historical prior, 13-week momentum logistic, compact directional logistic, and compact transition logistic.

**Safeguards:** prequential holdout and outcome-maturity filtering.

**Result:** momentum is the strongest simple probability baseline, but threshold 0.5 yields positive recall 1.0 and negative recall 0.231.

**Status:** active baseline; operating threshold and calibration are not finalized.

### `track_b_destination/prepare_branch_destination_paths.py`

**Question:** Can classical and Rydberg reservoirs receive an identical causal path tensor?

**Output:** 131 x 13 x 8 episode tensor plus metadata and fixed pre-split standardization.

**Status:** active shared data preparation for Track B.

### `track_b_destination/run_branch_destination_esn_baseline.py`

**Question:** Does one matched classical reservoir configuration add information beyond the momentum offset?

**Result:** the one tested ESN configuration worsened AUC, log loss, Brier score, and negative recall.

**Status:** configuration-level negative result only. ESNs have not been fairly characterized.

### `track_b_destination/prepare_branch_destination_rydberg_inputs.py`

**Question:** How can the causal Track B paths be mapped to compact, bounded, label-independent controls?

**Encodings:** return only; return plus uncertainty; return plus volatility as an ablation.

**Status:** active input-preparation experiment. Volatility remains an ablation, not the primary target.

### `track_b_destination/run_branch_destination_rydberg_simulator.py`

**Question:** Does a first six-atom temporal Rydberg architecture add destination information beyond momentum?

**Architecture:** exact statevector, six-atom chain, sequential same-control detuning encoding, final/mean occupations and nearest-neighbor pairs.

**Result:** return-plus-uncertainty improved AUC from 0.664 to 0.681 but worsened log loss and Brier score.

**Status:** preliminary architecture probe; not promoted and not evidence against Rydberg QRC generally.

### `track_b_destination/analyze_branch_destination_rydberg_increment.py`

**Question:** Is the apparent Rydberg ranking gain stable episode by episode?

**Outputs:** paired losses, bootstrap interval, class-conditional probability shifts, and threshold tradeoffs.

**Result:** lower loss on 22 of 48 episodes and higher loss on 26; mean excess log loss +0.0081 with a wide interval crossing zero.

**Status:** diagnostic showing exploratory ranking signal without stable probability improvement.

## Integration rule

When a script is added or materially changed, update this index and the relevant protocol/experiment note in the same commit. Durable results must also identify the producing script and source commit. A script without these references is unfinished repository work.
