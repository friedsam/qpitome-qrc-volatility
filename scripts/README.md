# Script organization

The `scripts/` tree is organized by scientific and operational topic.

Reusable model, evaluation, and data logic belongs in `src/qpitome_qrc/`. Scripts should remain thin entry points that configure and execute workflows.

## Canonical workflows

`canonical/`

Primary comparison and benchmark orchestration:

- `run_canonical_suite.py` — canonical suite orchestrator
- `run_canonical_comparison.py` — canonical comparison entry point
- `run_canonical_har.py` — canonical HAR workflow
- `run_canonical_tfim.py` — canonical TFIM workflow
- `run_master_comparison.py` — shared comparison implementation used by canonical and diagnostic workflows

## Data preparation

`data/`

Dataset construction and extension:

- `prepare_phase2_spy_vix_dataset.py`
- `extend_spy_vix_dataset.py`

## Classical baselines

`baselines/esn/`

ESN and matched-path evaluation workflows. Active branch-state runners:

- `run_branch_path_reservoir_front.py` — matched full-path linear versus reset/continuous ESN classifier on the modern branch sample
- `run_long_history_branch_resolution_replication.py` — frozen direct recovery-versus-relapse replication on the canonical 1950–2026 branch episodes
- `run_branch_har_residual_path_front.py` — modern-sample HAR-residual path benchmark
- `run_branch_har_residual_input_families.py` — input-family experiment implementation
- `run_branch_har_residual_input_families_v2.py` — corrected supported entry point for the input-family experiment
- `run_long_history_har_residual_replication.py` — reusable long-history volatility-residual replication engine
- `run_long_history_har_residual_replication_canonical.py` — supported canonical 1950–2026 volatility-residual replication entry point

`baselines/comparison/`

Cheap controls and cross-baseline comparison workflows. Active Phase 3 runners:

- `run_residual_front_baselines.py` — continuous innovation persistence/HAR front
- `run_regime_front_baselines.py` — broad next-regime controls
- `run_regime_change_front.py` — generic regime-change controls
- `run_branch_probabilistic_front.py` — boring branch-resolution probability baseline
- `run_branch_har_front.py` — causal HAR-derived branch-resolution baseline

No new runner should be created directly under `scripts/baselines/`.

## Quantum reservoir computing

`qrc/rf_qrc/`

RF-QRC experiment family, including tail behavior, temporal multiplexing, structured encoding, readout, scaling, and plotting workflows.

`qrc/rydberg/`

Temporal Rydberg-QRC workflows, including purged walk-forward evaluation, temporal probes, reservoir experiments, and robustness studies.

## Diagnostics

`diagnostics/mechanism/`

Mechanistic tests of reservoir behavior, including temporal memory and dual-chain structure.

`diagnostics/tail_behavior/`

Tail-event and finite-shot diagnostics.

`diagnostics/har/`

HAR failure analysis and matched-path diagnostics.

Active branch-state audit:

- `audit_branch_har_continuous.py` — branch-conditioned continuous HAR versus persistence

`diagnostics/regime/`

Market-regime discovery and state interpretation.

Active long-history audit:

- `audit_long_history_reconstruction_discrepancy.py` — forensic RV-convention and episode-count audit

`diagnostics/temporal_context/`

Tests of predictive context and rolling HAR/ESN behavior.

`diagnostics/temporal_structure/`

Frequency- and timescale-structure analysis.

## Hardware

`hardware/`

Aquila/qBraid readiness, preparation, and transfer workflows.

## Placement rules

New scripts must be placed in the appropriate topic directory from the start.

The same topic names should be used across:

- `scripts/`
- `results/`
- `docs/`

Historical scripts belong under `archive/`, not in the active script tree.

Reusable implementation logic belongs in `src/qpitome_qrc/`, not in scripts.
