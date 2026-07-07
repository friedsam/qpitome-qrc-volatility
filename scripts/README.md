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

ESN evaluation workflows.

`baselines/comparison/`

Cross-baseline comparison workflows.

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

`diagnostics/regime/`

Market-regime discovery and state interpretation.

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
