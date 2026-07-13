# Documentation index

This repository uses documentation as part of the experimental protocol. Durable claims should be traceable to a protocol, script, result directory, and commit.

## Current scientific state

- [`PROJECT_STATE.md`](PROJECT_STATE.md): concise current project state and locked decisions.

## Protocols

- [`protocols/track_b_destination_protocol.md`](protocols/track_b_destination_protocol.md): current Track B target, evaluation protocol, baseline ladder, and evidence status.
- [`experiments/weekly_regime_transition_redesign.md`](experiments/weekly_regime_transition_redesign.md): rationale and evidence for replacing the barrier-defined primary task with weekly regime-transition Tasks A and B.

## Experiment notes

Use `docs/experiments/` for interpretation and decision records tied to one experiment family. These files may describe exploratory work but must label it as such.

## Result-directory standard

Every durable result directory should contain, where applicable:

- `manifest.json` or `run_manifest.json`;
- `summary_metrics.csv` or `metrics.csv`;
- `predictions.csv`;
- a short `README.md` explaining the question, data, protocol, comparator, and interpretation;
- commit SHA and input provenance.

Temporary `/tmp` outputs are acceptable during exploration but are not durable evidence.

## Script organization

Follow [`repo_organization.md`](repo_organization.md): organize by purpose, not by model family.

- current operational runners remain at `scripts/` root until superseded;
- model-development work belongs in `scripts/exploratory/`;
- mechanism, falsification, and ablation work belongs in `scripts/diagnostics/`;
- historical comparison runners belong in `scripts/reference/`.

Do not create new model-family directories such as `classical_models/` or `quantum_models/`. Do not add one-off maintenance scripts to the repository.
