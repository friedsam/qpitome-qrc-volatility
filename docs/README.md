# Documentation index

This repository uses documentation as part of the experimental protocol. Durable claims should be traceable to a protocol, script, result directory, and commit.

## Current scientific state

- [`PROJECT_STATE.md`](PROJECT_STATE.md): concise current project state and locked decisions.
- [`handoffs/phase3_post_d1_failure_and_track_b_rebuild_2026-07-13.md`](handoffs/phase3_post_d1_failure_and_track_b_rebuild_2026-07-13.md): full continuation handoff after the D1 failure analysis, weekly-HMM redesign, Task A audit, and preliminary Task B reservoir work.

## Protocols

- [`protocols/track_b_destination_protocol.md`](protocols/track_b_destination_protocol.md): current Task B target, evaluation protocol, baseline ladder, and evidence status.
- [`experiments/weekly_regime_transition_redesign.md`](experiments/weekly_regime_transition_redesign.md): rationale and evidence for replacing the barrier-defined primary task with weekly regime-transition Tasks A and B.

## Experiment notes

Use `docs/experiments/` for interpretation and decision records tied to one experiment family. These files may describe exploratory work but must label it as such.

## Repository documentation

- `docs/repository/layout_audit.md`: generated inventory after running the repository reorganization tool.
- `docs/repository/modeling_reorganization_manifest.json`: machine-readable record of script moves.

## Handoffs

Use `docs/handoffs/` for dated continuation handoffs. A handoff must distinguish:

1. established findings;
2. failed or invalidated claims;
3. exploratory results;
4. unresolved questions;
5. exact repository state and commands;
6. next-step constraints.

## Result-directory standard

Every durable result directory should contain, where applicable:

- `manifest.json` or `run_manifest.json`;
- `summary_metrics.csv` or `metrics.csv`;
- `predictions.csv`;
- a short `README.md` explaining the question, data, protocol, comparator, and interpretation;
- commit SHA and input provenance.

Temporary `/tmp` outputs are acceptable during exploration but are not durable evidence.

## Naming standard

Scripts are organized by scientific role rather than model buzzword:

```text
scripts/modeling/day5_barrier/
scripts/modeling/weekly_regimes/
scripts/modeling/task_a_onset/
scripts/modeling/task_b_destination/
scripts/modeling/classical_models/
scripts/modeling/quantum_models/
scripts/modeling/legacy_unclassified/
```

New files should be placed directly into the correct subdirectory. Do not recreate a flat `scripts/modeling` directory.
