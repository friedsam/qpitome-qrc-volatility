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

## Modeling workbench

The modeling scripts remain under `scripts/modeling/`, but the directory is subdivided by scientific purpose:

```text
scripts/modeling/regimes/
scripts/modeling/branching_day5/
scripts/modeling/track_a_onset/
scripts/modeling/track_b_destination/
scripts/modeling/shared/
scripts/modeling/archive/
```

See [`repo_organization.md`](repo_organization.md) for folder responsibilities and migration rules. See `scripts/modeling/README.md` for the script-to-question-to-result index.

Every nontrivial modeling script or tightly coupled script group must be referenced from:

1. `scripts/modeling/README.md`;
2. the relevant protocol or experiment note;
3. the durable result directory, when durable results exist.

The documentation must identify the scientific question, script or `src/` implementation, inputs, outputs, evaluation safeguards, result status, and whether the work is active, diagnostic, retired, or archived.

Do not add one-off maintenance scripts to the repository. Repository cleanup should be performed with `git mv` and committed together with reference repairs and documentation updates.
