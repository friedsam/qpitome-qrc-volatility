# QPITOME QRC Volatility

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/friedsam/qpitome-qrc-volatility.git)

Phase 3 Global Industry Challenge project for qBraid / MITRE / JonesTrading, Track A: Financial Volatility Prediction.

## Scientific objective

Forecast ten-day log-volatility paths from ordered forty-day market histories, emphasizing abrupt calm-to-crisis transitions at leads 1, 5, and 10. The classical comparison contains persistence, HAR, sequence ridge, Student-t GARCH(1,1), a tuned direct ESN, and an identically specified shuffled-order ESN control. Quantum-reservoir stages are integrated separately.

## Judge quick start: qBraid Agent Mode

The **Launch on qBraid** button clones the repository when it is public. Open the cloned repository in qBraid Lab, connect qBraid AI to the workspace, enable **Agent Mode**, and give the agent this single instruction:

```text
Reproduce and audit this submission. First locate */qbraid_skill/qpitome-qrc-volatility/SKILL.md under /home/jovyan and read it by absolute path. Resolve every relative path in that skill against the directory containing SKILL.md, not against the repository root, and use absolute paths for file reads. Establish the repository root before running commands and use it as the working directory. Follow the skill exactly. Create and manage the required environment yourself. Do not ask me to run terminal commands. Do not modify source code, scientific parameters, data contracts, or the reserved test partition, and do not submit hardware jobs. Stop and report the first blocking defect rather than improvising around it.
```

The agent independently:

1. locates the skill and repository roots;
2. records commit, branch, and working-tree state;
3. creates or reuses the repository-local `.venv`;
4. installs declared dependencies;
5. runs data and classical preflights plus focused contract tests;
6. probes anonymous Kaggle access and verifies the committed fallback;
7. uses only the source mode reported by preflight;
8. runs the canonical `financial-classical` submission workflow;
9. validates provenance, manifests, checksums, scientific identity, model parameters, and result contents;
10. reports success, failure, or the first precise blocker.

The standards-compliant Agent Skill entry is:

```text
qbraid_skill/qpitome-qrc-volatility/SKILL.md
```

## Current executable boundary

This branch now automates:

- verified transition-data acquisition and reconstruction;
- one-channel forty-session input tensors;
- eight chronologically purged, fold-locally rematched datasets;
- persistence, canonical HAR, and direct sequence ridge;
- zero-mean Student-t GARCH(1,1) with the `arch` backend;
- the frozen tuned direct ESN and shuffled-order control;
- exact-common-row tables for Transition, L1, L5, L10, Controls, and Pooled;
- RMSE, log-volatility QLIKE, and Mincer-Zarnowitz diagnostics;
- aggregate manifests, logs, hashes, coverage, predictions, and validation.

Financial QRC, MNIST, qubit scaling, noise studies, and final cross-topic artifact collection still need to be added to the same submission framework. The current classical workflow does not query or submit hardware jobs. Completed hardware results will be verified and reported from committed artifacts rather than rerun by the default agent workflow.

## Frozen classical specification

The judge workflow does not retune models. Its single parameter authority is:

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
```

Submitted models:

```text
persistence
har
sequence_ridge
garch_1_1_t
esn_direct_tuned
esn_shuffled_tuned
```

## Manual diagnostic path

Use this only to diagnose an agent failure, not as the primary judge workflow.

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json

.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --json
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight_classical.py --json
```

Bootstrap is idempotent. It creates `.venv` when needed, installs the project and test dependencies, runs both preflights, and runs the focused contract suite. It does not execute scientific results. Do not use a bare `pip` command or depend on shell activation persisting between commands.

## Canonical financial-classical workflow

Generate a literal run ID:

```bash
date -u +qbraid-financial-classical-%Y%m%dT%H%M%SZ
```

Use the exact source mode reported by strict preflight:

```bash
.venv/bin/python scripts/runs/run_submission_stage_layout.py financial-classical \
  --run-id <RUN_ID_FROM_PREVIOUS_COMMAND> \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

Do not use `--force` for a new run.

The aggregate judge-facing run is written under:

```text
results/runs/<run-id>/
```

Scientific artifacts are grouped by pipeline stage beneath:

```text
results/runs/<run-id>/files/
```

The planned top-level stage names are:

```text
data/
classical_baselines/
qrc/
quantum_studies/
mnist/
```

Only stages included in the selected workflow are created. Committed hardware results belong beneath `files/qrc/hardware/`; the default agent workflow validates and reports them but does not submit a new hardware job.

The current classical stage is:

```text
files/classical_baselines/
    linear/run/<run-id>/
    garch/run/<run-id>/
    esn/run/<run-id>/
    canonical/run/<run-id>/
    validation/classical_baseline_audit.json
```

The current data stage is:

```text
files/data/
    raw/
    processed/global_transition_dataset_1d/
    validation/
```

`scripts/runs/run_submission_stage_layout.py` is the judge-facing aggregation exception to the normal scientific-result mapping. It delegates scientific execution to the established runner and changes only aggregate path resolution. Ordinary producers follow:

```text
src/<domain>/<experiment>/*.py
scripts/<domain>/<experiment>/*.py
tests/<domain>/<experiment>/test_*.py
results/<domain>/<experiment>/run/<run-id>/*
```

Run directories may contain data products, configuration snapshots, manifests, logs, predictions, metrics, figures, and checksums. They must not contain copied source code.

## Data-source policy

- public Kaggle access is probed anonymously;
- the committed fallback is accepted only after manifest, hash, file-set, and schema verification;
- when both sources are available, the live candidate is compared with the fallback;
- any missing, extra, or changed live file causes fallback substitution before installation;
- without a fallback, live data must match the frozen raw inventory exactly;
- the installed source is verified again and the decision is recorded in `raw_acquisition_manifest.json`.

## Local development outside qBraid

```bash
conda env create -f environment.yml
conda activate qrc-volatility
```

For an existing environment:

```bash
conda env update -f environment.yml --prune
```

Only validated components are promoted to `stage1-dev`; only the final validated submission state is intended to be merged into `main`.
