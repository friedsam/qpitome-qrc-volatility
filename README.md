# QPITOME QRC Volatility

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/friedsam/qpitome-qrc-volatility.git)

Phase 3 Global Industry Challenge project for qBraid / MITRE / JonesTrading, Track A: Financial Volatility Prediction.

## Scientific objective

Forecast ten-day volatility paths from ordered forty-day market histories, emphasizing abrupt calm-to-crisis transitions at leads 1, 5, and 10. Classical HAR forecasts provide the persistence baseline; Rydberg quantum-reservoir features are evaluated as corrections to the HAR residual path.

## Judge quick start: qBraid Agent Mode

The **Launch on qBraid** button clones the repository when it is public. Open the cloned repository in qBraid Lab, connect qBraid AI to the workspace, enable **Agent Mode**, and give the agent this single instruction:

```text
Reproduce and audit this submission. First locate */qbraid_skill/qpitome-qrc-volatility/SKILL.md under /home/jovyan and read it by absolute path. Resolve every relative path in that skill against the directory containing SKILL.md, not against the repository root, and use absolute paths for file reads. Establish the repository root before running commands and use it as the working directory. Follow the skill exactly. Create and manage the required environment yourself. Do not ask me to run terminal commands. Do not modify source code, scientific parameters, data contracts, or the reserved test partition, and do not submit hardware jobs. Stop and report the first blocking defect rather than improvising around it.
```

The agent should independently:

1. locate the skill and repository roots without assuming its initial working directory;
2. record the repository commit, branch, and working-tree state;
3. create the repository-local `.venv`;
4. install dependencies;
5. run preflight and focused contract tests;
6. probe anonymous Kaggle access and verify any committed fallback;
7. use only the source mode reported by preflight;
8. run the canonical submission runner;
9. validate data provenance, manifests, checksums, scientific identity, and result contents;
10. report success, failure, or a precise blocker.

The standards-compliant Agent Skill entry is:

```text
qbraid_skill/qpitome-qrc-volatility/SKILL.md
```

### Current development boundary

This branch currently validates only the Stage-1 transition-data workflow. The final submission must also add the financial QRC, classical comparison, MNIST, qubit-scaling, noise, and final artifact-collection stages to the same automated workflow.

The source policy is already deterministic:

- public Kaggle access is probed anonymously;
- a committed fallback is accepted only after complete manifest/hash/schema verification;
- when both are available, the live candidate is compared with the fallback;
- any missing, extra, or changed live file causes fallback substitution before the pipeline runs;
- without a fallback, live data must match the frozen raw inventory exactly;
- the installed source is verified again and the decision is recorded in `raw_acquisition_manifest.json`.

## Manual diagnostic path

Use this only to diagnose an agent failure, not as the primary judge workflow.

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json

.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --json
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
```

The bootstrap is idempotent. It creates `.venv` if needed, installs the project and test dependencies, runs preflight, and runs the focused contract suite. It does not execute scientific results. Do not use a bare `pip` command or depend on shell activation persisting between commands.

## Current Stage-1 workflow

Generate a literal run ID:

```bash
date -u +qbraid-stage1-%Y%m%dT%H%M%SZ
```

Use the exact source mode reported by preflight. When both anonymous live access and the verified fallback are ready:

```bash
.venv/bin/python scripts/runs/run_submission.py transition-data \
  --run-id <RUN_ID_FROM_PREVIOUS_COMMAND> \
  --transition-source-mode auto
```

Use `fallback` or `live` only when preflight reports that as the sole verified mode. Do not use `--force` for a new run.

The acquisition manifest must record:

```text
authoritative_source_verified: true
installed_source_comparison.matched: true
candidate_comparison
fallback_substitution
substitution_reason
```

The aggregate judge-facing run is written under:

```text
results/runs/<run-id>/
```

`scripts/runs/run_submission.py` is the sole planned exception to the normal scientific-result mapping because it assembles a structured judge-facing package. Normal scientific outputs follow:

```text
src/<domain>/<experiment>/*.py
scripts/<domain>/<experiment>/*.py
tests/<domain>/<experiment>/test_*.py
results/<domain>/<experiment>/run/<run-id>/*
```

Run directories may contain data products, configuration snapshots, manifests, logs, predictions, metrics, figures, and checksums. They must not contain copied source code.

## Current canonical Stage-1 scope

1. Acquire or restore verified global-index OHLC inputs.
2. Apply the frozen structural-quality policy.
3. Build one-channel forty-session log-volatility sequences.
4. Construct the binary pre-control candidate pool.
5. Create eight chronologically purged walk-forward folds.
6. Rematch controls within each fold partition.
7. Validate and checksum generated artifacts.

The former derived three-channel representation is not part of the canonical pipeline. The fixed final test partition must remain unopened during development and model selection.

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
