# QPITOME QRC Volatility

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/friedsam/qpitome-qrc-volatility.git)

Phase 3 Global Industry Challenge submission for qBraid / MITRE / JonesTrading, Track A: Financial Volatility Prediction.

## Submission identity

- **Team Name:** Qpitome
- **Team Member:** Claudia Friedsam
- **Project Title:** QPITOME QRC Volatility
- **Challenge:** qBraid, MITRE & JonesTrading — Quantum Reservoir Computing for Time-Series Intelligence
- **Challenge Track:** Track A — Financial Volatility Prediction

## Objective

Forecast ten-day log-volatility paths from ordered forty-session market histories, with particular attention to calm-to-crisis transitions. The submission combines a leakage-controlled financial benchmark, an exact six-atom Rydberg quantum reservoir, previously completed Aquila hardware evidence, and Phase-3 studies of MNIST transfer, finite-shot behavior, noise, and reservoir scaling.

## Main contributions

- Public global-index OHLC data with a checksum-verified fallback and explicit provenance.
- Eight chronologically purged, fold-locally rematched train/validation/test folds; the reserved test partition is not opened by the submitted development workflows.
- Frozen classical comparison: persistence, HAR, sequence ridge, Student-t GARCH(1,1), tuned ESN, and an identically specified shuffled-history ESN control.
- Canonical Case151 QRC: six-atom staggered Rydberg ladder, A/4 → B/2 → A/4 palindrome, three probe times, and 63 occupation/pair observables.
- Fold-specific chronological selection of an intercept-free HAR-residual Ridge readout.
- Retrieval-only packaging of three existing Aquila jobs; no new hardware job is submitted by the repository.
- Phase-3 benchmark families: MNIST palindrome classification, density-matrix noise, 5–12-atom exact scaling with 5–20-atom resource reporting, and finite-shot direction preservation through 20,000 shots.

## Headline frozen results

The canonical Case151 development reproduction records:

| Scope | Model | QLIKE | RMSE |
|---|---:|---:|---:|
| Pooled path | HAR | 0.784580 | 0.519081 |
| Pooled path | QRC | 0.776660 | 0.517044 |
| Transition | HAR | 1.118613 | 0.512641 |
| Transition | QRC | 1.105739 | 0.508642 |

The selected fold-8 Case151 readout uses Ridge alpha `0.1` and correction lambda `0.25`. The three previously completed Aquila jobs reproduce 63 hardware observables with pooled simulator–hardware Pearson correlation `0.943073`, RMSE `0.043933`, and through-origin attenuation `1.068182` under the adapted hardware-native schedule.

These are development/frozen-oracle results, not claims from the untouched final test partition. The Aquila evidence validates observable transfer; it is not an end-to-end hardware volatility forecast.

## Judge quick start: qBraid Agent Mode

The standards-compliant Agent Skill is:

```text
qbraid_skill/qpitome-qrc-volatility/SKILL.md
```

Open the repository in qBraid Lab, enable **Agent Mode**, and provide this single prompt:

```text
Reproduce and audit this submission using the complete default full-smoke scope. First locate */qbraid_skill/qpitome-qrc-volatility/SKILL.md under /home/jovyan and read it by absolute path. Resolve every relative path in that skill against the directory containing SKILL.md. Establish the repository root and use it as the working directory. Follow the skill exactly, create and manage the environment yourself, and use its single state-free orchestration command so no shell variables or cross-action terminal state are required. Run verified data, frozen classical baselines, canonical Case151 QRC, and the bounded MNIST, noise, scaling, and finite-shot smoke studies, followed by the read-only validate-existing pass. Do not modify scientific contracts or open the reserved financial test partition. Do not ask me to run terminal commands. Do not submit, query, select, retrieve, or package hardware jobs. Stop and report the first blocking defect rather than improvising.
```

The Agent executes one Python orchestration entry point. It generates the run ID internally, computes every path internally, runs all stages under one aggregate folder, and writes `agent_scope_manifest.json`. It does not depend on `RUN_ID`, `RUN_DIR`, `cd`, environment activation, or exported variables persisting between Agent actions.

Hardware evidence remains a separate optional retrieval-only task and is never required by the Agent workflow.

## Setup and environment

Requirements:

- Python `>=3.10`;
- a qBraid Lab CPU environment;
- repository-local `.venv` created by the supplied bootstrap;
- declared runtime packages: `arch`, `kaggle`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, and `scipy`;
- `pytest` for validation.

No private data credentials are required for the verified fallback workflow. Anonymous live acquisition is used only when preflight confirms it is available and matches the frozen source contract.

From the repository root, run:

```bash
python3 qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py --json

.venv/bin/python -m compileall -q src scripts qbraid_skill/qpitome-qrc-volatility/scripts

.venv/bin/python -m pytest -q \
  tests/qbraid_skill/test_skill_contract.py \
  tests/runs/test_run_submission.py \
  tests/runs/test_run_submission_classical.py \
  tests/runs/test_run_submission_benchmarks.py \
  tests/transition_forecasting/modeling/classical_benchmarks \
  tests/transition_forecasting/qrc/test_case151_reproduction_contract.py \
  tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py \
  tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py \
  tests/transition_forecasting/qrc/test_palindrome_scaling_assay.py \
  tests/transition_forecasting/qrc/test_palindrome_shot_assay.py
```

The bootstrap is the preferred setup path. It creates or reuses `.venv`, installs the project with test dependencies, runs both preflights, and runs focused data, classical, QRC, benchmark, orchestration, and hardware-safety contract checks.

## Step-by-step qBraid execution

### 1. Resolve the verified source mode

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight_classical.py --json
```

Use the exact `auto`, `fallback`, or `live` mode reported by preflight.

### 2. Run the complete submission workflow with one command

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope full-smoke \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

This single command invokes the existing canonical stages:

```text
scripts/runs/run_submission_stage_layout.py financial-classical
scripts/reproduction/run_case151_simulation.py
scripts/runs/run_submission_benchmarks.py all --profile smoke
scripts/runs/run_submission_benchmarks.py validate-existing --profile smoke
```

It fails closed after the first unsuccessful command or acceptance check. No shell variables are required. On success it prints the exact run ID and aggregate directory.

For an explicitly requested core-only run that stops after Case151 verification:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope core \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

A previously accepted `financial-classical` aggregate may be resumed with its exact literal run ID:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/run_submission_scope.py \
  --scope full-smoke \
  --transition-source-mode <MODE_FROM_PREFLIGHT> \
  --run-id <EXACT_RUN_ID> \
  --resume-existing
```

### 3. Existing Aquila hardware evidence — separate and optional

Hardware collection is retrieval-only and outside the Agent workflow. It queries three fixed completed job IDs and never submits a new task:

```bash
.venv/bin/python scripts/hardware/aquila_case151_collect.py collect \
  --outdir results/runs/<RUN_ID>/files/qrc/hardware/aquila_case151_existing_jobs_001
```

The hardware package contains provenance, job identities, device/mapping information, raw observables, metrics, runtime, artifact hashes, and an explicit statement that end-to-end hardware predictions are not applicable.

## Expected inputs and outputs

### Inputs

- Global-index OHLC source files acquired anonymously when available, or restored from the committed checksum-verified fallback.
- Frozen data, fold, model, and QRC contracts under `config/`.
- Immutable Case151 reference files under `reference/case151/freeze_001/`.
- Anonymous MNIST acquisition for the bounded MNIST benchmark.
- Three fixed, previously completed Aquila job IDs for optional retrieval-only hardware packaging.

### Outputs

Every aggregate execution uses one literal run ID and writes:

```text
results/runs/<RUN_ID>/
├── run_manifest.json
├── agent_scope_manifest.json
├── logs/
└── files/
    ├── data/
    ├── classical_baselines/
    ├── qrc/
    │   ├── simulation/run/<RUN_ID>/
    │   └── hardware/<HARDWARE_RUN_ID>/
    ├── quantum_studies/
    │   ├── benchmark_manifest.json
    │   ├── benchmark_artifact_inventory.json
    │   ├── noise/run/<RUN_ID>/
    │   ├── scaling/run/<RUN_ID>/
    │   └── shots/run/<RUN_ID>/
    └── mnist/
        ├── raw/
        ├── shards/
        └── run/<RUN_ID>/
```

Outputs include provenance records, parameters, predictions, metrics, coverage tables, validation reports, plots, command logs, runtime records, and SHA-256 artifact inventories. No source code is copied into generated result directories.

## Known limitations and assumptions

- Model selection and reporting use validation folds only; the reserved financial test partition remains unopened.
- Case151 is a frozen development example and hardware narrative, not a representative test-set claim.
- The six-mode density/curvature readout used by the Phase-3 studies is distinct from the canonical 63-feature Case151 hardware-story model.
- Exact simulation scaling is reported only through 12 atoms; larger systems receive resource estimates rather than infeasible statevector execution.
- Hardware evidence uses an adapted Aquila-native schedule and validates observable transfer, not an end-to-end hardware forecast.
- The Agent Skill does not submit, query, select, retrieve, or require hardware execution.
- Full-smoke is bounded integration coverage; full primary MNIST, noise, scaling, and shot studies are more computationally expensive.
- Generated aggregate runs can be large and are intentionally excluded from ordinary Git history; compact verified evidence packages may be committed separately.

## Key contracts

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
config/case151/expected_metrics.json
config/case151/agent_run_spec.json
docs/transition_forecasting/qrc/case151_reproduction_contract.md
docs/transition_forecasting/qrc/phase3_findings_writeup.md
qbraid_skill/qpitome-qrc-volatility/references/run-contract.md
```
