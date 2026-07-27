# QPITOME QRC Volatility

[<img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="150" alt="Launch on qBraid">](https://account.qbraid.com?gitHubUrl=https://github.com/friedsam/qpitome-qrc-volatility.git)

Phase 3 Global Industry Challenge submission for qBraid / MITRE / JonesTrading, Track A: Financial Volatility Prediction.

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

Open the repository in qBraid Lab, enable **Agent Mode**, and provide:

```text
Reproduce and audit this submission. First locate */qbraid_skill/qpitome-qrc-volatility/SKILL.md under /home/jovyan and read it by absolute path. Resolve every relative path in that skill against the directory containing SKILL.md. Establish the repository root and use it as the working directory. Follow the skill exactly, create and manage the environment yourself, and do not modify scientific contracts or open the reserved test partition. Do not ask me to run terminal commands. Do not submit hardware jobs. Stop and report the first blocking defect rather than improvising.
```

The default Skill reproduces and audits the verified data and frozen classical comparison. The additional QRC, benchmark, and hardware-evidence stages are explicit below so judges can run only the desired scope.

## Environment and contract tests

From the repository root:

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

## 1. Data and frozen classical comparison

Resolve the verified source mode:

```bash
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py --strict-data-source
.venv/bin/python qbraid_skill/qpitome-qrc-volatility/scripts/preflight_classical.py --json
```

Then run:

```bash
RUN_ID=$(date -u +qbraid-financial-classical-%Y%m%dT%H%M%SZ)

.venv/bin/python scripts/runs/run_submission_stage_layout.py financial-classical \
  --run-id "$RUN_ID" \
  --transition-source-mode <MODE_FROM_PREFLIGHT>
```

A verified qBraid execution completed successfully as:

```text
qbraid-financial-classical-20260727T025306Z
```

The complete generated run remains local under `results/runs/` because bulky run artifacts are intentionally ignored by Git.

## 2. Canonical Case151 QRC simulation

Using the fold tensors produced by the aggregate run:

```bash
RUN_DIR="results/runs/$RUN_ID"

.venv/bin/python scripts/reproduction/run_case151_simulation.py \
  --fold-dir "$RUN_DIR/files/data/processed/global_transition_dataset_1d/purged_walk_forward_folds" \
  --output-root "$RUN_DIR/files/qrc/simulation/run" \
  --run-id "$RUN_ID"
```

The runner fails closed unless the frozen metrics, 63-feature width, fold-8 alpha/lambda identity, and zero-test-row contract are reproduced.

## 3. Phase-3 benchmark studies

A bounded integration smoke run uses the same aggregate folder:

```bash
.venv/bin/python scripts/runs/run_submission_benchmarks.py \
  all \
  --profile smoke \
  --run-id "$RUN_ID" \
  --run-dir "$RUN_DIR" \
  --resume \
  --reuse-existing
```

For the full frozen benchmark sizes, replace `--profile smoke` with `--profile primary`. MNIST uses resumable shards; the primary profile uses 2,000 training and 1,000 official test images.

## 4. Existing Aquila hardware evidence

Hardware collection is optional and retrieval-only. It queries the three fixed completed job IDs and never submits a new task:

```bash
.venv/bin/python scripts/hardware/aquila_case151_collect.py collect \
  --outdir "$RUN_DIR/files/qrc/hardware/aquila_case151_existing_jobs_001"
```

The hardware package contains provenance, job identities, device/mapping information, raw observables, metrics, runtime, artifact hashes, and an explicit statement that end-to-end hardware predictions are not applicable.

## Judge-facing result layout

```text
results/runs/<RUN_ID>/
├── run_manifest.json
├── logs/
└── files/
    ├── data/
    ├── classical_baselines/
    ├── qrc/
    │   ├── simulation/run/<RUN_ID>/
    │   └── hardware/<HARDWARE_RUN_ID>/
    ├── quantum_studies/
    │   ├── noise/run/<RUN_ID>/
    │   ├── scaling/run/<RUN_ID>/
    │   └── shots/run/<RUN_ID>/
    └── mnist/
        ├── raw/
        ├── shards/
        └── run/<RUN_ID>/
```

No source code is copied into generated result directories.

## Scientific boundaries and limitations

- Model selection and reporting use validation folds only; the reserved test partition remains unopened.
- Case151 is a frozen development example and hardware narrative, not a representative test-set claim.
- The six-mode density/curvature readout used by the Phase-3 studies is distinct from the canonical 63-feature Case151 hardware-story model.
- Exact simulation scaling is reported only through 12 atoms; larger systems receive resource estimates rather than infeasible statevector execution.
- Hardware evidence uses an adapted Aquila-native schedule and validates observable transfer, not an end-to-end hardware forecast.
- The default Agent Skill does not submit or require hardware execution.

## Key contracts

```text
config/transition_forecasting/classical_benchmarks/frozen_submission.json
config/case151/expected_metrics.json
config/case151/agent_run_spec.json
docs/transition_forecasting/qrc/case151_reproduction_contract.md
docs/transition_forecasting/qrc/phase3_findings_writeup.md
qbraid_skill/qpitome-qrc-volatility/references/run-contract.md
```
