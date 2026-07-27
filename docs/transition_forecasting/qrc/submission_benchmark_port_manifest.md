# Phase 3 benchmark selective-port manifest

## Scope and authority

This manifest records the selective migration of the required Phase 3 benchmark studies from `qrc-freeze-comparison-work` into the submission integration branch. The development branch must not be merged wholesale.

Source handoff:

```text
branch: qrc-freeze-comparison-work
handoff head: 19f279ff244c27f06b9df666c72862a514d674aa
submission branch: agent/phase3-benchmark-migration
```

The final result hierarchy in this document supersedes the legacy development result paths. Historical run IDs remain valid provenance labels, but judge-facing executions use the aggregate submission `<RUN_ID>` by default.

## Ported benchmark families

### MNIST

```text
src/transition_forecasting/data/mnist_acquisition.py
scripts/transition_forecasting/data/acquire_mnist.py
tests/transition_forecasting/data/test_mnist_acquisition.py
src/transition_forecasting/qrc/mnist_palindrome_benchmark.py
scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py
tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py
```

Frozen primary contract:

- anonymous checksum-pinned MNIST source;
- canonical 60,000 training and 10,000 official test arrays;
- deterministic balanced subsets of 2,000 training and 1,000 test examples;
- 4×4 average pooling;
- pooled intensity plus local orthogonal-neighbour contrast;
- 16 two-channel steps;
- six-atom A/4→B/2→A/4 palindrome;
- `occupation_pair_raw`, 63 features;
- multinomial logistic readout;
- resumable deterministic shards.

The optional position-encoded MNIST study is not part of the required primary workflow.

### Noise

```text
src/transition_forecasting/qrc/temporal_rydberg_noise.py
src/transition_forecasting/qrc/temporal_rydberg_noise_assay.py
src/transition_forecasting/qrc/palindrome_rydberg_noise.py
src/transition_forecasting/qrc/palindrome_noise_assay.py
scripts/transition_forecasting/qrc/run_palindrome_noise_assay.py
tests/transition_forecasting/qrc/test_temporal_rydberg_noise.py
tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py
```

`temporal_rydberg_noise_assay.py` was missing from the handoff guide but is required by the retained noise contract tests. It was added during migration rather than suppressing the test.

Frozen primary contract:

- development-only fold 5, lead 5, at most 12 rows per class in train and validation;
- no test rows;
- `level_instability` with train-only robust scaling;
- six-atom palindrome and `six_mode_density_curvature` readout;
- Ridge alpha 100, no intercept, correction lambda 1;
- ideal readout fitted once and frozen across T1, T2, depolarizing, and combined scenarios;
- ideal density/statevector parity required before accepting noisy results.

This is a local Markovian robustness study, not a calibrated Aquila noise model.

### Reservoir-size scaling

```text
src/transition_forecasting/qrc/palindrome_scaling_assay.py
scripts/transition_forecasting/qrc/run_palindrome_scaling_assay.py
tests/transition_forecasting/qrc/test_palindrome_scaling_assay.py
```

Frozen primary contract:

- same fold-5/lead-5 development panel and six-feature readout;
- exact performance for 5–12 atoms only;
- analytical memory and empirical runtime resource reporting through 20 atoms;
- no extrapolated forecast metrics above 12 atoms;
- readout width remains fixed at six while atom count changes.

### Finite shots

```text
src/transition_forecasting/qrc/palindrome_shot_assay.py
scripts/transition_forecasting/qrc/run_palindrome_shot_assay.py
tests/transition_forecasting/qrc/test_palindrome_shot_assay.py
```

Frozen primary contract:

- same fold-5/lead-5 development panel;
- exact probabilities, scaler, and no-intercept Ridge head generated once and frozen;
- shot grid: 100, 250, 500, 1,000, 2,000, 5,000, 10,000, and 20,000;
- ten deterministic measurement seeds;
- transition/control and horizon-specific direction preservation are primary;
- pooled QLIKE and RMSE are secondary diagnostics.

The verified aggregate direction threshold is 10,000 shots; 20,000 is preferred for horizon-resolved interpretation. One thousand shots must not be presented as a stable horizon-resolved threshold.

## Shared QRC closure

These shared files are provided by the Case151 migration or the clean submission branch and must not be silently replaced with older incompatible copies:

```text
src/experiments/runs.py
src/transition_forecasting/modeling/fold_selection.py
src/transition_forecasting/modeling/stage_e_classical_baselines.py
src/transition_forecasting/modeling/stage_e_sequence_models.py
src/transition_forecasting/qrc/temporal_rydberg_chain.py
src/transition_forecasting/qrc/temporal_rydberg_ladder.py
src/transition_forecasting/qrc/bivariate_capacity_dynamics.py
src/transition_forecasting/qrc/bivariate_crossover_assay.py
src/transition_forecasting/qrc/palindrome_real_task_relevance_assay.py
src/transition_forecasting/qrc/representation_candidates.py
src/transition_forecasting/qrc/representation_screen_analysis.py
src/transition_forecasting/qrc/rydberg_representation_screen.py
src/transition_forecasting/qrc/temporal_rydberg_chain_experiment.py
src/transition_forecasting/qrc/ladder_mode_readout_tools.py
src/transition_forecasting/qrc/ladder_finite_shot_sampling.py
```

The six-mode benchmark studies remain scientifically distinct from the 63-feature Case151 hardware-story model.

## Orchestration

Ported integration files:

```text
scripts/runs/run_submission_benchmarks.py
tests/runs/test_run_submission_benchmarks.py
docs/transition_forecasting/qrc/phase3_findings_writeup.md
docs/transition_forecasting/qrc/submission_benchmark_porting_guide.md
docs/transition_forecasting/qrc/submission_benchmark_port_manifest.md
```

The benchmark helper supports:

```text
mnist-palindrome
palindrome-noise
palindrome-scaling
palindrome-shots
all
validate-existing
```

The helper:

- accepts an existing aggregate `--run-dir`;
- uses explicit literal run IDs;
- never discovers the newest result directory;
- records commands, logs, repository/environment identity, required-output checks, and recursive SHA-256 inventories;
- does not create or replace the aggregate `run_manifest.json` owned by `scripts/runs/run_submission.py`.

Recommended aggregate invocation:

```bash
python scripts/runs/run_submission_benchmarks.py all \
  --profile primary \
  --run-id <RUN_ID> \
  --run-dir results/runs/<RUN_ID> \
  --resume \
  --reuse-existing
```

## Final result contract

```text
results/runs/<RUN_ID>/
├── run_manifest.json                         # main submission runner
├── logs/
│   └── benchmarks/
└── files/
    ├── mnist/
    │   ├── raw/
    │   ├── shards/
    │   └── run/<RUN_ID>/
    └── quantum_studies/
        ├── noise/run/<RUN_ID>/
        ├── scaling/run/<RUN_ID>/
        ├── shots/run/<RUN_ID>/
        ├── benchmark_manifest.json
        └── benchmark_artifact_inventory.json
```

Required family outputs are validated recursively by the helper. Preserved historical result packages may be validated with explicit per-family run IDs, but no historical result is selected by modification time.

## Maintainer-owned work intentionally excluded

The selective migration does not modify:

```text
scripts/runs/run_submission.py
qbraid_skill/qpitome-qrc-volatility/SKILL.md
qbraid_skill/qpitome-qrc-volatility/references/repository-map.md
qbraid_skill/qpitome-qrc-volatility/references/run-contract.md
qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py
qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py
tests/qbraid_skill/test_skill_contract.py
README.md
```

The final repository maintainer must wire one aggregate benchmark command into `run_submission.py` and update the current qBraid Skill and README in place. An older Skill must not be copied wholesale.

## Claims that must remain bounded

- Do not replace the primary six-atom MNIST result with the optional position-encoded sweep.
- Do not describe the noise grid as calibrated Aquila noise.
- Do not report scaling performance above 12 atoms.
- Do not present 1,000 shots as a stable horizon-resolved threshold.
- Do not describe Aquila observable transfer as an end-to-end hardware volatility forecast.
- Do not claim interaction-specific quantum advantage; the matched interaction-off control performed better.
- Do not open or evaluate the reserved financial test partition.

## Validation gates

```bash
python -m compileall -q \
  src/transition_forecasting/qrc \
  scripts/transition_forecasting/qrc \
  scripts/transition_forecasting/data \
  scripts/runs/run_submission_benchmarks.py

python -m pytest -q --tb=short \
  tests/transition_forecasting/data/test_mnist_acquisition.py \
  tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py \
  tests/transition_forecasting/qrc/test_temporal_rydberg_noise.py \
  tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py \
  tests/transition_forecasting/qrc/test_palindrome_scaling_assay.py \
  tests/transition_forecasting/qrc/test_palindrome_shot_assay.py \
  tests/runs/test_run_submission_benchmarks.py
```

A bounded smoke run and validation of any preserved primary outputs remain execution checks on qBraid; they are not replaced by source-contract tests.
