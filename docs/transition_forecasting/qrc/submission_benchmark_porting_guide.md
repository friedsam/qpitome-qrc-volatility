# Phase 3 benchmark selective-port guide

## Purpose

This guide tells the submission maintainer exactly which files from `qrc-freeze-comparison-work` are needed to reproduce the final MNIST, noise, reservoir-size scaling, and finite-shot studies on the clean submission branch.

Do **not** merge the development branch wholesale. Port the files below selectively, preserve the clean branch's newer data and qBraid Skill infrastructure, run the listed contract tests, and only then wire the benchmark orchestrator into `scripts/runs/run_submission.py`.

## Scientific contracts that must not change during the port

- Financial development panel: fold 5, lead 5, 40-step sequences, at most 12 rows per class in train and validation; no test rows.
- Input representation: `level_instability`, with robust channel scaling fit on training rows only.
- Reservoir: six-atom staggered ladder for the final model, interaction scale 1.25, `0.02` microseconds per observation, probes at `0.25`, `0.5`, and `1.0`.
- Encoding: symmetric `A/4 -> B/2 -> A/4` palindrome with equal amplitude and detuning exposure per channel.
- Financial readout: `six_mode_density_curvature`, `StandardScaler`, `Ridge(alpha=100, fit_intercept=False)`, correction lambda 1.0.
- Noise and finite-shot studies: fit the exact clean readout once and freeze it across perturbations.
- Scaling study: keep the classical readout width fixed at six while atom count changes.
- MNIST primary study: 4x4 average pooling, intensity plus local orthogonal-neighbor contrast, 16 two-channel steps, `occupation_pair_raw` readout, 2,000 train and 1,000 official test examples.
- Use literal run IDs. Never locate the newest result directory.

## 1. Shared files that should already exist on the clean branch

Verify these files before copying benchmark-specific code. Prefer the clean branch's validated version when it is API-compatible.

```text
src/experiments/runs.py
src/transition_forecasting/modeling/stage_e_classical_baselines.py
src/transition_forecasting/qrc/representation_screen_analysis.py
src/transition_forecasting/qrc/rydberg_representation_screen.py
src/transition_forecasting/qrc/temporal_rydberg_chain_experiment.py
```

The financial benchmark modules import the dataset loader, HAR baseline, target columns, fold-row selector, and prequential residual builder from this existing stack. Do not replace these blindly with older development-branch copies.

## 2. Common palindrome QRC dependency closure

Port or verify these shared QRC files before any benchmark family:

```text
src/transition_forecasting/qrc/temporal_rydberg_chain.py
src/transition_forecasting/qrc/temporal_rydberg_ladder.py
src/transition_forecasting/qrc/bivariate_capacity_dynamics.py
src/transition_forecasting/qrc/bivariate_crossover_assay.py
src/transition_forecasting/qrc/palindrome_real_task_relevance_assay.py
src/transition_forecasting/qrc/representation_candidates.py
src/transition_forecasting/qrc/ladder_mode_readout_tools.py
src/transition_forecasting/qrc/ladder_finite_shot_sampling.py
```

Important APIs used by the final benchmarks include:

```text
TemporalRydbergChainConfig
StaggeredLadderGeometryConfig
_evolve_segment_batch
_branch_drive
build_crossover_feature_banks
_resolve_schedule
evolve_palindrome_probabilities
_mean_qlike
_rmse
validate_har_contract
build_candidate_sequences
fit_channel_scaler
transform_candidate_sequences
sample_probe_probabilities
```

After porting, the six-atom exact features must remain numerically identical to the frozen reference. Noise, scaling, and shot smoke runs all contain a six-atom parity gate.

## 3. MNIST acquisition and primary benchmark

### Required files

```text
src/transition_forecasting/data/mnist_acquisition.py
scripts/transition_forecasting/data/acquire_mnist.py
tests/transition_forecasting/data/test_mnist_acquisition.py

src/transition_forecasting/qrc/mnist_palindrome_benchmark.py
scripts/transition_forecasting/qrc/run_mnist_palindrome_benchmark.py
tests/transition_forecasting/qrc/test_mnist_palindrome_benchmark.py
```

### Existing acquisition dependency

```text
src/transition_forecasting/data/acquisition.py
```

Reuse the clean branch's version if it still provides:

```text
file_inventory
install_candidate
utc_now
verify_fallback_manifest
```

### Data source contract

```text
URL: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
SHA-256: 731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1
```

The acquisition code must validate the canonical 60,000 training and 10,000 test arrays and must not require TensorFlow, Keras, Kaggle credentials, or an unverified fallback.

### Optional, not required for the primary submission

```text
scripts/transition_forecasting/data/write_mnist_fallback_manifest.py
src/transition_forecasting/qrc/mnist_position_benchmark.py
src/transition_forecasting/qrc/position_encoded_rydberg.py
scripts/transition_forecasting/qrc/run_mnist_position_benchmark.py
tests/transition_forecasting/qrc/test_mnist_position_benchmark.py
```

The position-encoded study is diagnostic because input dimension changes with atom count. It must not replace the primary palindrome MNIST benchmark.

## 4. Noise benchmark

### Required files

```text
src/transition_forecasting/qrc/temporal_rydberg_noise.py
src/transition_forecasting/qrc/palindrome_rydberg_noise.py
src/transition_forecasting/qrc/palindrome_noise_assay.py
scripts/transition_forecasting/qrc/run_palindrome_noise_assay.py
tests/transition_forecasting/qrc/test_temporal_rydberg_noise.py
tests/transition_forecasting/qrc/test_palindrome_rydberg_noise.py
```

### Required result contract

```text
results/transition_forecasting/qrc/palindrome_noise_assay/<RUN_ID>/
    params.json
    summary.json
    noise_metrics.csv
    predictions.csv.gz
    retained_samples.csv
    frozen_readout.npz
    features/
    plots/forecast_metric_change_vs_ideal.png
    plots/feature_distortion_by_noise.png
    plots/feature_correlation_by_noise.png
```

The density-matrix ideal scenario must match the statevector features within `1e-8`; the verified run matched at approximately `2.7e-15` maximum absolute feature error.

## 5. Reservoir-size scaling benchmark

### Required files

```text
src/transition_forecasting/qrc/palindrome_scaling_assay.py
scripts/transition_forecasting/qrc/run_palindrome_scaling_assay.py
tests/transition_forecasting/qrc/test_palindrome_scaling_assay.py
```

### Required result contract

```text
results/transition_forecasting/qrc/palindrome_scaling_assay/<RUN_ID>/
    params.json
    summary.json
    scaling_metrics.csv
    resource_scaling.csv
    predictions.csv.gz
    retained_samples.csv
    features/
    plots/validation_qlike_vs_atoms.png
    plots/runtime_scaling.png
    plots/statevector_memory_scaling.png
```

Keep measured exact performance rows for 5-12 atoms separate from analytical memory and empirical runtime projections through 20 atoms. Do not extrapolate forecast metrics.

## 6. Finite-shot benchmark

### Required files

```text
src/transition_forecasting/qrc/palindrome_shot_assay.py
scripts/transition_forecasting/qrc/run_palindrome_shot_assay.py
tests/transition_forecasting/qrc/test_palindrome_shot_assay.py
```

`src/transition_forecasting/qrc/ladder_finite_shot_sampling.py` is part of the shared dependency closure and must include stable sample/probe seeding and row-order-invariant multinomial sampling.

### Required result contract

```text
results/transition_forecasting/qrc/palindrome_shot_assay/<RUN_ID>/
    params.json
    summary.json
    shot_metrics.csv
    shot_summary.csv
    direction_metrics.csv
    predictions.csv.gz
    retained_samples.csv
    exact_reference.npz
    plots/warning_gap_preservation_vs_shots.png
    plots/correction_correlation_vs_shots.png
    plots/transition_control_gap_vs_shots.png
```

The transition/control and horizon-specific direction tables are primary. Pooled QLIKE and RMSE are secondary diagnostics.

## 7. Submission orchestration and documentation

Port these new integration files:

```text
scripts/runs/run_submission_benchmarks.py
tests/runs/test_run_submission_benchmarks.py
docs/transition_forecasting/qrc/phase3_findings_writeup.md
docs/transition_forecasting/qrc/submission_benchmark_porting_guide.md
docs/transition_forecasting/qrc/submission_benchmark_port_manifest.md
```

The standalone benchmark runner exists to avoid overwriting an actively maintained `scripts/runs/run_submission.py` on another branch. It supports:

```text
mnist-palindrome
palindrome-noise
palindrome-scaling
palindrome-shots
all
validate-existing
```

It records command logs, repository/environment identity, required-output validation, and recursive SHA-256 inventories. It uses explicit benchmark run IDs and does not scan by modification time.

### Recommended clean integration into `scripts/runs/run_submission.py`

Add a workflow such as `phase3-benchmarks` whose command plan invokes the helper once:

```python
(
    sys.executable,
    "scripts/runs/run_submission_benchmarks.py",
    "all",
    "--profile",
    "primary",
    "--run-id",
    "phase3_benchmarks_primary_001",
    "--mnist-run-id",
    "mnist_palindrome_primary_001",
    "--noise-run-id",
    "palindrome_noise_primary_001",
    "--scaling-run-id",
    "palindrome_scaling_primary_001",
    "--shots-run-id",
    "palindrome_shots_primary_001",
    "--resume",
    "--reuse-existing",
)
```

The main submission runner only needs to require:

```text
results/runs/phase3_benchmarks_primary_001/benchmark_manifest.json
results/runs/phase3_benchmarks_primary_001/benchmark_artifact_inventory.json
```

The helper validates the complete per-benchmark output closure.

## 8. qBraid Skill files that must be updated on the clean branch

Do not copy an older Skill wholesale. Update the current clean-branch files in place:

```text
qbraid_skill/qpitome-qrc-volatility/SKILL.md
qbraid_skill/qpitome-qrc-volatility/references/repository-map.md
qbraid_skill/qpitome-qrc-volatility/references/run-contract.md
qbraid_skill/qpitome-qrc-volatility/scripts/preflight.py
qbraid_skill/qpitome-qrc-volatility/scripts/bootstrap.py
tests/qbraid_skill/test_skill_contract.py
README.md
```

Required changes:

1. Replace the statement that only Stage 1 is executable.
2. Add the four benchmark workflows and their exact run IDs.
3. Document MNIST's anonymous checksum-pinned source.
4. Add preflight checks for all benchmark runners and required shared modules.
5. Add focused tests for MNIST acquisition, MNIST palindrome, density/noise, scaling, shots, and both submission runners.
6. Preserve the prohibitions against changing scientific defaults, opening reserved test rows, selecting newest directories, or submitting new hardware jobs during judge reproduction.
7. State clearly that the Aquila result is observable transfer under an adapted schedule, not an end-to-end hardware forecast.

## 9. Focused validation after the port

Run from the clean branch after `python -m pip install -e ".[test]"`:

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
  tests/runs/test_run_submission.py \
  tests/runs/test_run_submission_benchmarks.py \
  tests/qbraid_skill/test_skill_contract.py
```

Then run the bounded smoke profile:

```bash
python scripts/runs/run_submission_benchmarks.py all \
  --profile smoke \
  --run-id phase3_benchmarks_smoke_001
```

Finally validate the preserved primary result directories without rerunning them:

```bash
python scripts/runs/run_submission_benchmarks.py validate-existing \
  --profile primary \
  --run-id phase3_benchmarks_validation_001 \
  --mnist-run-id mnist_palindrome_primary_001 \
  --noise-run-id palindrome_noise_primary_001 \
  --scaling-run-id palindrome_scaling_primary_001 \
  --shots-run-id palindrome_shots_primary_001
```

## 10. Files and claims that must not be silently substituted

- Do not replace the primary six-atom MNIST result with the optional position-encoded sweep.
- Do not call the local Markovian noise grid a calibrated Aquila device model.
- Do not report scaling performance above 12 atoms; only resources are projected through 20.
- Do not report 1,000 shots as a stable horizon-resolved threshold. The verified aggregate threshold is 10,000 shots, and 20,000 is preferred for the horizon profile.
- Do not describe the Aquila scatter as a hardware volatility forecast.
- Do not claim interaction-specific quantum advantage; the matched interaction-off control performed better.
- Do not open or evaluate the reserved test partition during integration.
