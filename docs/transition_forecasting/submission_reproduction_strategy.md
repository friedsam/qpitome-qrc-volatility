# Submission Reproduction Strategy

## Status

This document records the current design proposal for the final automated submission workflow. It is intentionally narrower than the research workflow and should be updated when the final paper claims, required artifacts, and challenge execution behavior are frozen.

The challenge materials do not clearly define the resources, time budget, credentials, or exact behavior of the automated evaluator. The design therefore favors a conservative default: one unattended command that reproduces the central paper evidence without requiring manual decisions, hardware access, or long exploratory sweeps.

## Objectives

The final submission workflow should satisfy five goals:

1. An automated evaluator can run one obvious command without selecting profiles or understanding repository history.
2. The command reproduces the central results used in the five-page write-up.
3. Judge-facing outputs are collected in one shallow, self-explanatory directory.
4. Expensive supporting studies remain reproducible through explicit standalone scripts.
5. Hardware jobs are never submitted implicitly by the default workflow.

This design separates submission reproducibility from research organization. Research code and results may remain deeply organized by experiment, while the submission run exposes only the small set of artifacts needed to evaluate the final claims.

## Default automated entry point

The intended default command is:

```bash
python scripts/runs/run_submission.py
```

The default invocation should require no flags, prompts, credentials, or interactive choices. It should terminate with a nonzero exit code when a paper-critical stage fails or a required artifact is missing.

The script should print a short, unambiguous completion message containing:

- overall success or failure;
- the final submission run directory;
- the primary summary file;
- the artifact index;
- any optional stages that were skipped.

Example:

```text
SUBMISSION RUN SUCCEEDED
Summary: results/submission/<run_id>/summary.json
Report: results/submission/<run_id>/reproduction_report.md
Artifacts: results/submission/<run_id>/artifact_index.csv
```

## Default workflow scope

The default workflow should reproduce only the evidence required for the central claims in the final write-up, plus inexpensive integrity checks.

Expected scope:

1. Validate or reconstruct the frozen public-data input used by the final pipeline.
2. Validate the frozen folds, controls, sample identifiers, and configuration.
3. Run the principal classical comparison used in the paper.
4. Run the frozen exact-simulator QRC configuration used in the paper.
5. Reproduce the principal forecasting metrics, tables, and figures.
6. Validate and collect any paper-critical precomputed auxiliary artifacts.
7. Write provenance, runtime, environment, command, and checksum records.

The default workflow should not automatically run every experiment developed during the project. A result belongs in the default path only when it directly supports a claim or figure in the final five-page submission.

## Expensive supporting studies

Finite-shot analysis, scaling studies, noise studies, MNIST, and hardware validation may be scientifically important but can be too slow, resource-dependent, or queue-dependent for an unattended default run.

Each expensive result used in the paper should therefore have both:

1. a frozen result artifact that the default workflow validates; and
2. a standalone regeneration script with exact instructions.

The final script names should follow the repository path-matching convention. Illustrative locations are:

```text
scripts/transition_forecasting/qrc/shot_study/run_shot_study.py
scripts/transition_forecasting/qrc/scaling/run_scaling_study.py
scripts/transition_forecasting/qrc/noise/run_noise_study.py
scripts/benchmarks/mnist/run_mnist_qrc.py
scripts/hardware/neutral_atom/run_hardware_validation.py
```

The exact names may change when the corresponding source, test, script, and result paths are frozen.

For every expensive result included in the write-up, the repository should retain:

- the exact producing command;
- source commit SHA;
- configuration and random seed;
- input checksums;
- runtime and machine or backend information;
- raw or minimally processed output;
- artifact checksum;
- regeneration instructions;
- known limitations.

The default workflow should fail if a missing or corrupted precomputed artifact is required for a central claim. It may warn rather than fail for optional supplemental material.

## Hardware policy

The default automated workflow must not submit a fresh hardware job.

Reasons:

- evaluator credentials and access are unknown;
- queue time is nondeterministic;
- hardware availability may change;
- automatic submission may consume limited resources;
- a queued job may prevent the default run from completing.

Instead, the default workflow should:

1. validate frozen raw hardware outputs;
2. record backend, provider, job identifier, timestamp, shot count, and configuration;
3. regenerate the hardware comparison table or figure from those outputs;
4. state clearly that fresh submission is a separate manual action.

A standalone hardware script may support fresh submission when credentials and allocation are available, but it must require explicit invocation and must never be called implicitly by `run_submission.py`.

## Judge-facing output layout

Research results should continue to use the repository's mirrored, human-readable organization:

```text
src/<domain>/<experiment>/*.py
scripts/<domain>/<experiment>/*.py
tests/<domain>/<experiment>/test_*.py
results/<domain>/<experiment>/<run_id>/*
```

The submission workflow has a different audience. An automated evaluator or judge should not need to search many deep result trees. The final run should collect the small set of relevant artifacts into one shallow directory.

Proposed layout:

```text
results/submission/<run_id>/
    README.txt
    run_manifest.json
    summary.json
    reproduction_report.md
    artifact_index.csv
    data/
    classical/
    qrc/
    forecasting/
    supplemental/
    logs/
```

The directory names are provisional. The final taxonomy should remain shallow, suggestive, and limited to categories represented in the paper.

### Top-level files

`README.txt`
: Explains what the run reproduced, where the primary artifacts are, which expensive studies were validated rather than rerun, and how to regenerate them.

`run_manifest.json`
: Records repository commit, branch, environment, commands, parameters, timings, status, and failures.

`summary.json`
: Contains the small set of primary metrics and conclusions used by automated checking or rapid review.

`reproduction_report.md`
: Human-readable account of executed stages, outputs, skipped optional work, and warnings.

`artifact_index.csv`
: Maps each paper table, figure, or reported result to its source producer, collected destination, checksum, and reproduction status.

Suggested columns:

```text
artifact_id,paper_reference,artifact_type,producer_script,source_path,collected_path,status,sha256,notes
```

## Artifact collection

During development, producers should continue writing to their matched experiment result paths. The submission workflow should not force all research producers to share one output directory.

For the final workflow, two implementation approaches are possible:

1. Direct each producer into the submission run directory through an explicit output argument.
2. Let each producer write to its normal run directory, then copy only validated paper-critical artifacts into the submission directory.

Given the project deadline and the risk of destabilizing working producers, the initial implementation should prefer the second approach.

The collection stage must:

- copy rather than move artifacts;
- preserve original research outputs;
- verify checksums after copying;
- record source and destination paths;
- reject missing paper-critical files;
- avoid silently selecting the newest result directory;
- use explicit run identifiers or manifest references.

No historical result should be inferred from a current default path.

## Reproduction classes

Final results should be classified explicitly.

### Class 1: Core results

Central paper claims. These should be regenerated by the default workflow whenever computationally reasonable.

Examples may include:

- principal data integrity checks;
- principal classical baseline;
- frozen exact-simulator QRC forecast;
- primary QLIKE, RMSE, and calibration tables;
- principal transition-path figure.

### Class 2: Expensive simulator results

Supporting claims that are reproducible locally but too expensive for the default path.

Examples may include:

- full finite-shot sweep;
- broad qubit or atom-count scaling;
- broad noise sweep;
- full MNIST experiment.

The default workflow may validate their frozen artifacts and regenerate only the compact subset directly reported in the paper.

### Class 3: Hardware results

Results dependent on credentials, queue state, backend availability, or limited allocation.

The default workflow should validate and analyze frozen raw outputs. Fresh submission remains separately reproducible where access permits.

## README requirements

The repository README or a dedicated submission README should distinguish clearly between:

### Default automated reproduction

```bash
python scripts/runs/run_submission.py
```

This is the unattended path expected to reproduce the central write-up artifacts.

### Full regeneration of supporting evidence

List exact standalone commands for:

- finite-shot analysis;
- scaling analysis;
- noise analysis;
- MNIST;
- hardware submission or retrieval.

Each command should include expected runtime, resource requirements, output location, and whether credentials are required.

## Non-goals

The default workflow is not intended to:

- rerun the complete research history;
- execute every ablation or exploratory screen;
- submit hardware jobs automatically;
- hide expensive computations behind an apparently simple command;
- require the evaluator to navigate deep historical result trees;
- replace the mirrored research path convention.

## Decisions deferred until the paper is frozen

The following cannot be finalized before the central claims and figures are selected:

- exact default stages;
- exact judge-facing folder names;
- which shot, scaling, noise, MNIST, and hardware results appear in the paper;
- which expensive artifacts are regenerated versus validated;
- acceptable default runtime;
- whether any compact supporting study is cheap enough to run by default.

## Implementation sequence

1. Freeze the final QRC and evaluation protocol.
2. Freeze the finite-shot, scaling, noise, MNIST, and hardware result subsets used in the paper.
3. Select the final paper tables and figures.
4. Define the artifact index and judge-facing directory taxonomy.
5. Extend `scripts/runs/run_submission.py` with the minimal unattended workflow.
6. Add standalone regeneration scripts and documentation for expensive studies.
7. Test the default run from a clean environment without credentials.
8. Verify that all paper-critical outputs are present, checksummed, and easy to locate.

## Governing principle

The submission workflow should be small because the final paper is small, not because the underlying research is simple.

The automated path should reproduce the central evidence reliably. The repository should still expose complete, well-documented scripts for judges who choose to regenerate expensive supporting studies.