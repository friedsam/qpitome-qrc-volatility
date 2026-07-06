# Phase 3 Runbook

This runbook defines the default working procedure for the Phase 3 branch. It favors reproducible scripts and tests over notebook state.

## 1. Environment

Activate the project environment from the repository root.

```bash
conda activate qrc-volatility
python -m pip install -e .
```

## 2. Standard Validation Suite

Run this after pulling changes or before making a handoff commit.

```bash
python -m pytest tests/test_metrics.py tests/test_data_pipeline.py tests/test_qrc_usage_procedure.py tests/test_phase2_feedback_qrc_reference_script.py
```

Expected result at the time of this runbook:

```text
16 passed
```

## 3. QRC Quick Check

Use the quick check to verify that the canonical QRC script path works without running the full Phase 2 dataset.

```bash
python archive/phase2/scripts/run_phase2_feedback_tfim_qrc_reference.py --quick-check --no-write
```

Expected behavior:

- builds a tiny deterministic QRC sequence dataset;
- runs the feedback-TFIM QRC feature path;
- fits the readout;
- prints summary metrics and a prediction preview;
- writes no files when `--no-write` is used.

The quick-check metrics are not scientific evidence. They only validate the execution path.

## 4. Full Phase 2 Feedback-QRC Reference Run

Use this only when the processed Phase 2 dataset is present and the full digital-QRC reference needs to be regenerated.

```bash
python archive/phase2/scripts/run_phase2_feedback_tfim_qrc_reference.py
```

Expected input:

```text
data/processed/phase2_spy_vix_volatility.csv
```

Expected outputs:

```text
results/tables/phase2_feedback_tfim_qrc_reference_metrics.csv
results/tables/phase2_feedback_tfim_qrc_reference_predictions.csv
```

The full run is slower than the quick check and should not be part of routine unit testing.

## 5. Current Canonical QRC Components

Usage procedure:

```text
docs/phase3/qrc_usage_procedure.md
```

Reference script:

```text
archive/phase2/scripts/run_phase2_feedback_tfim_qrc_reference.py
```

Core implementation:

```text
src/qpitome_qrc/qrc/feedback_tfim_reservoir.py
```

Protected tests:

```text
tests/test_qrc_usage_procedure.py
tests/test_phase2_feedback_qrc_reference_script.py
```

## 6. Notebook Policy

Notebooks are evidence and exploration records. They are not the canonical execution path.

Do not rely on notebook execution counts, saved widget state, embedded plots, or cached outputs as the reproducibility layer.

Do not commit notebook output changes unless the output is intentionally part of the project evidence.

## 7. Hardware/QPU Policy

Do not include live QPU execution in routine tests.

Phase 3 hardware work should use a separate simulator-first path and only then a limited QPU pilot. Any QPU run should be isolated, parameterized, and documented before execution.

## 8. Recommended Daily Start

```bash
git status
git pull
python -m pytest tests/test_metrics.py tests/test_data_pipeline.py tests/test_qrc_usage_procedure.py tests/test_phase2_feedback_qrc_reference_script.py
```

Then run the QRC quick check only if the script path is being changed.
