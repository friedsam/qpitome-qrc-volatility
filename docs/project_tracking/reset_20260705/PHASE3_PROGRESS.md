# Phase 3 progress dashboard

Source checklist: `PHASE3_REQUIREMENTS_AND_QRC_GUIDANCE.md`

## Overall progress

```text
[█░░░░░░░░░░░░░░░░░░░] 14 / 190 checked (7.4%)
```

| Status | Count | Share |
|---|---:|---:|
| Checked | 14 | 7.4% |
| Not yet checked | 176 | 92.6% |
| Total | 190 | 100% |

## Important interpretation

The denominator includes many late-stage tasks that cannot be completed yet, including:

- final target and protocol reporting;
- QRC architecture details;
- qubit scaling;
- shot studies;
- noise studies;
- Aquila validation;
- MNIST;
- qBraid Skill;
- final packaging and writeup requirements.

So the percentage is a compliance tracker, not a measure of scientific progress or time remaining.

## Currently checked work

The completed items currently come from:

- Track A selection;
- public data provenance;
- frequency and time-range documentation;
- missingness protocol;
- leakage-aware feature timing;
- ESN implementation;
- persistence baseline;
- AR(1) baseline;
- HAR baseline;
- HARX baseline;
- Ridge baseline;
- RMSE implementation;
- QLIKE implementation;
- Mincer-Zarnowitz implementation.

The checklist count will be resynchronized after the master requirements file is updated for the frozen target/protocol and integrated GARCH baseline.

## Current scientific stage

```text
challenge reread        COMPLETE
paper sanity benchmark  COMPLETE ENOUGH
ESN sanity/autopsy      COMPLETE ENOUGH
final target selection  COMPLETE
GARCH baseline          INTEGRATED
LSTM baseline           PENDING
TFIM control             RUNNING
Rydberg architecture    NEXT
hardware validation     NOT STARTED ON FINAL TASK
submission packaging    LATER
```

## Updating this dashboard

Run:

```bash
python scripts/project/update_challenge_progress.py
```

The script reads markdown checkboxes from the master checklist and rewrites the overall progress counts and bar.
