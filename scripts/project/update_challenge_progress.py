#!/usr/bin/env python3
"""Update the Phase 3 markdown checklist progress dashboard."""
from __future__ import annotations

import re
from pathlib import Path

CHECKLIST = Path('docs/challenge/PHASE3_REQUIREMENTS_AND_QRC_GUIDANCE.md')
DASHBOARD = Path('docs/challenge/PHASE3_PROGRESS.md')
BAR_WIDTH = 20


def main() -> None:
    text = CHECKLIST.read_text()
    checked = len(re.findall(r'^\s*(?:[-*]|\d+\.)\s+\[[xX]\]\s+', text, flags=re.MULTILINE))
    unchecked = len(re.findall(r'^\s*(?:[-*]|\d+\.)\s+\[ \]\s+', text, flags=re.MULTILINE))
    total = checked + unchecked
    if total == 0:
        raise ValueError(f'No markdown checkboxes found in {CHECKLIST}')

    pct = 100.0 * checked / total
    filled = round(BAR_WIDTH * checked / total)
    bar = '█' * filled + '░' * (BAR_WIDTH - filled)

    content = f'''# Phase 3 progress dashboard

Source checklist: `PHASE3_REQUIREMENTS_AND_QRC_GUIDANCE.md`

## Overall progress

```text
[{bar}] {checked} / {total} checked ({pct:.1f}%)
```

| Status | Count | Share |
|---|---:|---:|
| Checked | {checked} | {100.0 * checked / total:.1f}% |
| Not yet checked | {unchecked} | {100.0 * unchecked / total:.1f}% |
| Total | {total} | 100% |

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

## Current scientific stage

```text
challenge reread        COMPLETE
paper sanity benchmark  COMPLETE ENOUGH
ESN sanity/autopsy      COMPLETE ENOUGH
final target selection  ACTIVE
QRC architecture        NOT STARTED ON FINAL TASK
hardware validation     NOT STARTED ON FINAL TASK
submission packaging    LATER
```

## Updating this dashboard

Run:

```bash
python scripts/project/update_challenge_progress.py
```

The script reads markdown checkboxes from the master checklist and rewrites this file.
'''
    DASHBOARD.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD.write_text(content)
    print(f'Updated {DASHBOARD}: {checked}/{total} checked ({pct:.1f}%)')


if __name__ == '__main__':
    main()
