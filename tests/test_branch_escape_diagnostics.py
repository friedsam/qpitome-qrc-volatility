from __future__ import annotations

import numpy as np
import pandas as pd

from qpitome_qrc.regimes.branch_escape_diagnostics import (
    EscapeDiagnosticConfig,
    commitment_window,
    rank_biserial_recovery_minus_relapse,
    residence_direction_summary,
)


def test_residence_direction_summary_and_rank_effect() -> None:
    events = pd.DataFrame(
        {
            "resolved_within_followup": [True] * 6,
            "event_type": ["recovery"] * 3 + ["relapse"] * 3,
            "event_day": [20, 30, 40, 5, 10, 15],
        }
    )
    summary = residence_direction_summary(events)
    recovery = summary.loc[summary["event_type"] == "recovery"].iloc[0]
    relapse = summary.loc[summary["event_type"] == "relapse"].iloc[0]
    assert recovery["median_event_day"] == 30
    assert relapse["median_event_day"] == 10
    assert rank_biserial_recovery_minus_relapse(events) == 1.0


def test_commitment_window_requires_sustained_multifeature_signal() -> None:
    rows = []
    for lead in range(10, 0, -1):
        for feature in ("a", "b", "c"):
            strong = lead in {6, 5, 4} and feature in {"a", "b"}
            rows.append(
                {
                    "lead_days_before_escape": lead,
                    "feature": feature,
                    "smd_recovery_minus_relapse": 1.0 if strong else 0.1,
                }
            )
    separation = pd.DataFrame(rows)
    cfg = EscapeDiagnosticConfig(
        commitment_max_lead=10,
        commitment_required_metrics=2,
        commitment_smd_threshold=0.8,
        commitment_consecutive_days=3,
    )
    assert commitment_window(separation, cfg) == 6


def test_commitment_window_can_be_absent() -> None:
    separation = pd.DataFrame(
        {
            "lead_days_before_escape": np.repeat([3, 2, 1], 2),
            "feature": ["a", "b"] * 3,
            "smd_recovery_minus_relapse": [0.1] * 6,
        }
    )
    cfg = EscapeDiagnosticConfig(commitment_max_lead=3)
    assert commitment_window(separation, cfg) is None
