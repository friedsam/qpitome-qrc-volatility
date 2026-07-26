from __future__ import annotations

import pandas as pd

from transition_forecasting.modeling.classical_benchmarks.esn_tuning import summarize_candidates


def test_transition_first_ranking_applies_order_and_baseline_guardrails() -> None:
    rows = []
    for config_id, order, transition, controls, pooled, transition_rmse in (
        ("good", "ordered", 1.40, 0.57, 0.82, 0.58),
        ("good", "shuffled", 1.50, 0.58, 0.84, 0.59),
        ("bad_control", "ordered", 1.20, 0.80, 0.90, 0.57),
        ("bad_control", "shuffled", 1.30, 0.78, 0.91, 0.58),
    ):
        for fold in (4, 5, 6):
            values = {
                "Transition": (transition, transition_rmse),
                "Controls": (controls, 0.455 if config_id == "good" else 0.50),
                "Pooled": (pooled, 0.49 if config_id == "good" else 0.53),
                "L1": (transition + 0.4, transition_rmse + 0.1),
                "L5": (transition, transition_rmse),
                "L10": (0.4, 0.42),
            }
            for group, (qlike, rmse) in values.items():
                rows.append(
                    {
                        "config_id": config_id,
                        "alpha": 1000.0,
                        "order": order,
                        "group": group,
                        "fold": fold,
                        "qlike": qlike,
                        "rmse": rmse,
                        "mz_beta": 1.0,
                        "mz_r2": 0.2,
                    }
                )
    ranking = summarize_candidates(pd.DataFrame(rows))
    assert ranking.iloc[0]["config_id"] == "good"
    assert bool(ranking.iloc[0]["admissible"])
    assert not bool(ranking[ranking["config_id"].eq("bad_control")].iloc[0]["admissible"])
