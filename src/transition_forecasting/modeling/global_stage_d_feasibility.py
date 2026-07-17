from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from transition_forecasting.catalogue.global_transition_catalogue import _load_generic_ohlc
from transition_forecasting.catalogue.transition_events import HORIZON, LEADS, WINDOW, log_parkinson


def assess_stage_d_feasibility(
    representative_catalogue_path: Path,
    inventory_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    catalogue = pd.read_csv(representative_catalogue_path, parse_dates=["onset_date"])
    inventory = pd.read_csv(inventory_path)
    path_by_index = {
        str(row["index"]): Path(str(row["path"]))
        for _, row in inventory.iterrows()
    }

    series_by_index: dict[str, pd.Series] = {}
    rows: list[dict[str, object]] = []
    for _, event in catalogue.iterrows():
        index_name = str(event["index"])
        if index_name not in series_by_index:
            frame = _load_generic_ohlc(path_by_index[index_name])
            effective_start = pd.Timestamp(event["effective_start"])
            series_by_index[index_name] = log_parkinson(frame[frame.index >= effective_start])
        series = series_by_index[index_name]
        onset_location = int(series.index.get_indexer([pd.Timestamp(event["onset_date"])])[0])
        for lead in LEADS:
            origin = onset_location - lead
            history_available = origin >= WINDOW - 1
            target_available = origin >= 0 and origin + HORIZON < len(series)
            rows.append(
                {
                    "episode_id": event["episode_id"],
                    "index": index_name,
                    "market_group": event["market_group"],
                    "onset_date": event["onset_date"],
                    "lead": lead,
                    "origin_date": series.index[origin] if origin >= 0 else pd.NaT,
                    "history_available": history_available,
                    "target_available": target_available,
                    "eligible_positive": history_available and target_available,
                }
            )

    detail = pd.DataFrame(rows)
    eligible = detail[detail["eligible_positive"]]
    by_lead = (
        detail.groupby("lead", as_index=False)
        .agg(
            candidate_events=("episode_id", "size"),
            eligible_positive_events=("eligible_positive", "sum"),
            eligible_episodes=("episode_id", lambda values: values[detail.loc[values.index, "eligible_positive"]].nunique()),
        )
    )
    summary = {
        "representative_events": int(len(catalogue)),
        "global_episodes": int(catalogue["episode_id"].nunique()),
        "eligible_positive_rows": int(len(eligible)),
        "eligible_positive_rows_by_lead": {
            str(int(row["lead"])): int(row["eligible_positive_events"])
            for _, row in by_lead.iterrows()
        },
        "eligible_episodes_by_lead": {
            str(int(row["lead"])): int(row["eligible_episodes"])
            for _, row in by_lead.iterrows()
        },
        "events_missing_history": int((~detail["history_available"]).sum()),
        "events_missing_target": int((~detail["target_available"]).sum()),
        "sequence_window": WINDOW,
        "target_horizon": HORIZON,
        "leads": list(LEADS),
    }
    return detail, summary


def write_stage_d_feasibility(
    representative_catalogue_path: Path,
    inventory_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    detail, summary = assess_stage_d_feasibility(
        representative_catalogue_path,
        inventory_path,
    )
    detail.to_csv(run_dir / "positive_eligibility.csv", index=False)
    (run_dir / "stage_d_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary
