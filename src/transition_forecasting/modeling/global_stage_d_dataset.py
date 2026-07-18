from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.catalogue.global_transition_catalogue import _load_generic_ohlc
from transition_forecasting.catalogue.transition_events import (
    HORIZON,
    LEADS,
    MATCH_FEATURES,
    NEG_EXCL,
    NEG_PER_POS,
    TRAIN_CUTOFF,
    WINDOW,
    causal_features,
    log_parkinson,
    standardized_mean_differences,
)

VALIDATION_CUTOFF = pd.Timestamp("2013-01-01")


def _load_series(
    catalogue: pd.DataFrame,
    inventory: pd.DataFrame,
) -> dict[str, pd.Series]:
    path_by_index = {str(row["index"]): Path(str(row["path"])) for _, row in inventory.iterrows()}
    result: dict[str, pd.Series] = {}
    for index_name, group in catalogue.groupby("index"):
        frame = _load_generic_ohlc(path_by_index[str(index_name)])
        effective_start = pd.Timestamp(group["effective_start"].iloc[0])
        result[str(index_name)] = log_parkinson(frame[frame.index >= effective_start])
    return result


def _split(date: pd.Timestamp) -> str:
    if date < VALIDATION_CUTOFF:
        return "train"
    if date < TRAIN_CUTOFF:
        return "val"
    return "test"


def _episode_split_assignments(catalogue: pd.DataFrame) -> dict[str, str]:
    """Assign every global episode to exactly one chronological split.

    A global episode can contain market-specific onsets around a cutoff. Using
    each row's onset independently leaks one episode across splits. The
    episode's earliest onset is therefore the canonical split date and is
    propagated to all rows and matched controls.
    """
    required = {"episode_id", "onset_date"}
    missing = required.difference(catalogue.columns)
    if missing:
        raise ValueError(f"catalogue is missing required columns: {sorted(missing)}")

    episode_dates = (
        catalogue.assign(onset_date=pd.to_datetime(catalogue["onset_date"]))
        .groupby("episode_id", sort=False)["onset_date"]
        .min()
    )
    return {str(episode_id): _split(pd.Timestamp(date)) for episode_id, date in episode_dates.items()}


def _validate_manifest(manifest: pd.DataFrame) -> None:
    """Fail fast on split leakage or incomplete positive/control groups."""
    if manifest.empty:
        raise ValueError("Stage D manifest is empty")

    split_counts = manifest.groupby("episode_id")["split"].nunique()
    leaking = split_counts[split_counts > 1]
    if len(leaking):
        raise ValueError(f"global episodes span multiple splits: {list(leaking.index.astype(str))}")

    positives = manifest.loc[manifest["label"] == 1, "sample_id"].astype(str)
    controls = manifest.loc[manifest["label"] == 0]
    control_counts = controls.groupby("matched_positive_id").size()
    incomplete = {
        sample_id: int(control_counts.get(sample_id, 0))
        for sample_id in positives
        if int(control_counts.get(sample_id, 0)) != NEG_PER_POS
    }
    if incomplete:
        preview = dict(list(incomplete.items())[:10])
        raise ValueError(
            f"positive samples without exactly {NEG_PER_POS} controls: {preview}"
        )


def build_global_stage_d_dataset(
    representative_catalogue_path: Path,
    inventory_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, object]]:
    catalogue = pd.read_csv(
        representative_catalogue_path,
        parse_dates=["onset_date", "effective_start"],
    )
    inventory = pd.read_csv(inventory_path)
    series_by_index = _load_series(catalogue, inventory)
    episode_splits = _episode_split_assignments(catalogue)

    positives: list[dict[str, object]] = []
    onset_positions: dict[str, np.ndarray] = {}
    for index_name, group in catalogue.groupby("index"):
        series = series_by_index[str(index_name)]
        positions = series.index.get_indexer(pd.to_datetime(group["onset_date"]))
        onset_positions[str(index_name)] = positions[positions >= 0]

    for _, event in catalogue.iterrows():
        index_name = str(event["index"])
        episode_id = str(event["episode_id"])
        series = series_by_index[index_name]
        onset = int(series.index.get_indexer([pd.Timestamp(event["onset_date"])])[0])
        for lead in LEADS:
            origin = onset - lead
            features = causal_features(series, origin)
            if features is None or origin < WINDOW - 1 or origin + HORIZON >= len(series):
                continue
            target = series.iloc[origin + 1 : origin + 1 + HORIZON].to_numpy(dtype=float)
            positives.append(
                {
                    "sample_id": f"P_{episode_id}_{index_name}_L{lead}",
                    "label": 1,
                    "index": index_name,
                    "market_group": event["market_group"],
                    "episode_id": episode_id,
                    "split": episode_splits[episode_id],
                    "event_onset": event["onset_date"],
                    "origin_date": series.index[origin],
                    "origin_pos": origin,
                    "lead": lead,
                    "matched_positive_id": None,
                    "match_distance": np.nan,
                    **features,
                    **{f"target_x_h{i + 1}": float(value) for i, value in enumerate(target)},
                }
            )

    positive_frame = pd.DataFrame(positives)
    negatives: list[dict[str, object]] = []
    for (index_name, lead, split), group in positive_frame.groupby(["index", "lead", "split"]):
        series = series_by_index[str(index_name)]
        event_positions = onset_positions[str(index_name)]
        candidates: list[dict[str, object]] = []
        for position in range(WINDOW - 1, len(series) - HORIZON):
            date = series.index[position]
            if _split(date) != split:
                continue
            if len(event_positions) and np.min(np.abs(event_positions - position)) <= NEG_EXCL:
                continue
            features = causal_features(series, position)
            if features is not None:
                candidates.append({"origin_pos": position, "origin_date": date, **features})
        if not candidates:
            continue

        candidate_frame = pd.DataFrame(candidates)
        mean = candidate_frame[MATCH_FEATURES].astype(float).mean()
        scale = candidate_frame[MATCH_FEATURES].astype(float).std().replace(0, 1)
        standardized_candidates = (candidate_frame[MATCH_FEATURES].astype(float) - mean) / scale
        used: set[int] = set()

        for _, positive in group.iterrows():
            standardized_positive = (positive[MATCH_FEATURES].astype(float) - mean) / scale
            distances = np.sqrt(
                ((standardized_candidates - standardized_positive.to_numpy(dtype=float)) ** 2).sum(axis=1)
            )
            picked = 0
            for candidate_index in np.argsort(distances.to_numpy(dtype=float)):
                position = int(candidate_frame.iloc[candidate_index]["origin_pos"])
                if position in used:
                    continue
                used.add(position)
                picked += 1
                target = series.iloc[position + 1 : position + 1 + HORIZON].to_numpy(dtype=float)
                negatives.append(
                    {
                        "sample_id": f"N_{positive['sample_id']}_{picked}",
                        "label": 0,
                        "index": index_name,
                        "market_group": positive["market_group"],
                        "episode_id": positive["episode_id"],
                        "split": split,
                        "event_onset": positive["event_onset"],
                        "origin_date": series.index[position],
                        "origin_pos": position,
                        "lead": int(lead),
                        "matched_positive_id": positive["sample_id"],
                        "match_distance": float(distances.iloc[candidate_index]),
                        **{
                            feature: float(candidate_frame.iloc[candidate_index][feature])
                            for feature in MATCH_FEATURES
                        },
                        **{f"target_x_h{i + 1}": float(value) for i, value in enumerate(target)},
                    }
                )
                if picked >= NEG_PER_POS:
                    break

    manifest = pd.concat([positive_frame, pd.DataFrame(negatives)], ignore_index=True, sort=False)
    _validate_manifest(manifest)

    sequences = []
    for _, row in manifest.iterrows():
        series = series_by_index[str(row["index"])]
        position = int(row["origin_pos"])
        values = series.iloc[position - WINDOW + 1 : position + 1].to_numpy(dtype=float)
        sequences.append(values[:, None])
    tensor = np.asarray(sequences, dtype=float)
    balance = standardized_mean_differences(manifest)

    split_episode_sets = {
        split: set(manifest.loc[manifest["split"] == split, "episode_id"].astype(str))
        for split in ("train", "val", "test")
    }
    overlap = (
        split_episode_sets["train"].intersection(split_episode_sets["val"])
        | split_episode_sets["train"].intersection(split_episode_sets["test"])
        | split_episode_sets["val"].intersection(split_episode_sets["test"])
    )
    summary = {
        "positive_samples": int((manifest["label"] == 1).sum()),
        "negative_samples": int((manifest["label"] == 0).sum()),
        "total_samples": int(len(manifest)),
        "global_episodes": int(manifest["episode_id"].nunique()),
        "train_episodes": len(split_episode_sets["train"]),
        "val_episodes": len(split_episode_sets["val"]),
        "test_episodes": len(split_episode_sets["test"]),
        "episode_split_overlap": len(overlap),
        "validation_cutoff": VALIDATION_CUTOFF.date().isoformat(),
        "test_cutoff": TRAIN_CUTOFF.date().isoformat(),
        "samples_by_split": {
            str(split): int(count)
            for split, count in manifest.groupby("split").size().items()
        },
        "samples_by_lead": {
            str(int(lead)): int(count)
            for lead, count in manifest.groupby("lead").size().items()
        },
        "sequence_shape": list(tensor.shape),
        "max_abs_smd": float(balance["smd"].abs().max()) if len(balance) else None,
        "controls_per_positive_target": NEG_PER_POS,
        "sequence_window": WINDOW,
        "target_horizon": HORIZON,
    }
    return manifest, tensor, balance, summary


def write_global_stage_d_dataset(
    representative_catalogue_path: Path,
    inventory_path: Path,
    run_dir: Path,
) -> dict[str, object]:
    manifest, tensor, balance, summary = build_global_stage_d_dataset(
        representative_catalogue_path,
        inventory_path,
    )
    manifest.drop(columns=["origin_pos"]).to_csv(run_dir / "sample_manifest.csv", index=False)
    balance.to_csv(run_dir / "matching_balance_smd.csv", index=False)
    np.savez_compressed(
        run_dir / "sequence_tensors.npz",
        X=tensor,
        sample_id=manifest["sample_id"].astype(str).to_numpy(),
        label=manifest["label"].astype(int).to_numpy(),
        lead=manifest["lead"].astype(int).to_numpy(),
        episode_id=manifest["episode_id"].astype(str).to_numpy(),
        split=manifest["split"].astype(str).to_numpy(),
    )
    (run_dir / "stage_d_dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
