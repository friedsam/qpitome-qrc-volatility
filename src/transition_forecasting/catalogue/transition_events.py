from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = ["SPX", "GDAXI", "FCHI", "FTSE", "OMXSPI", "N225", "KS11", "HSI"]
TRAIN_CUTOFF = pd.Timestamp("2016-01-01")
K, M = 10, 15
PRIOR_WIN, PRIOR_MAX = 10, 2
SEP = 60
CLUSTER_DAYS = 7
LEADS = (1, 5, 10)
HORIZON = 10
NEG_PER_POS = 3
NEG_EXCL = 60
MATCH_FEATURES = ["level", "mean5", "mean20", "slope5", "slope20", "std20", "max20"]
WINDOW = 40
DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d", "%d/%m/%Y")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dates_strict(values: pd.Series, source: Path) -> pd.DatetimeIndex:
    raw = values.astype(str).str.strip()
    successful: list[tuple[str, pd.Series]] = []
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        if parsed.notna().all():
            successful.append((fmt, parsed))
    if not successful:
        raise ValueError(f"{source}: Date column does not match {DATE_FORMATS}")
    unique_results = {tuple(parsed.astype("int64")) for _, parsed in successful}
    if len(unique_results) > 1:
        raise ValueError(f"{source}: ambiguous date interpretation")
    return pd.DatetimeIndex(successful[0][1])


def load_ohlc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"Date", "High", "Low"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    frame["Date"] = parse_dates_strict(frame["Date"], path)
    for column in ["Price", "Open", "High", "Low"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column].astype(str).str.replace(",", "", regex=False), errors="coerce"
            )
    frame = frame.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
    frame = frame[(frame.High > 0) & (frame.Low > 0) & (frame.High >= frame.Low)]
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError(f"{path}: invalid date index")
    return frame


def log_parkinson(frame: pd.DataFrame) -> pd.Series:
    volatility = np.abs(np.log(frame.High / frame.Low)) / np.sqrt(4 * np.log(2))
    return np.log(volatility.replace(0, np.nan)).dropna()


def detect_onsets(series: pd.Series, threshold: float) -> list[int]:
    high = (series.to_numpy() >= threshold).astype(np.int8)
    onsets: list[int] = []
    last = -10**9
    for position in range(PRIOR_WIN, len(series) - M + 1):
        if not high[position] or high[position : position + M].sum() < K:
            continue
        if high[position - PRIOR_WIN : position].sum() > PRIOR_MAX or position - last < SEP:
            continue
        onsets.append(position)
        last = position
    return onsets


def slope(values: np.ndarray) -> float:
    if len(values) < 2 or not np.isfinite(values).all():
        return np.nan
    return float(np.polyfit(np.arange(len(values), dtype=float), values.astype(float), 1)[0])


def causal_features(series: pd.Series, position: int) -> dict[str, float] | None:
    if position < 20:
        return None
    recent5 = series.iloc[position - 4 : position + 1].to_numpy(dtype=float)
    recent20 = series.iloc[position - 19 : position + 1].to_numpy(dtype=float)
    if len(recent20) != 20 or not np.isfinite(recent20).all():
        return None
    return {
        "level": float(series.iloc[position]),
        "mean5": float(recent5.mean()),
        "mean20": float(recent20.mean()),
        "slope5": slope(recent5),
        "slope20": slope(recent20),
        "std20": float(recent20.std(ddof=1)),
        "max20": float(recent20.max()),
    }


def cluster_events(catalogue: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(catalogue.onset_date).unique())
    clusters: list[list[pd.Timestamp]] = []
    for raw_date in dates:
        date = pd.Timestamp(raw_date)
        if not clusters or (date - clusters[-1][-1]).days > CLUSTER_DAYS:
            clusters.append([date])
        else:
            clusters[-1].append(date)
    mapping = {
        date: f"E{cluster_number:03d}"
        for cluster_number, cluster in enumerate(clusters, 1)
        for date in cluster
    }
    result = catalogue.copy()
    result["episode_id"] = pd.to_datetime(result.onset_date).map(mapping)
    return result


def era(date: pd.Timestamp) -> str:
    return "train" if date < TRAIN_CUTOFF else "test"


def recovery_time(series: pd.Series, onset: int, threshold: float, max_days: int = 60) -> float:
    for position in range(onset + 1, min(len(series), onset + max_days)):
        if series.iloc[position] < threshold:
            return float(position - onset)
    return np.nan


def event_anatomy(series: pd.Series, onset: int, threshold: float) -> dict[str, float]:
    future = series.iloc[onset : onset + 60]
    previous = float(series.iloc[onset - 1])
    current = float(series.iloc[onset])
    return {
        "pre_onset_level": previous,
        "onset_level": current,
        "jump_logvol": current - previous,
        "peak60": float(future.max()),
        "peak_gain60": float(future.max() - previous),
        "persist60": int((future >= threshold).sum()),
        "recovery_days": recovery_time(series, onset, threshold),
    }


def standardized_mean_differences(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (index, lead), group in manifest.groupby(["index", "lead"]):
        positives = group[group.label == 1]
        negatives = group[group.label == 0]
        for feature in MATCH_FEATURES:
            positive_values = positives[feature].astype(float)
            negative_values = negatives[feature].astype(float)
            pooled = np.sqrt((positive_values.var(ddof=1) + negative_values.var(ddof=1)) / 2)
            smd = 0.0 if not np.isfinite(pooled) or pooled == 0 else float(
                (positive_values.mean() - negative_values.mean()) / pooled
            )
            rows.append(
                {
                    "index": index,
                    "lead": int(lead),
                    "feature": feature,
                    "smd": smd,
                    "n_positive": len(positive_values),
                    "n_negative": len(negative_values),
                }
            )
    return pd.DataFrame(rows)


def save_sequence_tensors(manifest: pd.DataFrame, series_by_index: dict[str, pd.Series], run_dir: Path) -> None:
    sequences: list[np.ndarray] = []
    sample_ids: list[str] = []
    for _, row in manifest.iterrows():
        series = series_by_index[str(row["index"])]
        location = series.index.get_indexer([pd.Timestamp(row["origin_date"])])[0]
        if location < WINDOW - 1:
            continue
        sequence = series.iloc[location - WINDOW + 1 : location + 1].to_numpy(dtype=float)
        if len(sequence) == WINDOW and np.isfinite(sequence).all():
            sequences.append(sequence[:, None])
            sample_ids.append(str(row["sample_id"]))
    np.savez_compressed(
        run_dir / "sequence_tensors.npz",
        X=np.asarray(sequences),
        sample_id=np.asarray(sample_ids),
    )


def write_plots(
    catalogue: pd.DataFrame,
    manifest: pd.DataFrame,
    series_by_index: dict[str, pd.Series],
    thresholds: dict[str, dict[str, float]],
    run_dir: Path,
) -> None:
    if "SPX" in series_by_index:
        series = series_by_index["SPX"]
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.plot(series.index, series.values, linewidth=0.8)
        for date in pd.to_datetime(catalogue[catalogue["index"] == "SPX"].onset_date):
            axis.axvline(date, linewidth=0.6, alpha=0.5)
        axis.axhline(thresholds["SPX"]["T80"], linestyle="--", linewidth=0.8)
        axis.set_title("SPX log Parkinson volatility and persistent-transition onsets")
        axis.set_xlabel("Date")
        axis.set_ylabel("log volatility")
        figure.tight_layout()
        figure.savefig(run_dir / "spx_onsets.png", dpi=180)
        plt.close(figure)

    for lead in LEADS:
        subset = manifest[manifest.lead == lead]
        figure, axis = plt.subplots(figsize=(8, 4))
        for label, name in [(1, "onset"), (0, "matched control")]:
            curves: list[np.ndarray] = []
            for _, row in subset[subset.label == label].iterrows():
                series = series_by_index[str(row["index"])]
                location = series.index.get_indexer([pd.Timestamp(row["origin_date"])])[0]
                if location >= WINDOW - 1:
                    values = series.iloc[location - WINDOW + 1 : location + 1].to_numpy(dtype=float)
                    curves.append(values - values[-1])
            if curves:
                array = np.asarray(curves)
                x_axis = np.arange(-WINDOW + 1, 1)
                axis.plot(x_axis, np.nanmedian(array, axis=0), label=name)
                lower, upper = np.nanquantile(array, [0.25, 0.75], axis=0)
                axis.fill_between(x_axis, lower, upper, alpha=0.2)
        axis.axvline(0, linestyle="--", linewidth=0.8)
        axis.set_title(f"Causal pre-origin trajectories, lead={lead}")
        axis.set_xlabel("Sessions relative to forecast origin")
        axis.set_ylabel("log volatility relative to origin")
        axis.legend()
        figure.tight_layout()
        figure.savefig(run_dir / f"preorigin_trajectories_lead{lead}.png", dpi=180)
        plt.close(figure)

    distances = manifest.loc[manifest.label == 0, "match_distance"].dropna().astype(float)
    if len(distances):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(distances, bins=30)
        axis.set_title("Matched-control distance distribution")
        axis.set_xlabel("Standardized Euclidean distance")
        figure.tight_layout()
        figure.savefig(run_dir / "match_distance_distribution.png", dpi=180)
        plt.close(figure)


def write_checksums(paths: Iterable[Path], destination: Path) -> None:
    rows = [
        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(set(paths), key=str)
        if path.is_file()
    ]
    pd.DataFrame(rows).to_csv(destination, index=False)


def run_pipeline(data_dir: Path, run_dir: Path, seed: int = 42) -> dict[str, object]:
    del seed
    run_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [data_dir / "global index etf return" / f"{name}.csv" for name in NAMES]
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required OHLC files: {missing}")

    series_by_index: dict[str, pd.Series] = {}
    thresholds: dict[str, dict[str, float]] = {}
    raw_events: list[dict[str, object]] = []
    for name, path in zip(NAMES, input_paths):
        series = log_parkinson(load_ohlc(path))
        series_by_index[name] = series
        training = series[series.index < TRAIN_CUTOFF]
        if len(training) < 250:
            raise ValueError(f"{name}: insufficient pre-2016 training history")
        turbulent = float(training.quantile(0.8))
        calm = float(training.quantile(0.5))
        thresholds[name] = {"T80": turbulent, "C50": calm, "train_days": int(len(training))}
        for onset in detect_onsets(series, turbulent):
            raw_events.append(
                {
                    "index": name,
                    "onset_date": series.index[onset],
                    "onset_pos": onset,
                    "origin_level": float(series.iloc[onset - 1]),
                    "origin_pctile": float((training <= series.iloc[onset - 1]).mean()),
                    "era": era(series.index[onset]),
                    **event_anatomy(series, onset, turbulent),
                }
            )

    catalogue = cluster_events(pd.DataFrame(raw_events).sort_values("onset_date").reset_index(drop=True))
    episodes = (
        catalogue.groupby("episode_id", as_index=False)
        .agg(
            start_date=("onset_date", "min"),
            end_date=("onset_date", "max"),
            n_indices=("index", "nunique"),
            indices=("index", lambda values: ",".join(sorted(set(values)))),
            era=("era", "first"),
        )
    )
    episodes["split"] = episodes["era"]
    catalogue["split"] = catalogue.episode_id.map(dict(zip(episodes.episode_id, episodes.split)))

    positive_rows: list[dict[str, object]] = []
    for _, event in catalogue.iterrows():
        series = series_by_index[str(event["index"])]
        for lead in LEADS:
            origin = int(event.onset_pos) - lead
            features = causal_features(series, origin)
            if features is None:
                continue
            target = series.iloc[origin + 1 : origin + 1 + HORIZON].to_numpy(dtype=float)
            if len(target) != HORIZON:
                continue
            positive_rows.append(
                {
                    "sample_id": f"P_{event.episode_id}_{event['index']}_L{lead}",
                    "label": 1,
                    "index": event["index"],
                    "episode_id": event.episode_id,
                    "era": event.era,
                    "split": event.split,
                    "event_onset": event.onset_date,
                    "origin_date": series.index[origin],
                    "origin_pos": origin,
                    "lead": lead,
                    **features,
                    **{f"target_x_h{offset + 1}": float(value) for offset, value in enumerate(target)},
                }
            )
    positives = pd.DataFrame(positive_rows)

    negative_rows: list[dict[str, object]] = []
    reuse_tracker: dict[tuple[str, pd.Timestamp], set[int]] = {}
    for name in NAMES:
        series = series_by_index[name]
        event_positions = catalogue.loc[catalogue["index"] == name, "onset_pos"].astype(int).to_numpy()
        for lead in LEADS:
            positive_subset = positives[(positives["index"] == name) & (positives["lead"] == lead)]
            for split in ("train", "test"):
                positive_split = positive_subset[positive_subset.era == split]
                if positive_split.empty:
                    continue
                candidates: list[dict[str, object]] = []
                for position in range(20, len(series) - HORIZON):
                    date = series.index[position]
                    if era(date) != split:
                        continue
                    if len(event_positions) and np.min(np.abs(event_positions - position)) <= NEG_EXCL:
                        continue
                    if any(position < event <= position + HORIZON for event in event_positions):
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
                for _, positive in positive_split.iterrows():
                    standardized_positive = (positive[MATCH_FEATURES].astype(float) - mean) / scale
                    distances = np.sqrt(
                        ((standardized_candidates - standardized_positive.to_numpy(dtype=float)) ** 2).sum(axis=1)
                    )
                    picked = 0
                    for candidate_index in np.argsort(distances.to_numpy(dtype=float)):
                        position = int(candidate_frame.iloc[candidate_index].origin_pos)
                        if position in used:
                            continue
                        used.add(position)
                        picked += 1
                        target = series.iloc[position + 1 : position + 1 + HORIZON].to_numpy(dtype=float)
                        reuse_tracker.setdefault((name, pd.Timestamp(series.index[position])), set()).add(int(lead))
                        negative_rows.append(
                            {
                                "sample_id": f"N_{positive.sample_id}_{picked}",
                                "label": 0,
                                "index": name,
                                "episode_id": positive.episode_id,
                                "era": split,
                                "split": positive.split,
                                "event_onset": positive.event_onset,
                                "origin_date": series.index[position],
                                "origin_pos": position,
                                "lead": lead,
                                "matched_positive_id": positive.sample_id,
                                "match_distance": float(distances.iloc[candidate_index]),
                                **{feature: float(candidate_frame.iloc[candidate_index][feature]) for feature in MATCH_FEATURES},
                                **{f"target_x_h{offset + 1}": float(value) for offset, value in enumerate(target)},
                            }
                        )
                        if picked >= NEG_PER_POS:
                            break

    negatives = pd.DataFrame(negative_rows)
    manifest = pd.concat([positives, negatives], ignore_index=True, sort=False)
    reused = {key for key, leads in reuse_tracker.items() if len(leads) > 1}
    manifest["control_reused_across_leads"] = False
    if reused:
        mask = (manifest.label == 0) & manifest.apply(
            lambda row: (str(row["index"]), pd.Timestamp(row["origin_date"])) in reused,
            axis=1,
        )
        manifest.loc[mask, "control_reused_across_leads"] = True

    artifacts: list[Path] = []

    def save_csv(frame: pd.DataFrame, name: str) -> None:
        path = run_dir / name
        frame.to_csv(path, index=False)
        artifacts.append(path)

    save_csv(catalogue.drop(columns=["onset_pos"]), "transition_catalogue.csv")
    save_csv(episodes, "global_episode_catalogue.csv")
    save_csv(manifest.drop(columns=["origin_pos"]), "sample_manifest.csv")
    save_csv(pd.DataFrame(thresholds).T.reset_index(names="index"), "training_thresholds.csv")
    balance = standardized_mean_differences(manifest)
    save_csv(balance, "matching_balance_smd.csv")
    reuse = (
        manifest[manifest.label == 0]
        .groupby(["index", "origin_date"], as_index=False)
        .agg(
            n_rows=("sample_id", "size"),
            n_leads=("lead", "nunique"),
            leads=("lead", lambda values: ",".join(map(str, sorted(set(values))))),
        )
    )
    save_csv(reuse, "control_reuse_report.csv")

    save_sequence_tensors(manifest, series_by_index, run_dir)
    artifacts.append(run_dir / "sequence_tensors.npz")
    write_plots(catalogue, manifest, series_by_index, thresholds, run_dir)
    artifacts.extend(run_dir.glob("*.png"))

    summary = {
        "events": int(len(catalogue)),
        "global_episodes": int(len(episodes)),
        "train_episodes": int((episodes.split == "train").sum()),
        "test_episodes": int((episodes.split == "test").sum()),
        "positive_samples": int(len(positives)),
        "negative_samples": int(len(negatives)),
        "leads": list(LEADS),
        "horizon": HORIZON,
        "sequence_window": WINDOW,
        "max_abs_smd": float(balance.smd.abs().max()) if len(balance) else None,
        "controls_reused_across_leads": int(reuse.loc[reuse.n_leads > 1, "n_rows"].sum()) if len(reuse) else 0,
        "rv_concordance_status": "disabled unless an exact date column is supplied",
        "independence_unit": "global episode_id",
        "pipeline_status": "completed_end_to_end",
    }
    summary_path = run_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    artifacts.append(summary_path)
    write_checksums(input_paths, run_dir / "input_checksums.csv")
    write_checksums(artifacts, run_dir / "output_checksums.csv")
    run_manifest = {
        "run_type": "persistent_volatility_transition_catalogue",
        "parameters": {
            "train_cutoff": str(TRAIN_CUTOFF.date()),
            "persistence_k": K,
            "persistence_m": M,
            "separation_sessions": SEP,
            "cluster_days": CLUSTER_DAYS,
            "leads": list(LEADS),
            "horizon": HORIZON,
            "window": WINDOW,
            "negative_controls_per_positive": NEG_PER_POS,
        },
        "summary": summary,
        "artifacts": sorted(path.name for path in artifacts),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, default=str))
    return summary
