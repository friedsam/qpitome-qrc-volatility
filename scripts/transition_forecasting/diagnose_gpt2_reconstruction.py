from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from transition_forecasting.catalogue.global_transition_catalogue import (
    build_global_transition_catalogue,
)

HISTORICAL_INVENTORY = Path(
    "results/transition_forecasting/quality/global_index_ohlc_audit/"
    "global_index_audit_001/global_index_ohlc_inventory.csv"
)
HISTORICAL_RANGE = Path(
    "results/transition_forecasting/quality/global_ohlc_range_quality/"
    "global_range_quality_001/global_range_quality.csv"
)
HISTORICAL_CATALOGUE_ROOT = Path(
    "results/transition_forecasting/catalogue/global_transition_catalogue/"
    "global_transition_catalogue_003"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_event_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["index"].astype(str)
        + "|"
        + pd.to_datetime(frame["onset_date"]).dt.strftime("%Y-%m-%d")
    )


def compare_source_hashes(raw_root: Path, historical_inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, item in historical_inventory.iterrows():
        historical_path = Path(str(item["path"]))
        if historical_path.name == "all_indices_data.csv":
            local_path = raw_root.parent / historical_path.name
        else:
            local_path = raw_root / historical_path.name
        actual = sha256_file(local_path) if local_path.is_file() else None
        expected = str(item["sha256"])
        rows.append(
            {
                "index": str(item["index"]),
                "local_path": str(local_path),
                "exists": local_path.is_file(),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_match": actual == expected,
                "historical_eligible": bool(item["eligible"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_event_delta(reference: pd.DataFrame, rebuilt: pd.DataFrame) -> dict[str, object]:
    ref = reference.copy()
    new = rebuilt.copy()
    ref["semantic_key"] = semantic_event_key(ref)
    new["semantic_key"] = semantic_event_key(new)
    ref_keys = set(ref["semantic_key"])
    new_keys = set(new["semantic_key"])
    removed = ref.loc[ref["semantic_key"].isin(ref_keys - new_keys)].copy()
    added = new.loc[new["semantic_key"].isin(new_keys - ref_keys)].copy()
    return {
        "reference_rows": int(len(ref)),
        "rebuilt_rows": int(len(new)),
        "shared_rows": int(len(ref_keys & new_keys)),
        "missing_from_rebuild": int(len(ref_keys - new_keys)),
        "added_in_rebuild": int(len(new_keys - ref_keys)),
        "missing_by_index": removed.groupby("index").size().sort_values(ascending=False).to_dict(),
        "added_by_index": added.groupby("index").size().sort_values(ascending=False).to_dict(),
        "missing_events": removed[["index", "onset_date"]].astype(str).to_dict(orient="records"),
        "added_events": added[["index", "onset_date"]].astype(str).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct GPT-2 directly from its committed source contract and locate any drift."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path(
            "data/raw/transition_forecasting/global_stock_indices_historical_data/"
            "individual_indices_data"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scratch/gpt2_reconstruction_diagnostic.json"),
    )
    args = parser.parse_args()

    inventory = pd.read_csv(HISTORICAL_INVENTORY)
    hash_report = compare_source_hashes(args.raw_root, inventory)

    # Preserve the exact historical eligibility and file ordering, but redirect
    # the committed paths to the caller's local raw files.
    redirected = inventory.copy()
    paths: list[str] = []
    for _, item in redirected.iterrows():
        historical_path = Path(str(item["path"]))
        if historical_path.name == "all_indices_data.csv":
            local_path = args.raw_root.parent / historical_path.name
        else:
            local_path = args.raw_root / historical_path.name
        paths.append(str(local_path.resolve()))
    redirected["path"] = paths

    with tempfile.TemporaryDirectory(prefix="gpt2-diagnostic-") as temp:
        inventory_path = Path(temp) / "historical_inventory_redirected.csv"
        redirected.to_csv(inventory_path, index=False)
        raw, clustered, representative, report = build_global_transition_catalogue(
            args.raw_root.parent,
            inventory_path,
            HISTORICAL_RANGE,
            apply_structural_corrections=False,
        )

    historical_raw = pd.read_csv(
        HISTORICAL_CATALOGUE_ROOT / "raw_transition_catalogue.csv",
        parse_dates=["onset_date"],
    )
    historical_clustered = pd.read_csv(
        HISTORICAL_CATALOGUE_ROOT / "clustered_transition_catalogue.csv",
        parse_dates=["onset_date"],
    )
    historical_representative = pd.read_csv(
        HISTORICAL_CATALOGUE_ROOT / "representative_transition_catalogue.csv",
        parse_dates=["onset_date"],
    )

    raw_delta = summarize_event_delta(historical_raw, raw)
    representative_delta = summarize_event_delta(historical_representative, representative)

    cluster_membership_reference = {
        key: value
        for key, value in zip(
            semantic_event_key(historical_clustered),
            historical_clustered["episode_id"].astype(str),
        )
    }
    cluster_membership_rebuilt = {
        key: value
        for key, value in zip(
            semantic_event_key(clustered),
            clustered["episode_id"].astype(str),
        )
    }
    shared_cluster_keys = set(cluster_membership_reference) & set(cluster_membership_rebuilt)
    cluster_id_disagreements = [
        {
            "event": key,
            "historical_episode_id": cluster_membership_reference[key],
            "rebuilt_episode_id": cluster_membership_rebuilt[key],
        }
        for key in sorted(shared_cluster_keys)
        if cluster_membership_reference[key] != cluster_membership_rebuilt[key]
    ]

    result = {
        "historical_contract": {
            "inventory": str(HISTORICAL_INVENTORY),
            "inventory_sha256": sha256_file(HISTORICAL_INVENTORY),
            "range_quality": str(HISTORICAL_RANGE),
            "range_quality_sha256": sha256_file(HISTORICAL_RANGE),
            "catalogue_root": str(HISTORICAL_CATALOGUE_ROOT),
        },
        "source_hashes": {
            "files": int(len(hash_report)),
            "matching": int(hash_report["hash_match"].sum()),
            "mismatching": int((~hash_report["hash_match"]).sum()),
            "mismatches": hash_report.loc[
                ~hash_report["hash_match"],
                ["index", "local_path", "exists", "expected_sha256", "actual_sha256", "historical_eligible"],
            ].to_dict(orient="records"),
        },
        "rebuilt_summary": report["summary"],
        "raw_event_delta": raw_delta,
        "representative_event_delta": representative_delta,
        "cluster_id_disagreements_on_shared_events": cluster_id_disagreements,
        "diagnosis": (
            "source_snapshot_mismatch"
            if (~hash_report["hash_match"]).any()
            else "algorithm_or_environment_mismatch"
            if raw_delta["missing_from_rebuild"] or raw_delta["added_in_rebuild"]
            else "raw_events_match_check_clustering"
            if cluster_id_disagreements
            else "exact_catalogue_reconstruction"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
