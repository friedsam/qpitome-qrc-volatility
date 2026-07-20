from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from transition_forecasting.catalogue.global_transition_catalogue import (
    write_global_transition_outputs,
)
from transition_forecasting.data.cleaning import build_cleaned_ohlc
from transition_forecasting.data.parity import audit_parity_control_dataset
from transition_forecasting.data.validation import audit_processed_dataset, sha256_file
from transition_forecasting.data.volatility import (
    build_daily_volatility,
    consolidate_cleaned_ohlc,
)
from transition_forecasting.modeling.global_stage_d_dataset import (
    write_global_stage_d_dataset,
)

FROZEN_INVENTORY = Path(
    "results/transition_forecasting/quality/global_index_ohlc_audit/"
    "global_index_audit_001/global_index_ohlc_inventory.csv"
)
FROZEN_RANGE_QUALITY = Path(
    "results/transition_forecasting/quality/global_ohlc_range_quality/"
    "global_range_quality_001/global_range_quality.csv"
)

FINAL_FILES = (
    "cleaned_ohlc.csv.gz",
    "daily_volatility.csv.gz",
    "transition_catalogue.csv",
    "sample_manifest.csv",
    "sequence_tensors.npz",
    "row_corrections.csv",
    "manifest.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_raw_file(raw_root: Path, individual_raw: Path, historical_path: str) -> Path:
    name = Path(historical_path).name
    if name == "all_indices_data.csv":
        return raw_root / name
    return individual_raw / name


def _prepare_frozen_gpt2_contract(
    raw_root: Path,
    individual_raw: Path,
    cleaned_root: Path,
    run_dir: Path,
) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    """Verify and materialize GPT-2's exact committed source contract.

    Eligibility, ordering, and effective starts are never recomputed. Every raw
    file is first checked against the SHA-256 recorded by GPT-2. Eligible file
    paths are then redirected to the correction-only copies while all frozen
    decisions remain unchanged.
    """
    if not FROZEN_INVENTORY.is_file():
        raise FileNotFoundError(f"Missing frozen GPT-2 inventory: {FROZEN_INVENTORY}")
    if not FROZEN_RANGE_QUALITY.is_file():
        raise FileNotFoundError(f"Missing frozen GPT-2 range contract: {FROZEN_RANGE_QUALITY}")

    run_dir.mkdir(parents=True, exist_ok=False)
    inventory = pd.read_csv(FROZEN_INVENTORY)
    mismatches: list[dict[str, object]] = []
    redirected_paths: list[str] = []

    for _, row in inventory.iterrows():
        raw_path = _resolve_raw_file(raw_root, individual_raw, str(row["path"]))
        expected = str(row["sha256"])
        actual = sha256_file(raw_path) if raw_path.is_file() else None
        if actual != expected:
            mismatches.append(
                {
                    "index": str(row["index"]),
                    "path": str(raw_path),
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                }
            )

        cleaned_path = cleaned_root / raw_path.name
        if bool(row["eligible"]):
            if not cleaned_path.is_file():
                raise FileNotFoundError(f"Missing corrected modeling input: {cleaned_path}")
            redirected_paths.append(str(cleaned_path.resolve()))
        else:
            redirected_paths.append(str(raw_path.resolve()))

    if mismatches:
        preview = mismatches[:5]
        raise RuntimeError(
            "Raw source snapshot does not match GPT-2's frozen inventory: "
            f"{len(mismatches)} mismatches; preview={preview}"
        )

    modeling_inventory = inventory.copy()
    modeling_inventory["path"] = redirected_paths
    inventory_path = run_dir / "gpt2_inventory_redirected.csv"
    modeling_inventory.to_csv(inventory_path, index=False)

    range_quality = pd.read_csv(FROZEN_RANGE_QUALITY)
    redirected_range_paths: list[str] = []
    for _, row in range_quality.iterrows():
        source_name = Path(str(row["path"])).name
        cleaned_path = cleaned_root / source_name
        if not cleaned_path.is_file():
            raise FileNotFoundError(f"Missing corrected range input: {cleaned_path}")
        redirected_range_paths.append(str(cleaned_path.resolve()))
    range_quality["path"] = redirected_range_paths
    range_path = run_dir / "gpt2_range_quality_redirected.csv"
    range_quality.to_csv(range_path, index=False)

    inventory_summary = {
        "contract": "frozen_gpt2",
        "source": str(FROZEN_INVENTORY),
        "source_sha256": sha256_file(FROZEN_INVENTORY),
        "files": int(len(inventory)),
        "eligible_files": int(inventory["eligible"].astype(bool).sum()),
        "raw_hashes_verified": int(len(inventory)),
        "raw_hash_mismatches": 0,
    }
    range_summary = {
        "contract": "frozen_gpt2",
        "source": str(FROZEN_RANGE_QUALITY),
        "source_sha256": sha256_file(FROZEN_RANGE_QUALITY),
        "rows": int(len(range_quality)),
        "effective_starts": int(range_quality["recommended_effective_start"].notna().sum()),
    }
    return inventory_path, range_path, inventory_summary, range_summary


def _file_inventory(root: Path) -> dict[str, dict[str, object]]:
    inventory: dict[str, dict[str, object]] = {}
    for name in FINAL_FILES:
        path = root / name
        if not path.is_file():
            raise RuntimeError(f"Missing final dataset file: {name}")
        inventory[name] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return inventory


def _promote_candidate(candidate: Path, output_dir: Path, *, force: bool) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")

    backup = candidate.parent / "prior-canonical-backup"
    if backup.exists():
        raise RuntimeError(f"unexpected transient backup collision: {backup}")

    if output_dir.exists():
        os.replace(output_dir, backup)
    try:
        os.replace(candidate, output_dir)
    except Exception:
        if backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _audit_for_mode(
    dataset_dir: Path,
    *,
    controls_per_positive: int,
    apply_structural_corrections: bool,
) -> dict[str, object]:
    if apply_structural_corrections:
        return audit_processed_dataset(
            dataset_dir,
            controls_per_positive=controls_per_positive,
        )
    return audit_parity_control_dataset(
        dataset_dir,
        controls_per_positive=controls_per_positive,
    )


def build_processed_dataset(
    raw_root: Path,
    output_dir: Path,
    *,
    expected_structural_flags: int = 12,
    expected_affected_indices: int = 2,
    controls_per_positive: int = 3,
    apply_structural_corrections: bool = True,
    force: bool = False,
) -> dict[str, object]:
    """Build, validate, and atomically publish one processed dataset.

    The build uses GPT-2's committed inventory, raw hashes, eligibility, and
    effective starts. The only intended data change is removal of the 12 frozen
    structural bad prints before volatility construction.

    ``apply_structural_corrections=False`` exists only for temporary parity
    reconstruction on the experimental branch and is not a submission mode.
    """
    raw_root = Path(raw_root).resolve()
    output_dir = Path(output_dir).resolve()
    individual_raw = raw_root / "individual_indices_data"
    if not individual_raw.is_dir():
        raise FileNotFoundError(f"Missing raw individual-index directory: {individual_raw}")
    if output_dir.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="transition-process-",
        dir=output_dir.parent,
    ) as temporary:
        temp_root = Path(temporary)
        cleaning_root = temp_root / "cleaning"
        contract_root = temp_root / "contract"
        catalogue_root = temp_root / "catalogue"
        stage_d_root = temp_root / "stage_d"
        final_root = temp_root / "final"
        final_root.mkdir()

        cleaning_summary = build_cleaned_ohlc(
            individual_raw,
            cleaning_root,
            expected_structural_flags=expected_structural_flags,
            expected_affected_indices=expected_affected_indices,
            apply_structural_corrections=apply_structural_corrections,
        )
        cleaned_root = cleaning_root / "individual_indices_data"

        (
            modeling_inventory_path,
            range_quality_path,
            inventory_summary,
            range_summary,
        ) = _prepare_frozen_gpt2_contract(
            raw_root,
            individual_raw,
            cleaned_root,
            contract_root,
        )

        catalogue_run = catalogue_root / "build"
        catalogue_run.mkdir(parents=True, exist_ok=False)
        transition_summary = write_global_transition_outputs(
            cleaned_root,
            modeling_inventory_path,
            range_quality_path,
            catalogue_run,
            apply_structural_corrections=False,
        )
        representative_path = catalogue_run / "representative_transition_catalogue.csv"

        stage_d_run = stage_d_root / "build"
        stage_d_run.mkdir(parents=True, exist_ok=False)
        stage_d_summary = write_global_stage_d_dataset(
            representative_path,
            modeling_inventory_path,
            stage_d_run,
        )

        cleaned = consolidate_cleaned_ohlc(
            cleaned_root,
            final_root / "cleaned_ohlc.csv.gz",
        )
        daily = build_daily_volatility(
            range_quality_path,
            final_root / "daily_volatility.csv.gz",
        )
        shutil.copy2(representative_path, final_root / "transition_catalogue.csv")
        shutil.copy2(stage_d_run / "sample_manifest.csv", final_root / "sample_manifest.csv")
        shutil.copy2(stage_d_run / "sequence_tensors.npz", final_root / "sequence_tensors.npz")
        shutil.copy2(cleaning_root / "row_corrections.csv", final_root / "row_corrections.csv")

        raw_acquisition_manifest = raw_root / "raw_acquisition_manifest.json"
        manifest: dict[str, object] = {
            "schema_version": 5,
            "dataset": "global_transition_dataset",
            "build_mode": (
                "corrected" if apply_structural_corrections else "parity_control_no_structural_removal"
            ),
            "temporary_parity_control": not apply_structural_corrections,
            "built_at_utc": utc_now(),
            "raw_root": str(raw_root),
            "raw_acquisition_manifest_sha256": (
                sha256_file(raw_acquisition_manifest)
                if raw_acquisition_manifest.is_file()
                else None
            ),
            "test_evaluated": False,
            "rules": {
                "interpolation": False,
                "forward_fill": False,
                "winsorization": False,
                "arbitrary_clipping": False,
                "synthetic_dates": False,
                "inventory_contract": "frozen_gpt2",
                "effective_start_contract": "frozen_gpt2",
                "raw_source_hashes_verified": True,
                "structural_bad_prints_detected": True,
                "structural_bad_prints_removed_before_volatility": apply_structural_corrections,
                "controls_rematched_after_correction": apply_structural_corrections,
            },
            "counts": {
                "cleaned_ohlc_rows": int(len(cleaned)),
                "daily_volatility_rows": int(len(daily)),
                "structural_detected_rows": int(cleaning_summary["structural_detected_rows"]),
                "structural_removed_rows": int(cleaning_summary["structural_removed_rows"]),
                "affected_indices": int(cleaning_summary["affected_indices"]),
                "transition_events": int(transition_summary["representative_market_events"]),
                "samples": int(stage_d_summary["total_samples"]),
                "positive_samples": int(stage_d_summary["positive_samples"]),
                "control_samples": int(stage_d_summary["negative_samples"]),
                "controls_per_positive": controls_per_positive,
            },
            "inventory_summary": inventory_summary,
            "range_summary": range_summary,
            "transition_summary": transition_summary,
            "stage_d_summary": stage_d_summary,
        }
        (final_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        audit = _audit_for_mode(
            final_root,
            controls_per_positive=controls_per_positive,
            apply_structural_corrections=apply_structural_corrections,
        )
        if not audit["passed"]:
            raise RuntimeError(
                "Candidate processed dataset failed validation: "
                + "; ".join(str(item) for item in audit["failures"])
            )

        manifest["files"] = _file_inventory(final_root)
        manifest["validation"] = audit
        (final_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        final_audit = _audit_for_mode(
            final_root,
            controls_per_positive=controls_per_positive,
            apply_structural_corrections=apply_structural_corrections,
        )
        if not final_audit["passed"]:
            raise RuntimeError(
                "Final candidate audit failed after manifest publication: "
                + "; ".join(str(item) for item in final_audit["failures"])
            )

        _promote_candidate(final_root, output_dir, force=force)

    return {
        "output_dir": str(output_dir),
        "build_mode": manifest["build_mode"],
        "temporary_parity_control": manifest["temporary_parity_control"],
        "files": _file_inventory(output_dir),
        "counts": manifest["counts"],
        "test_evaluated": False,
    }
