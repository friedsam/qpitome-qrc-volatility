from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from transition_forecasting.modeling.fold_selection import (
    DEVELOPMENT_FOLD_SPLITS,
    select_balanced_episode_rows,
    validate_fold_manifest_schema,
)
from transition_forecasting.qrc.rydberg_dense import RydbergDenseConfig, evolve_sequence


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    center = values.mean(axis=1, keepdims=True)
    scale = values.std(axis=1, keepdims=True)
    scale[scale < 1e-8] = 1.0
    result = np.clip((values - center) / scale, -4.0, 4.0)
    if not np.isfinite(result).all():
        raise ValueError("non-finite values remained after row standardization")
    return result


def _local_pattern(sequence: np.ndarray, n_atoms: int) -> np.ndarray:
    context = np.array(
        [sequence[-1], sequence.mean(), sequence[-5:].mean(), sequence.std()],
        dtype=float,
    )
    pattern = np.resize(context, n_atoms)
    return pattern / max(float(np.max(np.abs(pattern))), 1.0)


def _load_bad_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{path} has no sample_id column")
    return set(frame["sample_id"].astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fold-aware, quality-gated Rydberg chronology mechanism assay."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--contaminated-samples", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--max-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/transition_forecasting/qrc/fold_aware_rydberg_assay"),
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    validate_fold_manifest_schema(manifest)
    bad_ids = _load_bad_ids(args.contaminated_samples)

    config = RydbergDenseConfig(
        input_scale=0.60,
        nearest_interaction=0.75,
        step_duration=0.35,
    )
    probes = (args.sequence_length // 2, args.sequence_length)
    rng = np.random.default_rng(args.seed)
    feature_rows: list[dict[str, object]] = []
    selection_rows: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []
    channel_name: str | None = None

    for fold in args.folds:
        tensor_path = args.tensor_root / f"fold_{fold}" / "compact_tensor.npz"
        if not tensor_path.exists():
            raise FileNotFoundError(f"missing fold tensor: {tensor_path}")

        selected = select_balanced_episode_rows(
            manifest,
            fold=fold,
            lead=args.lead,
            max_per_class=args.max_per_class,
            excluded_sample_ids=bad_ids,
            splits=DEVELOPMENT_FOLD_SPLITS,
            seed=args.seed,
        )
        selected = selected.copy()
        selected["assay_fold"] = fold
        selection_rows.append(selected)

        with np.load(tensor_path, allow_pickle=True) as bundle:
            x = np.asarray(bundle["X"], dtype=float)
            ids = np.asarray(bundle["sample_id"], dtype=object).astype(str)
            valid = np.asarray(bundle["valid"], dtype=bool)
            names = [str(value) for value in np.asarray(bundle["channel_names"])]
        if args.channel >= x.shape[2]:
            raise ValueError(f"channel {args.channel} outside tensor shape {x.shape}")
        channel_name = names[args.channel]
        row_by_id = {sample_id: index for index, sample_id in enumerate(ids)}

        missing_ids = sorted(set(selected["sample_id"]) - set(row_by_id))
        if missing_ids:
            raise RuntimeError(
                f"fold {fold}: {len(missing_ids)} selected IDs absent from tensor"
            )

        for record in selected.itertuples(index=False):
            tensor_row = row_by_id[str(record.sample_id)]
            sequence = x[tensor_row, -args.sequence_length :, args.channel]
            if not valid[tensor_row] or not np.isfinite(sequence).all():
                skipped_rows.append(
                    {
                        "fold": fold,
                        "fold_split": record.fold_split,
                        "sample_id": str(record.sample_id),
                        "label": int(record.label),
                        "tensor_valid": bool(valid[tensor_row]),
                        "finite_window": bool(np.isfinite(sequence).all()),
                    }
                )
                continue

            sequence = _standardize_rows(sequence.reshape(1, -1))[0]
            local = _local_pattern(sequence, config.n_atoms)
            conditions = {
                "ordered": (sequence, True, False),
                "shuffled": (sequence[rng.permutation(len(sequence))], True, False),
                "reversed": (sequence[::-1], True, False),
                "reset": (sequence, True, True),
                "interaction_off": (sequence, False, False),
            }
            for condition, (values, interactions, reset_each_step) in conditions.items():
                features = evolve_sequence(
                    values,
                    local,
                    config,
                    probe_steps=probes,
                    interactions=interactions,
                    reset_each_step=reset_each_step,
                )
                for feature_index, value in enumerate(features):
                    feature_rows.append(
                        {
                            "fold": fold,
                            "fold_split": record.fold_split,
                            "sample_id": str(record.sample_id),
                            "episode_id": str(record.episode_id),
                            "label": int(record.label),
                            "condition": condition,
                            "feature_index": feature_index,
                            "value": float(value),
                        }
                    )

    features = pd.DataFrame(feature_rows)
    if features.empty:
        raise RuntimeError("no valid assay features were generated")
    selection = pd.concat(selection_rows, ignore_index=True)
    skipped = pd.DataFrame(skipped_rows)
    wide = features.pivot(
        index=[
            "fold",
            "fold_split",
            "sample_id",
            "episode_id",
            "label",
            "feature_index",
        ],
        columns="condition",
        values="value",
    ).reset_index()
    for condition in ("shuffled", "reversed", "reset", "interaction_off"):
        wide[f"abs_ordered_minus_{condition}"] = np.abs(
            wide["ordered"] - wide[condition]
        )

    summary_table = (
        wide.groupby(["fold", "fold_split", "label"], dropna=False)
        .agg(
            samples=("sample_id", "nunique"),
            ordered_feature_std=("ordered", "std"),
            ordered_vs_shuffled_mae=("abs_ordered_minus_shuffled", "mean"),
            ordered_vs_reversed_mae=("abs_ordered_minus_reversed", "mean"),
            ordered_vs_reset_mae=("abs_ordered_minus_reset", "mean"),
            ordered_vs_interaction_off_mae=(
                "abs_ordered_minus_interaction_off",
                "mean",
            ),
        )
        .reset_index()
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    selection.to_csv(outdir / "selected_samples.csv", index=False)
    if not skipped.empty:
        skipped.to_csv(outdir / "skipped_invalid_samples.csv", index=False)
    features.to_csv(outdir / "features_long.csv.gz", index=False)
    wide.to_csv(outdir / "paired_feature_differences.csv.gz", index=False)
    summary_table.to_csv(outdir / "summary_by_fold_split_label.csv", index=False)

    generated_pairs = features[["fold", "sample_id"]].drop_duplicates()
    summary = {
        "manifest": str(args.manifest),
        "tensor_root": str(args.tensor_root),
        "contaminated_samples": (
            str(args.contaminated_samples) if args.contaminated_samples else None
        ),
        "excluded_bad_sample_ids_available": len(bad_ids),
        "folds": args.folds,
        "lead": args.lead,
        "channel": args.channel,
        "channel_name": channel_name,
        "sequence_length": args.sequence_length,
        "probe_steps": list(probes),
        "config": config.__dict__,
        "selected_rows": int(len(selection)),
        "generated_fold_sample_pairs": int(len(generated_pairs)),
        "generated_globally_unique_sample_ids": int(features["sample_id"].nunique()),
        "skipped_invalid_rows": int(len(skipped)),
        "test_rows_used": int(features["fold_split"].eq("test").sum()),
        "selection_source": "transition_forecasting.modeling.fold_selection.select_balanced_episode_rows",
        "purpose": "fold-aware balanced mechanism confirmation; no forecasting claim",
    }
    if summary["test_rows_used"] != 0:
        raise RuntimeError("test rows entered the assay")

    expected_groups = {
        (fold, fold_split, label)
        for fold in args.folds
        for fold_split in DEVELOPMENT_FOLD_SPLITS
        for label in (0, 1)
    }
    observed_groups = set(
        summary_table[["fold", "fold_split", "label"]].itertuples(
            index=False, name=None
        )
    )
    missing_groups = sorted(expected_groups - observed_groups)
    if missing_groups:
        raise RuntimeError(f"missing fold/split/class output groups: {missing_groups}")

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"WROTE {outdir}")


if __name__ == "__main__":
    main()
