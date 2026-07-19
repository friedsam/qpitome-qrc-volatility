from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transition_forecasting.modeling.fold_selection import (
    DEVELOPMENT_FOLD_SPLITS,
    select_balanced_episode_rows,
    validate_fold_manifest_schema,
)
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.modeling.stage_e_fixed_spec_diagnostics import (
    horizon_mz_summary,
    mincer_zarnowitz,
)
from transition_forecasting.modeling.stage_e_sequence_models import har_predictions, metrics
from transition_forecasting.qrc.rydberg_dense import RydbergDenseConfig, evolve_sequence

CONDITIONS = ("ordered", "shuffled", "reset", "interaction_off")


def _load_bad_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"{path} has no sample_id column")
    return set(frame["sample_id"].astype(str))


def _standardize(sequence: np.ndarray) -> np.ndarray:
    center = float(np.mean(sequence))
    scale = float(np.std(sequence))
    if scale < 1e-8:
        scale = 1.0
    result = np.clip((sequence - center) / scale, -4.0, 4.0)
    if not np.isfinite(result).all():
        raise ValueError("non-finite sequence after standardization")
    return result


def _local_pattern(sequence: np.ndarray, n_atoms: int) -> np.ndarray:
    context = np.array(
        [sequence[-1], sequence.mean(), sequence[-5:].mean(), sequence.std()],
        dtype=float,
    )
    pattern = np.resize(context, n_atoms)
    return pattern / max(float(np.max(np.abs(pattern))), 1.0)


def _features_for_sequence(
    sequence: np.ndarray,
    *,
    config: RydbergDenseConfig,
    probes: tuple[int, ...],
    permutation: np.ndarray,
) -> dict[str, np.ndarray]:
    local = _local_pattern(sequence, config.n_atoms)
    settings = {
        "ordered": (sequence, True, False),
        "shuffled": (sequence[permutation], True, False),
        "reset": (sequence, True, True),
        "interaction_off": (sequence, False, False),
    }
    output: dict[str, np.ndarray] = {}
    for condition, (values, interactions, reset_each_step) in settings.items():
        output[condition] = evolve_sequence(
            values,
            local,
            config,
            probe_steps=probes,
            interactions=interactions,
            reset_each_step=reset_each_step,
        )
    return output


def _evaluate(
    *,
    fold: int,
    condition: str,
    alpha: float,
    y: np.ndarray,
    har: np.ndarray,
    residual: np.ndarray,
    states: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
) -> dict[str, float | int | str]:
    scaler = StandardScaler()
    state_train = scaler.fit_transform(states[train_mask])
    state_all = scaler.transform(states)
    readout = Ridge(alpha=float(alpha))
    readout.fit(state_train, residual[train_mask])
    prediction = har + readout.predict(state_all)
    qlike, rmse = metrics(y, prediction, val_mask)
    val_y = y[val_mask]
    val_prediction = prediction[val_mask]
    return {
        "fold": fold,
        "condition": condition,
        "alpha": float(alpha),
        "train_samples": int(train_mask.sum()),
        "val_samples": int(val_mask.sum()),
        "val_qlike": float(qlike),
        "val_rmse": float(rmse),
        **mincer_zarnowitz(val_y, val_prediction),
        **horizon_mz_summary(val_y, val_prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage E Rydberg residual-readout pilot using canonical classical fold logic and metrics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--contaminated-samples", type=Path, default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--lead", type=int, default=1)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--max-per-class", type=int, default=24)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("results/transition_forecasting/qrc/stage_e_rydberg_readout_pilot"),
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
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    channel_name: str | None = None

    for fold in args.folds:
        selected = select_balanced_episode_rows(
            manifest,
            fold=fold,
            lead=args.lead,
            max_per_class=args.max_per_class,
            excluded_sample_ids=bad_ids,
            splits=DEVELOPMENT_FOLD_SPLITS,
            seed=args.seed,
        )
        tensor_path = args.tensor_root / f"fold_{fold}" / "compact_tensor.npz"
        with np.load(tensor_path, allow_pickle=True) as bundle:
            x = np.asarray(bundle["X"], dtype=float)
            ids = np.asarray(bundle["sample_id"], dtype=object).astype(str)
            valid = np.asarray(bundle["valid"], dtype=bool)
            names = [str(value) for value in np.asarray(bundle["channel_names"])]
        channel_name = names[args.channel]
        row_by_id = {sample_id: index for index, sample_id in enumerate(ids)}

        state_by_condition: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
        retained_rows: list[pd.Series] = []
        for _, row in selected.iterrows():
            sample_id = str(row["sample_id"])
            tensor_row = row_by_id.get(sample_id)
            if tensor_row is None:
                raise RuntimeError(f"fold {fold}: selected sample absent from tensor: {sample_id}")
            sequence = x[tensor_row, -args.sequence_length :, args.channel]
            if not valid[tensor_row] or not np.isfinite(sequence).all():
                skipped_rows.append(
                    {
                        "fold": fold,
                        "fold_split": row["fold_split"],
                        "sample_id": sample_id,
                        "label": int(row["label"]),
                    }
                )
                continue
            sequence = _standardize(sequence)
            rng = np.random.default_rng(args.seed + fold * 1_000_003 + tensor_row)
            generated = _features_for_sequence(
                sequence,
                config=config,
                probes=probes,
                permutation=rng.permutation(len(sequence)),
            )
            for condition in CONDITIONS:
                state_by_condition[condition].append(generated[condition])
            retained_rows.append(row)

        frame = pd.DataFrame(retained_rows).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"fold {fold}: no valid selected samples")
        train_mask = frame["fold_split"].eq("train").to_numpy()
        val_mask = frame["fold_split"].eq("val").to_numpy()
        if not train_mask.any() or not val_mask.any():
            raise RuntimeError(f"fold {fold}: missing train or validation samples after tensor validity filtering")
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        har = har_predictions(frame, y, train_mask)
        residual = y - har

        har_qlike, har_rmse = metrics(y, har, val_mask)
        result_rows.append(
            {
                "fold": fold,
                "condition": "har_only",
                "alpha": np.nan,
                "train_samples": int(train_mask.sum()),
                "val_samples": int(val_mask.sum()),
                "val_qlike": float(har_qlike),
                "val_rmse": float(har_rmse),
                **mincer_zarnowitz(y[val_mask], har[val_mask]),
                **horizon_mz_summary(y[val_mask], har[val_mask]),
            }
        )

        for condition in CONDITIONS:
            states = np.vstack(state_by_condition[condition])
            for alpha in args.alphas:
                result_rows.append(
                    _evaluate(
                        fold=fold,
                        condition=condition,
                        alpha=alpha,
                        y=y,
                        har=har,
                        residual=residual,
                        states=states,
                        train_mask=train_mask,
                        val_mask=val_mask,
                    )
                )
            for row_index, sample in frame.iterrows():
                prediction_rows.append(
                    {
                        "fold": fold,
                        "fold_split": sample["fold_split"],
                        "sample_id": sample["sample_id"],
                        "label": int(sample["label"]),
                    }
                )

    results = pd.DataFrame(result_rows)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = args.out_root / run_id
    outdir.mkdir(parents=True, exist_ok=False)
    results.to_csv(outdir / "fold_results.csv", index=False)
    pd.DataFrame(prediction_rows).drop_duplicates().to_csv(
        outdir / "retained_samples.csv", index=False
    )
    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(outdir / "skipped_invalid_samples.csv", index=False)

    best = (
        results.loc[~results["condition"].eq("har_only")]
        .sort_values(["fold", "val_qlike", "val_rmse"])
        .groupby("fold", as_index=False)
        .head(1)
    )
    best.to_csv(outdir / "best_per_fold.csv", index=False)
    summary = {
        "manifest": str(args.manifest),
        "tensor_root": str(args.tensor_root),
        "folds": args.folds,
        "selection_source": "transition_forecasting.modeling.fold_selection.select_balanced_episode_rows",
        "har_source": "transition_forecasting.modeling.stage_e_sequence_models.har_predictions",
        "metrics_source": "transition_forecasting.modeling.stage_e_sequence_models.metrics",
        "channel": args.channel,
        "channel_name": channel_name,
        "sequence_length": args.sequence_length,
        "probe_steps": list(probes),
        "max_per_class": args.max_per_class,
        "conditions": list(CONDITIONS),
        "alphas": args.alphas,
        "config": config.__dict__,
        "test_rows_used": 0,
        "status": "pilot_only",
        "known_limitation": "HAR training residuals use the existing in-sample classical implementation; prequential/OoF correction remains pending",
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results.sort_values(["fold", "condition", "alpha"]).to_string(index=False))
    print("\nBEST PER FOLD")
    print(best.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"WROTE {outdir}")


if __name__ == "__main__":
    main()
