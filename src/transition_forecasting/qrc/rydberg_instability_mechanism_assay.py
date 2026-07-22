from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.runs import begin_run
from transition_forecasting.modeling.stage_e_classical_baselines import TARGET_COLUMNS
from transition_forecasting.qrc.instability_mechanism_tools import (
    OBSERVABLE_FAMILIES,
    InstabilityMechanismAssayConfig,
    case_name,
    case_specs,
    evaluate_individual_pcs,
    evaluate_pls,
    family_diagnostics,
    family_indices,
    reorder_level,
)
from transition_forecasting.qrc.representation_candidates import (
    CandidateFeatureConfig,
    build_candidate_sequences,
    fit_channel_scaler,
    transform_candidate_sequences,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _evaluate_feature_matrix,
    _fit_har,
    _metric_payload,
    _prequential_har_residuals,
)
from transition_forecasting.qrc.rydberg_representation_screen import (
    _select_rows_for_fold,
)
from transition_forecasting.qrc.temporal_rydberg_chain import (
    TemporalRydbergChainConfig,
    build_temporal_rydberg_chain_features,
)
from transition_forecasting.qrc.temporal_rydberg_chain_artifacts import (
    feature_names_from_metadata,
)
from transition_forecasting.qrc.temporal_rydberg_chain_experiment import (
    extract_level_windows,
    load_rolling_fold_dataset,
    resolve_level_channel,
)


ASSAY_REPRESENTATION = "level_instability"


def _decorate(
    rows: list[dict[str, object]],
    *,
    case: str,
    control: str,
    scale: float,
    family: str,
) -> None:
    for row in rows:
        row.update(
            {
                "case": case,
                "control": control,
                "interaction_scale": scale,
                "observable_family": family,
                "analysis": "pca_prefix",
            }
        )


def _fixed_prediction(
    features: np.ndarray,
    names: tuple[str, ...],
    *,
    family: str,
    components: int,
    alpha: float,
    y: np.ndarray,
    har: np.ndarray,
    residuals: np.ndarray,
    train_mask: np.ndarray,
    residual_mask: np.ndarray,
) -> np.ndarray:
    matrix = features[:, family_indices(names, family)]
    scaler = StandardScaler().fit(matrix[train_mask])
    scaled = scaler.transform(matrix)
    count = min(components, int(train_mask.sum()), matrix.shape[1])
    if count < matrix.shape[1]:
        pca = PCA(n_components=count, svd_solver="full").fit(
            scaled[train_mask]
        )
        transformed = pca.transform(scaled)
    else:
        transformed = scaled
    model = Ridge(alpha=alpha).fit(
        transformed[residual_mask],
        residuals[residual_mask],
    )
    return har + model.predict(transformed)


def _prediction_frame(
    frame: pd.DataFrame,
    y: np.ndarray,
    prediction: np.ndarray,
    *,
    fold: int,
    case: str,
    control: str,
    scale: float,
    family: str,
    components: int,
) -> pd.DataFrame:
    horizons = y.shape[1]
    return pd.DataFrame(
        {
            "fold": np.repeat(fold, len(frame) * horizons),
            "sample_id": np.repeat(
                frame["sample_id"].astype(str), horizons
            ),
            "fold_split": np.repeat(
                frame["fold_split"].astype(str), horizons
            ),
            "lead": np.repeat(frame["lead"].astype(int), horizons),
            "label": np.repeat(frame["label"].astype(int), horizons),
            "episode_id": np.repeat(
                frame["episode_id"].astype(str), horizons
            ),
            "origin_date": np.repeat(
                frame["origin_date"].astype(str), horizons
            ),
            "case": np.repeat(case, len(frame) * horizons),
            "control": np.repeat(control, len(frame) * horizons),
            "interaction_scale": np.repeat(
                scale, len(frame) * horizons
            ),
            "observable_family": np.repeat(
                family, len(frame) * horizons
            ),
            "components": np.repeat(
                components, len(frame) * horizons
            ),
            "horizon": np.tile(
                np.arange(1, horizons + 1), len(frame)
            ),
            "y_true": y.reshape(-1),
            "y_pred": prediction.reshape(-1),
        }
    )


def _write_archive(
    path: Path,
    blocks: list[dict[str, object]],
    names: tuple[str, ...],
) -> None:
    arrays = {
        "feature_matrix": np.concatenate(
            [np.asarray(x["features"]) for x in blocks]
        ),
        "feature_names": np.asarray(names, dtype=str),
        "source_level": np.concatenate(
            [np.asarray(x["source_level"]) for x in blocks]
        ),
        "encoded_sequence": np.concatenate(
            [np.asarray(x["encoded_sequence"]) for x in blocks]
        ),
        "target_path": np.concatenate(
            [np.asarray(x["targets"]) for x in blocks]
        ),
        "har_prediction_path": np.concatenate(
            [np.asarray(x["har"]) for x in blocks]
        ),
        "prequential_residual_path": np.concatenate(
            [np.asarray(x["residuals"]) for x in blocks]
        ),
        "prequential_residual_valid": np.concatenate(
            [np.asarray(x["residual_valid"]) for x in blocks]
        ),
    }
    for key in (
        "fold",
        "sample_id",
        "fold_split",
        "lead",
        "label",
        "episode_id",
        "origin_date",
    ):
        values = np.concatenate(
            [np.asarray(x[key]) for x in blocks]
        )
        arrays[key] = (
            values.astype(int)
            if key in {"fold", "lead", "label"}
            else values.astype(str)
        )
    for key in ("case", "control", "interaction_scale"):
        arrays[key] = np.concatenate(
            [
                np.repeat(x[key], len(x["sample_id"]))
                for x in blocks
            ]
        )
    np.savez_compressed(path, **arrays)


def _complete_summary(
    metrics: pd.DataFrame,
    folds: int,
) -> pd.DataFrame:
    keys = [
        "case",
        "control",
        "interaction_scale",
        "analysis",
        "observable_family",
        "readout_mode",
        "components",
        "alpha",
    ]
    out = metrics.groupby(
        keys,
        dropna=False,
        as_index=False,
    ).agg(
        mean_val_qlike=("val_qlike", "mean"),
        mean_val_rmse=("val_rmse", "mean"),
        mean_val_mz_r2=("val_mz_r2", "mean"),
        folds=("fold", "nunique"),
    )
    return out.loc[out["folds"].eq(folds)].copy()


def _write_plots(
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    run_dir: Path,
) -> None:
    sweep = summary.loc[
        summary["control"].eq("ordered")
        & summary["analysis"].eq("pca_prefix")
        & summary["readout_mode"].eq(
            "prequential_har_residual"
        )
        & summary["observable_family"].isin(
            ["all", "occupation", "connected"]
        )
    ]
    if not sweep.empty:
        best = (
            sweep.sort_values("mean_val_qlike")
            .groupby(
                ["observable_family", "interaction_scale"],
                as_index=False,
            )
            .head(1)
        )
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        for family, group in best.groupby(
            "observable_family"
        ):
            ax.plot(
                group["interaction_scale"],
                group["mean_val_qlike"],
                marker="o",
                label=family,
            )
        ax.set(
            xlabel="Interaction scale",
            ylabel="Mean validation QLIKE",
            title="Instability QRC interaction sweep",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            run_dir / "interaction_sweep_qlike.png",
            dpi=180,
        )
        plt.close(fig)

    l5 = predictions.loc[
        predictions["fold_split"].eq("val")
        & predictions["lead"].eq(5)
        & predictions["label"].eq(1)
    ].copy()
    rows = []
    keys = [
        "case",
        "control",
        "interaction_scale",
        "observable_family",
        "components",
        "horizon",
    ]
    for key, group in l5.groupby(keys, dropna=False):
        payload = dict(zip(keys, key))
        mask = np.ones(len(group), dtype=bool)
        payload.update(
            _metric_payload(
                group["y_true"].to_numpy()[:, None],
                group["y_pred"].to_numpy()[:, None],
                mask,
            )
        )
        rows.append(payload)
    horizon = pd.DataFrame(rows)
    horizon.to_csv(
        run_dir / "l5_horizon_metrics.csv",
        index=False,
    )
    if horizon.empty:
        return

    scores = (
        horizon.groupby(keys[:-1], as_index=False)
        .agg(score=("qlike", "mean"))
        .sort_values("score")
    )
    chosen = []
    for mask in (
        scores["control"].eq("ordered"),
        ~scores["control"].isin(["ordered", "har"]),
    ):
        if mask.any():
            chosen.append(scores.loc[mask].iloc[0])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    actual = l5.groupby(
        "horizon", as_index=False
    )["y_true"].mean()
    ax.plot(
        actual["horizon"],
        actual["y_true"],
        marker="o",
        label="Actual",
    )
    har_frame = l5.loc[l5["control"].eq("har")]
    if not har_frame.empty:
        mean = har_frame.groupby(
            "horizon", as_index=False
        )["y_pred"].mean()
        ax.plot(
            mean["horizon"],
            mean["y_pred"],
            marker="o",
            label="HAR",
        )
    for row in chosen:
        mask = (
            l5["case"].eq(row["case"])
            & l5["observable_family"].eq(
                row["observable_family"]
            )
        )
        mean = l5.loc[mask].groupby(
            "horizon", as_index=False
        )["y_pred"].mean()
        ax.plot(
            mean["horizon"],
            mean["y_pred"],
            marker="o",
            label=(
                f"{row['control']} "
                f"V={row['interaction_scale']} "
                f"{row['observable_family']}"
            ),
        )
    ax.axvline(5, linestyle="--", linewidth=1)
    ax.set(
        xlabel="Forecast horizon",
        ylabel="Log volatility",
        title="L5 transition forecast anatomy",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        run_dir / "l5_mean_forecast_overlay.png",
        dpi=180,
    )
    plt.close(fig)


def run_rydberg_instability_mechanism_assay(
    *,
    fold_dir: Path,
    results_root: Path,
    assay: InstabilityMechanismAssayConfig,
    candidate_features: CandidateFeatureConfig,
    reservoir: TemporalRydbergChainConfig,
    run_id: str | None = None,
) -> Path:
    assay.validate()
    candidate_features.validate()
    reservoir.validate()
    run_dir = begin_run(
        results_root,
        {
            "fold_dir": str(fold_dir),
            "assay": assay.to_dict(),
            "candidate_features": candidate_features.to_dict(),
            "reservoir": reservoir.to_dict(),
        },
        run_id=run_id,
    )
    dataset = load_rolling_fold_dataset(fold_dir)
    channel = resolve_level_channel(
        dataset,
        name=assay.level_channel_name,
        fallback=assay.fallback_level_channel,
    )
    metrics: list[dict[str, object]] = []
    groups: list[dict[str, object]] = []
    pc_metrics: list[dict[str, object]] = []
    pc_diag: list[dict[str, object]] = []
    pls_metrics: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    archive: list[dict[str, object]] = []
    names: tuple[str, ...] | None = None

    for fold in assay.folds:
        frame = _select_rows_for_fold(
            dataset.manifest,
            fold=fold,
            leads=assay.leads,
            max_per_class=assay.max_per_class,
            seed=assay.seed,
        )
        rows = frame["_tensor_row"].to_numpy(dtype=int)
        source = extract_level_windows(
            dataset,
            rows,
            sequence_length=assay.sequence_length,
            level_channel=channel,
        )
        y = frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
        train = frame["fold_split"].eq("train").to_numpy()
        val = frame["fold_split"].eq("val").to_numpy()
        har = _fit_har(frame, y, train)
        residuals, residual_mask = _prequential_har_residuals(
            frame,
            y,
            train,
            blocks=assay.prequential_blocks,
        )
        predictions.append(
            _prediction_frame(
                frame,
                y,
                har,
                fold=fold,
                case="har",
                control="har",
                scale=np.nan,
                family="har",
                components=0,
            )
        )
        metrics.append(
            {
                "fold": fold,
                "representation": "har",
                "second_channel": "har",
                "model_family": "har",
                "seed": 0,
                "readout_mode": "direct",
                "components": 0,
                "feature_width": 3,
                "explained_variance": np.nan,
                "alpha": 100.0,
                "train_rows_used": int(train.sum()),
                "val_rows": int(val.sum()),
                "case": "har",
                "control": "har",
                "interaction_scale": np.nan,
                "observable_family": "har",
                "analysis": "har",
                **{
                    f"val_{key}": value
                    for key, value in _metric_payload(
                        y,
                        har,
                        val,
                    ).items()
                },
            }
        )
        ordered_raw = build_candidate_sequences(
            source,
            "level_instability",
            candidate_features,
        )
        scaler = fit_channel_scaler(ordered_raw, train)

        for control, scale in case_specs(assay):
            control_seed = (
                assay.seed
                + fold * 1_000_003
                + sum(
                    (i + 1) * ord(c)
                    for i, c in enumerate(control)
                )
            )
            reordered = reorder_level(
                source,
                control,
                seed=control_seed,
                block_size=assay.block_size,
            )
            raw = build_candidate_sequences(
                reordered,
                "level_instability",
                candidate_features,
            )
            encoded = transform_candidate_sequences(
                raw,
                scaler,
            )
            current = replace(
                reservoir,
                c6_rad_um6_per_us=(
                    reservoir.c6_rad_um6_per_us * scale
                ),
            )
            features, metadata = (
                build_temporal_rydberg_chain_features(
                    encoded,
                    current,
                    condition=(
                        "reset"
                        if control == "reset"
                        else "ordered"
                    ),
                )
            )
            current_names = feature_names_from_metadata(
                metadata
            )
            if names is None:
                names = current_names
            elif names != current_names:
                raise RuntimeError(
                    "feature contract changed inside assay"
                )
            case = case_name(control, scale)
            archive.append(
                {
                    "features": features,
                    "source_level": source,
                    "encoded_sequence": raw,
                    "targets": y,
                    "har": har,
                    "residuals": residuals,
                    "residual_valid": residual_mask,
                    "fold": frame["fold"],
                    "sample_id": frame["sample_id"],
                    "fold_split": frame["fold_split"],
                    "lead": frame["lead"],
                    "label": frame["label"],
                    "episode_id": frame["episode_id"],
                    "origin_date": frame["origin_date"],
                    "case": case,
                    "control": control,
                    "interaction_scale": scale,
                }
            )
            diagnostics.extend(
                family_diagnostics(
                    features,
                    current_names,
                    train_mask=train,
                    fold=fold,
                    case=case,
                    control=control,
                    scale=scale,
                    probes=tuple(metadata["probe_steps"]),
                )
            )
            families = (
                OBSERVABLE_FAMILIES
                + tuple(
                    f"probe_{x}"
                    for x in metadata["probe_steps"]
                )
            )
            for family in families:
                idx = family_indices(
                    current_names,
                    family,
                )
                rows_out, groups_out = _evaluate_feature_matrix(
                    feature_matrix=features[:, idx],
                    frame=frame,
                    y=y,
                    har=har,
                    residuals=residuals,
                    residual_train_mask=residual_mask,
                    train_mask=train,
                    val_mask=val,
                    fold=fold,
                    # The shared evaluator expects one of the
                    # canonical input-representation names for
                    # SECOND_CHANNEL_NAMES. The case label is
                    # attached separately by _decorate().
                    representation=ASSAY_REPRESENTATION,
                    model_family="qrc",
                    seed=0,
                    alphas=assay.alphas,
                    components=assay.pca_components,
                    readout_modes=assay.readout_modes,
                )
                _decorate(
                    rows_out,
                    case=case,
                    control=control,
                    scale=scale,
                    family=family,
                )
                _decorate(
                    groups_out,
                    case=case,
                    control=control,
                    scale=scale,
                    family=family,
                )
                metrics.extend(rows_out)
                groups.extend(groups_out)

            rows_out, diag_out = evaluate_individual_pcs(
                features,
                y=y,
                har=har,
                residuals=residuals,
                train_mask=train,
                residual_mask=residual_mask,
                val_mask=val,
                fold=fold,
                case=case,
                control=control,
                scale=scale,
                alphas=assay.alphas,
                readout_modes=assay.readout_modes,
                pc_count=assay.individual_pc_count,
            )
            pc_metrics.extend(rows_out)
            pc_diag.extend(diag_out)
            pls_metrics.extend(
                evaluate_pls(
                    features,
                    y=y,
                    har=har,
                    residuals=residuals,
                    train_mask=train,
                    residual_mask=residual_mask,
                    val_mask=val,
                    fold=fold,
                    case=case,
                    control=control,
                    scale=scale,
                    components=assay.pls_components,
                    readout_modes=assay.readout_modes,
                )
            )
            for family, count in (
                ("occupation", 4),
                ("connected", 4),
                ("all", 8),
            ):
                prediction = _fixed_prediction(
                    features,
                    current_names,
                    family=family,
                    components=count,
                    alpha=1000.0,
                    y=y,
                    har=har,
                    residuals=residuals,
                    train_mask=train,
                    residual_mask=residual_mask,
                )
                predictions.append(
                    _prediction_frame(
                        frame,
                        y,
                        prediction,
                        fold=fold,
                        case=case,
                        control=control,
                        scale=scale,
                        family=family,
                        components=count,
                    )
                )

    if names is None:
        raise RuntimeError("assay produced no QRC features")

    metrics_frame = pd.DataFrame(metrics)
    groups_frame = pd.DataFrame(groups)
    metrics_frame.to_csv(
        run_dir / "model_metrics.csv",
        index=False,
    )
    groups_frame.to_csv(
        run_dir / "group_metrics.csv",
        index=False,
    )
    pd.DataFrame(pc_metrics).to_csv(
        run_dir / "individual_pc_metrics.csv",
        index=False,
    )
    pd.DataFrame(pc_diag).to_csv(
        run_dir / "pc_diagnostics.csv",
        index=False,
    )
    pd.DataFrame(pls_metrics).to_csv(
        run_dir / "pls_metrics.csv",
        index=False,
    )
    pd.DataFrame(diagnostics).to_csv(
        run_dir / "feature_diagnostics.csv",
        index=False,
    )
    _write_archive(
        run_dir / "qrc_features.npz",
        archive,
        names,
    )
    summary = _complete_summary(
        metrics_frame,
        len(assay.folds),
    )
    summary.to_csv(
        run_dir / "complete_fold_summary.csv",
        index=False,
    )
    pred = pd.concat(predictions, ignore_index=True)
    pred.to_csv(
        run_dir / "diagnostic_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    _write_plots(summary, pred, run_dir)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": (
                    "development_instability_mechanism_assay"
                ),
                "test_rows_used": 0,
                "cases": [
                    {
                        "control": control,
                        "interaction_scale": scale,
                        "case": case_name(
                            control,
                            scale,
                        ),
                    }
                    for control, scale in case_specs(assay)
                ],
                "known_limitations": [
                    (
                        "Development folds only; no untouched "
                        "test rows are used."
                    ),
                    (
                        "C6 scaling is a simulation diagnostic; "
                        "hardware realization would change geometry."
                    ),
                    (
                        "PCA, PLS, alpha, family, and interaction "
                        "choices are inspected on development folds."
                    ),
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir
