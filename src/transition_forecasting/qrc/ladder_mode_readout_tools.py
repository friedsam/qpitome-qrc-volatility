from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from transition_forecasting.qrc.frozen_chain_readout_tools import (
    chronological_inner_split,
)
from transition_forecasting.qrc.representation_screen_analysis import (
    _metric_payload,
)

ReadoutFamily = Literal[
    "occupation_pca4",
    "symmetric_modes",
    "antisymmetric_modes",
    "compact_modes",
    "all_modes",
]

READOUT_FAMILIES: tuple[ReadoutFamily, ...] = (
    "occupation_pca4",
    "symmetric_modes",
    "antisymmetric_modes",
    "compact_modes",
    "all_modes",
)

ORDERED_LADDER_CASES: tuple[str, ...] = (
    "ladder_ordered_0p75",
    "ladder_ordered_1p00",
    "ladder_ordered_1p25",
)

REFERENCE_CASES: tuple[str, ...] = (
    "chain_interaction_off",
    "chain_interacting_1p00",
)

CONTROL_CASES: tuple[str, ...] = (
    "ladder_reset_1p00",
    "ladder_shuffled_1p00",
    "ladder_reversed_1p00",
    "ladder_block_shuffled_1p00",
)

MODE_NAMES: tuple[str, ...] = (
    "symmetric_constant",
    "symmetric_gradient",
    "symmetric_curvature",
    "antisymmetric_constant",
    "antisymmetric_gradient",
    "antisymmetric_curvature",
)

FAMILY_MODE_INDICES: dict[str, tuple[int, ...]] = {
    "symmetric_modes": (0, 1, 2),
    "antisymmetric_modes": (3, 4, 5),
    "compact_modes": (0, 1, 3, 4),
    "all_modes": (0, 1, 2, 3, 4, 5),
}


@dataclass(frozen=True)
class LadderModeReadoutConfig:
    ordered_ladder_cases: tuple[str, ...] = ORDERED_LADDER_CASES
    reference_cases: tuple[str, ...] = REFERENCE_CASES
    control_cases: tuple[str, ...] = CONTROL_CASES
    readout_families: tuple[ReadoutFamily, ...] = READOUT_FAMILIES
    folds: tuple[int, ...] = (1, 2, 3)
    inner_holdout_fraction: float = 0.25
    global_lambdas: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
    )
    ridge_alpha: float = 100.0
    pca_components: int = 4

    def validate(self) -> None:
        if not self.ordered_ladder_cases:
            raise ValueError("ordered_ladder_cases cannot be empty")
        if not self.reference_cases:
            raise ValueError("reference_cases cannot be empty")
        if not self.folds:
            raise ValueError("folds cannot be empty")
        if not 0.1 <= self.inner_holdout_fraction <= 0.5:
            raise ValueError("inner_holdout_fraction must lie in [0.1, 0.5]")
        if not self.global_lambdas or any(value < 0 for value in self.global_lambdas):
            raise ValueError("global_lambdas must be nonnegative")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.pca_components < 1:
            raise ValueError("pca_components must be positive")
        unsupported = set(self.readout_families).difference(READOUT_FAMILIES)
        if unsupported:
            raise ValueError(f"unsupported readout families: {sorted(unsupported)}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureArchive:
    feature_matrix: np.ndarray
    feature_names: tuple[str, ...]
    target_path: np.ndarray
    har_prediction_path: np.ndarray
    prequential_residual_path: np.ndarray
    prequential_residual_valid: np.ndarray
    fold: np.ndarray
    sample_id: np.ndarray
    fold_split: np.ndarray
    lead: np.ndarray
    label: np.ndarray
    episode_id: np.ndarray
    origin_date: np.ndarray
    case: np.ndarray
    architecture: np.ndarray
    control: np.ndarray
    interaction_scale: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.fold))


def load_architecture_archive(path: Path) -> ArchitectureArchive:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "feature_matrix",
        "feature_names",
        "target_path",
        "har_prediction_path",
        "prequential_residual_path",
        "prequential_residual_valid",
        "fold",
        "sample_id",
        "fold_split",
        "lead",
        "label",
        "episode_id",
        "origin_date",
        "case",
        "architecture",
        "control",
        "interaction_scale",
    }
    with np.load(path, allow_pickle=False) as bundle:
        missing = required.difference(bundle.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        archive = ArchitectureArchive(
            feature_matrix=np.asarray(bundle["feature_matrix"], dtype=float),
            feature_names=tuple(
                str(value) for value in np.asarray(bundle["feature_names"]).astype(str)
            ),
            target_path=np.asarray(bundle["target_path"], dtype=float),
            har_prediction_path=np.asarray(
                bundle["har_prediction_path"], dtype=float
            ),
            prequential_residual_path=np.asarray(
                bundle["prequential_residual_path"], dtype=float
            ),
            prequential_residual_valid=np.asarray(
                bundle["prequential_residual_valid"], dtype=bool
            ),
            fold=np.asarray(bundle["fold"], dtype=int),
            sample_id=np.asarray(bundle["sample_id"]).astype(str),
            fold_split=np.asarray(bundle["fold_split"]).astype(str),
            lead=np.asarray(bundle["lead"], dtype=int),
            label=np.asarray(bundle["label"], dtype=int),
            episode_id=np.asarray(bundle["episode_id"]).astype(str),
            origin_date=np.asarray(bundle["origin_date"]).astype(str),
            case=np.asarray(bundle["case"]).astype(str),
            architecture=np.asarray(bundle["architecture"]).astype(str),
            control=np.asarray(bundle["control"]).astype(str),
            interaction_scale=np.asarray(
                bundle["interaction_scale"], dtype=float
            ),
        )
    aligned = {
        "feature_matrix": len(archive.feature_matrix),
        "target_path": len(archive.target_path),
        "har_prediction_path": len(archive.har_prediction_path),
        "prequential_residual_path": len(archive.prequential_residual_path),
        "prequential_residual_valid": len(archive.prequential_residual_valid),
        "sample_id": len(archive.sample_id),
        "fold_split": len(archive.fold_split),
        "lead": len(archive.lead),
        "label": len(archive.label),
        "episode_id": len(archive.episode_id),
        "origin_date": len(archive.origin_date),
        "case": len(archive.case),
        "architecture": len(archive.architecture),
        "control": len(archive.control),
        "interaction_scale": len(archive.interaction_scale),
    }
    if any(length != archive.rows for length in aligned.values()):
        raise ValueError(
            f"archive arrays are not aligned: rows={archive.rows}, arrays={aligned}"
        )
    if archive.feature_matrix.shape[1] != len(archive.feature_names):
        raise ValueError("feature_names do not match feature_matrix width")
    if archive.target_path.shape != archive.har_prediction_path.shape:
        raise ValueError("target and HAR paths have different shapes")
    if archive.target_path.shape != archive.prequential_residual_path.shape:
        raise ValueError("target and residual paths have different shapes")
    if not np.isfinite(archive.feature_matrix).all():
        raise ValueError("feature_matrix contains non-finite values")
    if len(np.unique(archive.case)) != 1:
        raise ValueError(f"{path} contains more than one case label")
    return archive


def validate_archive_alignment(
    archives: dict[str, ArchitectureArchive],
) -> None:
    if not archives:
        raise ValueError("archives cannot be empty")
    reference_name = next(iter(archives))
    reference = archives[reference_name]
    exact_fields = (
        "fold",
        "sample_id",
        "fold_split",
        "lead",
        "label",
        "episode_id",
        "origin_date",
        "prequential_residual_valid",
    )
    numeric_fields = (
        "target_path",
        "har_prediction_path",
        "prequential_residual_path",
    )
    for name, archive in archives.items():
        if archive.rows != reference.rows:
            raise ValueError(
                f"case {name} has {archive.rows} rows; expected {reference.rows}"
            )
        for field in exact_fields:
            if not np.array_equal(getattr(reference, field), getattr(archive, field)):
                raise ValueError(
                    f"case {name} is not aligned with {reference_name}: {field}"
                )
        for field in numeric_fields:
            if not np.allclose(
                getattr(reference, field),
                getattr(archive, field),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError(
                    f"case {name} is not aligned with {reference_name}: {field}"
                )


def occupation_probe_matrices(
    archive: ArchitectureArchive,
) -> tuple[tuple[int, ...], dict[int, np.ndarray]]:
    names = archive.feature_names
    probes = sorted(
        {
            int(name.split("_")[1])
            for name in names
            if "occupation_site_" in name
        }
    )
    if not probes:
        raise ValueError("archive contains no occupation observables")
    matrices: dict[int, np.ndarray] = {}
    for probe in probes:
        indices = []
        for site in range(6):
            name = f"probe_{probe}_occupation_site_{site}"
            try:
                indices.append(names.index(name))
            except ValueError as exc:
                raise ValueError(
                    f"archive is missing required occupation feature {name!r}"
                ) from exc
        matrices[int(probe)] = archive.feature_matrix[:, indices]
    return tuple(int(value) for value in probes), matrices


def ladder_mode_weights() -> np.ndarray:
    row_symmetric = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
    row_antisymmetric = np.asarray([1.0, -1.0]) / np.sqrt(2.0)
    column_constant = np.asarray([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    column_gradient = np.asarray([-1.0, 0.0, 1.0]) / np.sqrt(2.0)
    column_curvature = np.asarray([1.0, -2.0, 1.0]) / np.sqrt(6.0)
    row_vectors = (row_symmetric, row_antisymmetric)
    column_vectors = (
        column_constant,
        column_gradient,
        column_curvature,
    )
    weights = [
        np.outer(row, column).reshape(-1)
        for row in row_vectors
        for column in column_vectors
    ]
    matrix = np.column_stack(weights)
    gram = matrix.T @ matrix
    if not np.allclose(gram, np.eye(6), atol=1e-12, rtol=0.0):
        raise RuntimeError("ladder spatial mode weights are not orthonormal")
    return matrix


def build_readout_matrix(
    archive: ArchitectureArchive,
    family: ReadoutFamily,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    probes, occupations = occupation_probe_matrices(archive)
    if family == "occupation_pca4":
        matrix = np.concatenate([occupations[probe] for probe in probes], axis=1)
        names = tuple(
            f"probe_{probe}_occupation_site_{site}"
            for probe in probes
            for site in range(6)
        )
        return matrix, names, "pca"

    if family not in FAMILY_MODE_INDICES:
        raise ValueError(f"unsupported readout family: {family}")
    weights = ladder_mode_weights()
    selected = FAMILY_MODE_INDICES[family]
    blocks = []
    names = []
    for probe in probes:
        transformed = occupations[probe] @ weights
        blocks.append(transformed[:, selected])
        names.extend(
            f"probe_{probe}_{MODE_NAMES[index]}"
            for index in selected
        )
    return np.concatenate(blocks, axis=1), tuple(names), "direct"


def _fit_correction(
    matrix: np.ndarray,
    residuals: np.ndarray,
    fit_mask: np.ndarray,
    *,
    transform: str,
    pca_components: int,
    ridge_alpha: float,
) -> np.ndarray:
    features = np.asarray(matrix, dtype=float)
    targets = np.asarray(residuals, dtype=float)
    fit = np.asarray(fit_mask, dtype=bool)
    if features.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if targets.ndim != 2 or len(targets) != len(features):
        raise ValueError("residuals must align with the feature matrix")
    if fit.shape != (len(features),) or not fit.any():
        raise ValueError("fit_mask must select aligned rows")
    scaler = StandardScaler().fit(features[fit])
    transformed = scaler.transform(features)
    if transform == "pca":
        maximum = min(int(fit.sum()), transformed.shape[1])
        if pca_components > maximum:
            raise ValueError(
                f"PCA components {pca_components} exceed fit rank {maximum}"
            )
        pca = PCA(
            n_components=int(pca_components),
            svd_solver="full",
        ).fit(transformed[fit])
        design = pca.transform(transformed)
    elif transform == "direct":
        design = transformed
    else:
        raise ValueError(f"unsupported transform: {transform}")
    model = Ridge(alpha=float(ridge_alpha)).fit(design[fit], targets[fit])
    return np.asarray(model.predict(design), dtype=float)


def choose_global_lambda(
    correction: np.ndarray,
    *,
    y: np.ndarray,
    har: np.ndarray,
    tune_mask: np.ndarray,
    lambdas: tuple[float, ...],
) -> tuple[float, dict[str, float]]:
    target = np.asarray(y, dtype=float)
    baseline = np.asarray(har, dtype=float)
    tune = np.asarray(tune_mask, dtype=bool)
    if tune.shape != (len(target),) or not tune.any():
        raise ValueError("tune_mask must select aligned rows")
    rows = []
    for value in lambdas:
        prediction = baseline + float(value) * correction
        payload = _metric_payload(target, prediction, tune)
        rows.append(
            (
                float(payload["qlike"]),
                float(payload["rmse"]),
                float(value),
                payload,
            )
        )
    best = min(rows, key=lambda row: (row[0], row[1], row[2]))
    return best[2], {
        f"inner_{key}": float(value)
        for key, value in best[3].items()
    }


def evaluate_case_family_fold(
    archive: ArchitectureArchive,
    *,
    fold: int,
    family: ReadoutFamily,
    config: LadderModeReadoutConfig,
) -> tuple[dict[str, object], np.ndarray]:
    matrix, feature_names, transform = build_readout_matrix(archive, family)
    fold_mask = archive.fold == int(fold)
    train = fold_mask & (archive.fold_split == "train")
    validation = fold_mask & (archive.fold_split == "val")
    residual_train = train & archive.prequential_residual_valid
    inner_fit, inner_tune = chronological_inner_split(
        archive.origin_date,
        residual_train,
        holdout_fraction=config.inner_holdout_fraction,
    )
    inner_correction = _fit_correction(
        matrix,
        archive.prequential_residual_path,
        inner_fit,
        transform=transform,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
    )
    selected_lambda, inner_payload = choose_global_lambda(
        inner_correction,
        y=archive.target_path,
        har=archive.har_prediction_path,
        tune_mask=inner_tune,
        lambdas=config.global_lambdas,
    )
    full_correction = _fit_correction(
        matrix,
        archive.prequential_residual_path,
        residual_train,
        transform=transform,
        pca_components=config.pca_components,
        ridge_alpha=config.ridge_alpha,
    )
    prediction = archive.har_prediction_path + selected_lambda * full_correction
    outer_payload = _metric_payload(
        archive.target_path,
        prediction,
        validation,
    )
    case_name = str(archive.case[0])
    row = {
        "fold": int(fold),
        "case": case_name,
        "architecture": str(archive.architecture[0]),
        "control": str(archive.control[0]),
        "interaction_scale": float(archive.interaction_scale[0]),
        "readout_family": family,
        "feature_transform": transform,
        "feature_width": int(matrix.shape[1]),
        "pca_components": (
            int(config.pca_components) if transform == "pca" else 0
        ),
        "ridge_alpha": float(config.ridge_alpha),
        "selected_lambda": float(selected_lambda),
        "inner_fit_rows": int(inner_fit.sum()),
        "inner_tune_rows": int(inner_tune.sum()),
        "validation_rows": int(validation.sum()),
        "feature_names": "|".join(feature_names),
        **inner_payload,
        **{
            f"val_{key}": float(value)
            for key, value in outer_payload.items()
        },
    }
    return row, prediction


def selection_priority(family: str) -> int:
    order = {
        "occupation_pca4": 0,
        "symmetric_modes": 1,
        "antisymmetric_modes": 2,
        "compact_modes": 3,
        "all_modes": 4,
    }
    return int(order.get(str(family), 999))


def select_candidate(
    candidates: pd.DataFrame,
    *,
    policy: str,
) -> pd.Series:
    if policy == "frozen":
        subset = candidates.loc[
            candidates["readout_family"].eq("occupation_pca4")
        ].copy()
    elif policy == "modes":
        subset = candidates.loc[
            ~candidates["readout_family"].eq("occupation_pca4")
        ].copy()
    elif policy == "full":
        subset = candidates.copy()
    else:
        raise ValueError(f"unsupported selection policy: {policy}")
    if subset.empty:
        raise ValueError(f"selection policy {policy!r} has no candidates")
    subset["family_priority"] = subset["readout_family"].map(selection_priority)
    return subset.sort_values(
        [
            "inner_qlike",
            "inner_rmse",
            "selected_lambda",
            "feature_width",
            "family_priority",
            "interaction_scale",
            "case",
        ]
    ).iloc[0]


def prediction_frame(
    archive: ArchitectureArchive,
    prediction: np.ndarray,
    *,
    fold: int,
    model_name: str,
    readout_family: str,
    selected_lambda: float,
) -> pd.DataFrame:
    mask = (
        (archive.fold == int(fold))
        & (archive.fold_split == "val")
    )
    horizons = archive.target_path.shape[1]
    return pd.DataFrame(
        {
            "fold": np.repeat(archive.fold[mask], horizons),
            "sample_id": np.repeat(archive.sample_id[mask], horizons),
            "lead": np.repeat(archive.lead[mask], horizons),
            "label": np.repeat(archive.label[mask], horizons),
            "episode_id": np.repeat(archive.episode_id[mask], horizons),
            "origin_date": np.repeat(archive.origin_date[mask], horizons),
            "model_name": np.repeat(
                model_name,
                int(mask.sum()) * horizons,
            ),
            "case": np.repeat(
                str(archive.case[0]),
                int(mask.sum()) * horizons,
            ),
            "architecture": np.repeat(
                str(archive.architecture[0]),
                int(mask.sum()) * horizons,
            ),
            "control": np.repeat(
                str(archive.control[0]),
                int(mask.sum()) * horizons,
            ),
            "interaction_scale": np.repeat(
                float(archive.interaction_scale[0]),
                int(mask.sum()) * horizons,
            ),
            "readout_family": np.repeat(
                readout_family,
                int(mask.sum()) * horizons,
            ),
            "selected_lambda": np.repeat(
                float(selected_lambda),
                int(mask.sum()) * horizons,
            ),
            "horizon": np.tile(
                np.arange(1, horizons + 1),
                int(mask.sum()),
            ),
            "y_true": archive.target_path[mask].reshape(-1),
            "y_pred": np.asarray(prediction)[mask].reshape(-1),
            "har_pred": archive.har_prediction_path[mask].reshape(-1),
        }
    )


def grouped_prediction_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = (
        "model_name",
        "case",
        "architecture",
        "control",
        "interaction_scale",
        "readout_family",
    )
    for keys, local in predictions.groupby(
        list(group_columns),
        dropna=False,
    ):
        payload = dict(zip(group_columns, keys))
        metric = _metric_payload(
            local["y_true"].to_numpy()[:, None],
            local["y_pred"].to_numpy()[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                **payload,
                **{key: float(value) for key, value in metric.items()},
                "rows": int(len(local)),
                "folds": int(local["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def lead_label_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = (
        "model_name",
        "case",
        "architecture",
        "control",
        "interaction_scale",
        "readout_family",
        "lead",
        "label",
    )
    for keys, local in predictions.groupby(
        list(group_columns),
        dropna=False,
    ):
        payload = dict(zip(group_columns, keys))
        metric = _metric_payload(
            local["y_true"].to_numpy()[:, None],
            local["y_pred"].to_numpy()[:, None],
            np.ones(len(local), dtype=bool),
        )
        rows.append(
            {
                **payload,
                **{key: float(value) for key, value in metric.items()},
                "rows": int(len(local)),
                "folds": int(local["fold"].nunique()),
            }
        )
    return pd.DataFrame(rows)
