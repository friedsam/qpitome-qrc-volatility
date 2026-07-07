from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal

import numpy as np
import pandas as pd
from reservoirpy.nodes import Reservoir
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.baselines.esn import PoolingMode, reservoir_sequence_features, scale_sequence_splits
from qpitome_qrc.baselines.reservoir_readouts import ReadoutConfig, ReadoutFitResult, fit_readout

StateMode = Literal["last_layer", "concat_layers"]


@dataclass(frozen=True)
class DeepESNConfig:
    """Stacked reservoir configuration.

    A single-layer ESN is represented by layer_units=(300,). A two-layer
    DeepESN is represented by layer_units=(300, 300), etc.
    """

    layer_units: tuple[int, ...] = (300, 300)
    spectral_radius: tuple[float, ...] = (0.7, 0.7)
    leak_rate: tuple[float, ...] = (0.5, 0.5)
    reservoir_connectivity: tuple[float, ...] = (0.1, 0.1)
    input_scaling: float = 0.5
    input_connectivity: float = 0.5
    seed: int = 42
    washout: int = 0
    pooling: PoolingMode = "final"
    state_mode: StateMode = "last_layer"
    scale_inputs: bool = True


@dataclass
class DeepESNRunResult:
    config: DeepESNConfig
    readout_config: ReadoutConfig
    reservoirs: list[Reservoir]
    input_scaler: StandardScaler | None
    readout_result: ReadoutFitResult
    val_scores: np.ndarray
    test_scores: np.ndarray

    @property
    def threshold(self) -> float:
        return self.readout_result.threshold

    @property
    def val_metrics(self):
        return self.readout_result.val_metrics

    @property
    def test_metrics(self):
        return self.readout_result.test_metrics


def _as_layer_tuple(value, n_layers: int, name: str) -> tuple:
    if isinstance(value, tuple):
        if len(value) != n_layers:
            raise ValueError(f"{name} length must match n_layers={n_layers}")
        return value
    return tuple([value] * n_layers)


def validate_deep_config(config: DeepESNConfig) -> None:
    n_layers = len(config.layer_units)
    if n_layers < 1:
        raise ValueError("DeepESNConfig.layer_units must contain at least one layer")
    for name in ["spectral_radius", "leak_rate", "reservoir_connectivity"]:
        value = getattr(config, name)
        if len(value) != n_layers:
            raise ValueError(f"{name} length must match number of layers")


def build_deep_reservoirs(config: DeepESNConfig, input_dim: int) -> list[Reservoir]:
    """Build stacked ReservoirPy reservoirs."""
    validate_deep_config(config)
    reservoirs = []
    current_dim = input_dim

    for i, units in enumerate(config.layer_units):
        reservoirs.append(
            Reservoir(
                units=units,
                lr=config.leak_rate[i],
                sr=config.spectral_radius[i],
                input_scaling=config.input_scaling,
                input_connectivity=config.input_connectivity,
                rc_connectivity=config.reservoir_connectivity[i],
                input_dim=current_dim,
                seed=config.seed + i * 1009,
            )
        )
        current_dim = units

    return reservoirs


def _run_reservoir(reservoir: Reservoir, seq: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(reservoir.run(seq, reset=True))
    except TypeError:
        if hasattr(reservoir, "state"):
            reservoir.reset()
        return np.asarray(reservoir.run(seq))


def pooled_trajectory_features(
    trajectory: np.ndarray,
    washout: int = 0,
    pooling: PoolingMode = "final",
) -> np.ndarray:
    """Pool one reservoir trajectory into a fixed feature vector."""
    if trajectory.ndim != 2:
        raise ValueError(f"Expected 2D trajectory, got {trajectory.shape}")
    if washout < 0:
        raise ValueError("washout must be >= 0")
    if washout >= trajectory.shape[0]:
        raise ValueError("washout must be smaller than trajectory length")

    usable = trajectory[washout:] if washout > 0 else trajectory

    if pooling == "final":
        return usable[-1]
    if pooling == "mean":
        return usable.mean(axis=0)
    if pooling == "final_mean":
        return np.concatenate([usable[-1], usable.mean(axis=0)])
    if pooling == "final_mean_max_std":
        return np.concatenate([usable[-1], usable.mean(axis=0), usable.max(axis=0), usable.std(axis=0)])
    raise ValueError(f"Unknown pooling mode: {pooling}")


def deep_reservoir_sequence_features(
    reservoirs: list[Reservoir],
    X: np.ndarray,
    washout: int = 0,
    pooling: PoolingMode = "final",
    state_mode: StateMode = "last_layer",
) -> np.ndarray:
    """Transform input sequences into stacked-reservoir features."""
    if X.ndim != 3:
        raise ValueError(f"Expected X shape (samples, seq_len, features), got {X.shape}")

    features = []

    for seq in X:
        layer_input = seq
        layer_trajs = []
        for reservoir in reservoirs:
            traj = _run_reservoir(reservoir, layer_input)
            layer_trajs.append(traj)
            layer_input = traj

        if state_mode == "last_layer":
            feat = pooled_trajectory_features(layer_trajs[-1], washout=washout, pooling=pooling)
        elif state_mode == "concat_layers":
            feat = np.concatenate(
                [pooled_trajectory_features(traj, washout=washout, pooling=pooling) for traj in layer_trajs]
            )
        else:
            raise ValueError(f"Unknown state_mode: {state_mode}")

        features.append(feat)

    return np.asarray(features)


def fit_deep_esn(
    splits: dict,
    config: DeepESNConfig,
    readout_config: ReadoutConfig,
    tune_threshold: bool = True,
) -> DeepESNRunResult:
    """Fit stacked reservoir + configurable readout."""
    if config.scale_inputs:
        scaled, input_scaler = scale_sequence_splits(splits)
    else:
        scaled, input_scaler = splits, None

    X_train, y_train, _ = scaled["train"]
    X_val, y_val, _ = scaled["val"]
    X_test, y_test, _ = scaled["test"]

    reservoirs = build_deep_reservoirs(config, input_dim=X_train.shape[-1])

    H_train = deep_reservoir_sequence_features(
        reservoirs, X_train, washout=config.washout, pooling=config.pooling, state_mode=config.state_mode
    )
    H_val = deep_reservoir_sequence_features(
        reservoirs, X_val, washout=config.washout, pooling=config.pooling, state_mode=config.state_mode
    )
    H_test = deep_reservoir_sequence_features(
        reservoirs, X_test, washout=config.washout, pooling=config.pooling, state_mode=config.state_mode
    )

    readout_result = fit_readout(
        H_train=H_train,
        y_train=y_train,
        H_val=H_val,
        y_val=y_val,
        H_test=H_test,
        y_test=y_test,
        config=readout_config,
        tune_threshold=tune_threshold,
    )

    return DeepESNRunResult(
        config=config,
        readout_config=readout_config,
        reservoirs=reservoirs,
        input_scaler=input_scaler,
        readout_result=readout_result,
        val_scores=readout_result.val_scores,
        test_scores=readout_result.test_scores,
    )


def summarize_deep_esn_run(result: DeepESNRunResult, include_test: bool = True) -> dict:
    """Flatten run result for tabular comparison."""
    row = asdict(result.config)
    row.update(
        {
            "n_layers": len(result.config.layer_units),
            "readout_kind": result.readout_config.kind,
            "readout_scale_states": result.readout_config.scale_states,
            "readout_params": result.readout_config.params,
            "threshold": result.threshold,
            "val_balanced_accuracy": result.val_metrics.balanced_accuracy,
            "val_roc_auc": result.val_metrics.roc_auc,
            "val_pr_auc": result.val_metrics.pr_auc,
            "val_precision_class_1": result.val_metrics.precision_class_1,
            "val_recall_class_1": result.val_metrics.recall_class_1,
            "val_f1_class_1": result.val_metrics.f1_class_1,
        }
    )
    if include_test:
        row.update(
            {
                "test_balanced_accuracy": result.test_metrics.balanced_accuracy,
                "test_roc_auc": result.test_metrics.roc_auc,
                "test_pr_auc": result.test_metrics.pr_auc,
                "test_precision_class_1": result.test_metrics.precision_class_1,
                "test_recall_class_1": result.test_metrics.recall_class_1,
                "test_f1_class_1": result.test_metrics.f1_class_1,
            }
        )
    return row


def deep_esn_configs(
    layer_units=((300,), (300, 300), (600,), (600, 300)),
    spectral_radius=(0.7,),
    leak_rate=(0.5,),
    reservoir_connectivity=(0.1,),
    seeds=(1, 2, 3),
    washout=(0,),
    pooling=("final",),
    state_mode=("last_layer",),
    input_scaling=0.5,
    input_connectivity=0.5,
) -> list[DeepESNConfig]:
    """Generate DeepESN architecture configs."""
    configs = []
    for layers, sr, leak, conn, seed, wash, pool, smode in product(
        layer_units,
        spectral_radius,
        leak_rate,
        reservoir_connectivity,
        seeds,
        washout,
        pooling,
        state_mode,
    ):
        n_layers = len(layers)
        configs.append(
            DeepESNConfig(
                layer_units=tuple(layers),
                spectral_radius=_as_layer_tuple(sr, n_layers, "spectral_radius"),
                leak_rate=_as_layer_tuple(leak, n_layers, "leak_rate"),
                reservoir_connectivity=_as_layer_tuple(conn, n_layers, "reservoir_connectivity"),
                input_scaling=input_scaling,
                input_connectivity=input_connectivity,
                seed=seed,
                washout=wash,
                pooling=pool,
                state_mode=smode,
            )
        )
    return configs
