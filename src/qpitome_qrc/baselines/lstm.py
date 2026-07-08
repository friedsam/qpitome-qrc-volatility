"""Deterministic LSTM baseline mechanics for Phase 3.

This module owns sequence construction, the compact recurrent model, training,
and inference. Dataset loading, fold construction, scaling, model selection,
and reporting belong to experiment runners.

The implementation is adapted from the reset-branch monthly LSTM prototype, but
its monthly protocol, feature catalog, and target assumptions are intentionally
not carried into the canonical Phase 3 branch.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LSTMConfig:
    """Fixed small LSTM configuration for the classical baseline."""

    hidden_size: int = 32
    dropout: float = 0.1
    epochs: int = 40
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    seed: int = 42
    early_stopping_patience: int = 6

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")


@dataclass(frozen=True)
class LSTMFitResult:
    """Trained model and diagnostics returned by :func:`fit_lstm`."""

    model: Any
    final_train_mse: float
    n_sequences: int
    best_epoch: int
    epochs_ran: int
    best_val_mse: float


def build_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build local supervised sequences ending on each target date.

    The final feature row of each sequence and its target share the same local
    index. This matches the repository's rolling-window convention: a window of
    information ending at time ``t`` predicts the future-volatility target
    attached to row ``t``.
    """
    X = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError("features must have shape (time, n_features)")
    if y.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if len(X) != len(y):
        raise ValueError("features and targets must have equal length")
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if not np.all(np.isfinite(X)):
        raise ValueError("features contain non-finite values")
    if not np.all(np.isfinite(y)):
        raise ValueError("targets contain non-finite values")

    if len(X) < lookback:
        return (
            np.empty((0, lookback, X.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        X,
        window_shape=lookback,
        axis=0,
    )
    windows = np.moveaxis(windows, -1, 1).copy()
    aligned_targets = y[lookback - 1 :].copy()
    return windows, aligned_targets


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "LSTM baseline requires the optional dependency 'torch'. "
            "Install the baseline extras before running LSTM experiments."
        ) from exc
    return torch


def _make_model(n_features: int, config: LSTMConfig) -> Any:
    torch = _require_torch()

    class _LSTMForecaster(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=n_features,
                hidden_size=config.hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.dropout = torch.nn.Dropout(config.dropout)
            self.head = torch.nn.Linear(config.hidden_size, 1)

        def forward(self, x: Any) -> Any:
            sequence_states, _ = self.lstm(x)
            final_state = sequence_states[:, -1, :]
            return self.head(self.dropout(final_state)).squeeze(-1)

    return _LSTMForecaster()


def fit_lstm(
    sequences: np.ndarray,
    targets: np.ndarray,
    *,
    val_sequences: np.ndarray,
    val_targets: np.ndarray,
    config: LSTMConfig | None = None,
) -> LSTMFitResult:
    """Train one deterministic CPU LSTM with validation early stopping."""
    cfg = config or LSTMConfig()
    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    X_val = np.asarray(val_sequences, dtype=np.float32)
    y_val = np.asarray(val_targets, dtype=np.float32)

    if X.ndim != 3 or X_val.ndim != 3:
        raise ValueError("sequences must have shape (samples, lookback, features)")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("targets must be one-dimensional and match sequences")
    if y_val.ndim != 1 or len(y_val) != len(X_val):
        raise ValueError("validation targets must match validation sequences")
    if len(X) == 0 or len(X_val) == 0:
        raise ValueError("Need non-empty training and validation sequences")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("training data contain non-finite values")
    if not np.all(np.isfinite(X_val)) or not np.all(np.isfinite(y_val)):
        raise ValueError("validation data contain non-finite values")

    torch = _require_torch()
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )

    model = _make_model(X.shape[2], cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    loss_fn = torch.nn.MSELoss()
    val_X_tensor = torch.from_numpy(X_val)
    val_y_tensor = torch.from_numpy(y_val)

    final_loss = float("nan")
    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    epochs_ran = 0

    for epoch in range(cfg.epochs):
        model.train()
        weighted_loss = 0.0
        n_seen = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            prediction = model(X_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()

            batch_size = int(len(X_batch))
            weighted_loss += float(loss.item()) * batch_size
            n_seen += batch_size
        final_loss = weighted_loss / n_seen
        epochs_ran = epoch + 1

        model.eval()
        with torch.no_grad():
            val_prediction = model(val_X_tensor)
            val_mse = float(loss_fn(val_prediction, val_y_tensor).item())

        if val_mse < best_val_mse - 1e-9:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("Early stopping failed to record a validation checkpoint")
    model.load_state_dict(best_state)

    return LSTMFitResult(
        model=model,
        final_train_mse=float(final_loss),
        n_sequences=int(len(X)),
        best_epoch=int(best_epoch),
        epochs_ran=int(epochs_ran),
        best_val_mse=float(best_val_mse),
    )


def predict_lstm(model: Any, sequences: np.ndarray) -> np.ndarray:
    """Predict a batch of local sequences with a fitted LSTM."""
    X = np.asarray(sequences, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("sequences must have shape (samples, lookback, features)")
    if not np.all(np.isfinite(X)):
        raise ValueError("sequences contain non-finite values")

    torch = _require_torch()
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X)).detach().cpu().numpy().astype(float)


def config_to_dict(config: LSTMConfig) -> dict[str, Any]:
    """Return a JSON-safe configuration dictionary."""
    return asdict(config)
