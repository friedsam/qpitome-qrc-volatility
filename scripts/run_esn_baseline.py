from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/processed/market_stress_v0.csv")

FEATURES = [
    "spy_log_return",
    "spy_abs_return",
    "spy_range",
    "spy_dollar_volume",
    "rv_5d",
    "rv_10d",
    "rv_20d",
    "spy_drawdown_20d",
    "vix_close",
    "vix_change",
    "vix_pct_change",
    "vix_ma_5d",
]

TARGET = "future_high_vol_label"


def make_sequences(df: pd.DataFrame, features: list[str], target: str, seq_len: int = 20):
    X_raw = df[features].to_numpy(dtype=float)
    y_raw = df[target].to_numpy(dtype=int)
    dates = df["date"].to_numpy()

    X, y, out_dates = [], [], []

    for i in range(seq_len - 1, len(df)):
        X.append(X_raw[i - seq_len + 1 : i + 1])
        y.append(y_raw[i])
        out_dates.append(dates[i])

    return np.asarray(X), np.asarray(y), np.asarray(out_dates)


def chronological_split(X, y, dates):
    train_mask = dates < np.datetime64("2016-01-01")
    val_mask = (dates >= np.datetime64("2016-01-01")) & (dates < np.datetime64("2020-01-01"))
    test_mask = dates >= np.datetime64("2020-01-01")

    return (
        X[train_mask],
        y[train_mask],
        dates[train_mask],
        X[val_mask],
        y[val_mask],
        dates[val_mask],
        X[test_mask],
        y[test_mask],
        dates[test_mask],
    )


class SimpleESN:
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int = 300,
        spectral_radius: float = 0.9,
        input_scale: float = 0.5,
        leak_rate: float = 0.5,
        seed: int = 42,
    ):
        self.input_dim = input_dim
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.input_scale = input_scale
        self.leak_rate = leak_rate
        self.rng = np.random.default_rng(seed)

        self.Win = self.rng.uniform(
            -input_scale,
            input_scale,
            size=(reservoir_size, input_dim),
        )

        W = self.rng.normal(0, 1, size=(reservoir_size, reservoir_size))
        eigvals = np.linalg.eigvals(W)
        max_abs_eig = np.max(np.abs(eigvals))
        self.W = W / max_abs_eig * spectral_radius

    def transform(self, X_seq: np.ndarray) -> np.ndarray:
        """
        X_seq shape: (n_samples, seq_len, input_dim)
        Returns final reservoir state for each sample.
        """
        n_samples = X_seq.shape[0]
        states = np.zeros((n_samples, self.reservoir_size))

        for n in range(n_samples):
            x_state = np.zeros(self.reservoir_size)

            for t in range(X_seq.shape[1]):
                u = X_seq[n, t]
                pre = self.Win @ u + self.W @ x_state
                candidate = np.tanh(pre)
                x_state = (1 - self.leak_rate) * x_state + self.leak_rate * candidate

            states[n] = x_state

        return states


def evaluate(name: str, y_true, y_pred, y_score) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print("balanced_accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("roc_auc:", roc_auc_score(y_true, y_score))
    print("pr_auc:", average_precision_score(y_true, y_score))
    print("\nconfusion_matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nclassification_report:")
    print(classification_report(y_true, y_pred, digits=3))


def main() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    X, y, dates = make_sequences(df, FEATURES, TARGET, seq_len=20)

    (
        X_train,
        y_train,
        dates_train,
        X_val,
        y_val,
        dates_val,
        X_test,
        y_test,
        dates_test,
    ) = chronological_split(X, y, dates)

    print("Sequence shapes:")
    print("train:", X_train.shape, y_train.shape, dates_train.min(), dates_train.max())
    print("val:  ", X_val.shape, y_val.shape, dates_val.min(), dates_val.max())
    print("test: ", X_test.shape, y_test.shape, dates_test.min(), dates_test.max())

    print("\nLabel rates:")
    for name, yy in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(name, dict(zip(*np.unique(yy, return_counts=True))))

    # Scale features using train only.
    scaler = StandardScaler()
    n_train, seq_len, input_dim = X_train.shape

    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, input_dim)).reshape(n_train, seq_len, input_dim)
    X_val_scaled = scaler.transform(X_val.reshape(-1, input_dim)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, input_dim)).reshape(X_test.shape)

    esn = SimpleESN(
        input_dim=input_dim,
        reservoir_size=300,
        spectral_radius=0.9,
        input_scale=0.5,
        leak_rate=0.5,
        seed=42,
    )

    H_train = esn.transform(X_train_scaled)
    H_val = esn.transform(X_val_scaled)
    H_test = esn.transform(X_test_scaled)

    readout = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )
    readout.fit(H_train, y_train)

    for name, H, yy in [
        ("Validation", H_val, y_val),
        ("Test", H_test, y_test),
    ]:
        y_score = readout.predict_proba(H)[:, 1]
        y_pred = (y_score >= 0.5).astype(int)
        evaluate(name, yy, y_pred, y_score)


if __name__ == "__main__":
    main()