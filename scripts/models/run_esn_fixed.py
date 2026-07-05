#!/usr/bin/env python3
"""Run a fixed-configuration ESN on the shared one-step rolling protocol.

This is the first reservoir benchmark, not a tuned final ESN. The reservoir configuration
is fixed before evaluating the 245 out-of-sample targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET = 'target_log_rv_t_plus_1'


def score(y: np.ndarray, p: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    yv, pv = np.exp(2 * y), np.exp(2 * p)
    r = yv / pv
    qlike = float(np.mean(r - np.log(r) - 1))
    X = np.column_stack([np.ones(len(p)), p])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    ssr = float(np.sum((y - fit) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        'rmse_log_rv': rmse,
        'mae_log_rv': mae,
        'qlike_variance': qlike,
        'mz_intercept': float(beta[0]),
        'mz_slope': float(beta[1]),
        'mz_r2': float(1 - ssr / sst),
    }


def make_reservoir(n_inputs: int, n_reservoir: int, spectral_radius: float,
                   input_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    win = rng.uniform(-input_scale, input_scale, size=(n_reservoir, n_inputs + 1))
    w = rng.normal(0.0, 1.0, size=(n_reservoir, n_reservoir))
    mask = rng.random((n_reservoir, n_reservoir)) < 0.10
    w *= mask
    eig = np.linalg.eigvals(w)
    radius = float(np.max(np.abs(eig)))
    if radius <= 0:
        raise ValueError('Reservoir spectral radius is zero')
    w *= spectral_radius / radius
    return win, w


def run_sequence(X: np.ndarray, win: np.ndarray, w: np.ndarray, leak: float,
                 state0: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros(w.shape[0]) if state0 is None else state0.copy()
    states = np.empty((len(X), w.shape[0]), dtype=float)
    for i, u in enumerate(X):
        drive = win @ np.concatenate([[1.0], u]) + w @ state
        candidate = np.tanh(drive)
        state = (1.0 - leak) * state + leak * candidate
        states[i] = state
    return states, state


def ridge_readout(Z: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = np.eye(Z.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(Z.T @ Z + reg, Z.T @ y)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'))
    p.add_argument('--catalog', type=Path, default=Path('data/processed/paper_monthly/features/feature_catalog.csv'))
    p.add_argument('--folds', type=Path, default=Path('results/paper_monthly/protocol/paper_rolling_one_step/folds.csv'))
    p.add_argument('--outdir', type=Path, default=Path('results/paper_monthly/models/esn_fixed'))
    p.add_argument('--reservoir-size', type=int, default=200)
    p.add_argument('--spectral-radius', type=float, default=0.9)
    p.add_argument('--input-scale', type=float, default=0.5)
    p.add_argument('--leak', type=float, default=0.3)
    p.add_argument('--readout-alpha', type=float, default=1e-3)
    p.add_argument('--washout', type=int, default=24)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df['date'] = pd.to_datetime(df['date'])
    folds = pd.read_csv(args.folds, parse_dates=['forecast_date','prediction_origin','train_calendar_start','train_calendar_end'])
    feature_cols = pd.read_csv(args.catalog)['feature'].tolist()

    win, w = make_reservoir(len(feature_cols), args.reservoir_size, args.spectral_radius, args.input_scale, args.seed)
    rows = []

    for _, f in folds.iterrows():
        train_mask = (
            (df['date'] >= f['train_calendar_start']) &
            (df['date'] <= f['train_calendar_end']) &
            df[feature_cols].notna().all(axis=1) &
            df[TARGET].notna()
        )
        train = df.loc[train_mask].copy()
        pred_row = df.loc[df['date'].eq(f['prediction_origin'])].iloc[0]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(train[feature_cols])
        Xp = scaler.transform(pd.DataFrame([pred_row[feature_cols].to_dict()]))
        ytr = train[TARGET].to_numpy(float)

        states, final_state = run_sequence(Xtr, win, w, args.leak)
        if len(states) <= args.washout:
            raise ValueError('Washout leaves no training rows')
        Z = np.column_stack([np.ones(len(states)), Xtr, states])
        beta = ridge_readout(Z[args.washout:], ytr[args.washout:], args.readout_alpha)

        pred_state, _ = run_sequence(Xp, win, w, args.leak, final_state)
        Zp = np.concatenate([[1.0], Xp[0], pred_state[0]])
        y_pred = float(Zp @ beta)

        rows.append({
            'fold_id': int(f['fold_id']),
            'forecast_date': f['forecast_date'],
            'prediction_origin': f['prediction_origin'],
            'y_true_log_rv': float(pred_row[TARGET]),
            'y_pred_log_rv': y_pred,
            'train_rows': int(len(train)),
            'effective_readout_rows': int(len(train) - args.washout),
        })

    pred = pd.DataFrame(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.outdir / 'predictions.csv', index=False)
    metrics = score(pred['y_true_log_rv'].to_numpy(), pred['y_pred_log_rv'].to_numpy())
    pd.DataFrame([{'model':'esn_fixed', 'n_forecasts':len(pred), **metrics}]).to_csv(args.outdir / 'metrics.csv', index=False)

    manifest = {
        'protocol': 'paper_rolling_one_step',
        'model': 'esn_fixed',
        'feature_count': len(feature_cols),
        'reservoir_size': args.reservoir_size,
        'spectral_radius': args.spectral_radius,
        'input_scale': args.input_scale,
        'leak': args.leak,
        'readout_alpha': args.readout_alpha,
        'washout': args.washout,
        'seed': args.seed,
        'selection': 'none; configuration fixed before out-of-sample evaluation',
        'state_rule': 'zero reset at each rolling-window start; prediction origin continues from final training state',
        'readout': 'ridge on intercept + scaled inputs + reservoir state',
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(pd.DataFrame([{'model':'esn_fixed', 'n_forecasts':len(pred), **metrics}]).to_string(index=False))
    print('\n' + json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
