#!/usr/bin/env python3
"""Map linear multiscale explainability and ESN headroom across natural time scales.

Natural, non-overlapping period targets:
- daily: next trading day's |log return|
- weekly: next calendar week's realized volatility
- monthly: next calendar month's realized volatility

HAR and ESN receive the same three volatility-state inputs at every scale. This is a
controlled diagnostic for temporal scale, not a final challenge benchmark or tuning run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

SCALES = {
    'daily': {'rule': None, 'windows': (1, 5, 22)},
    'weekly': {'rule': 'W-FRI', 'windows': (1, 4, 13)},
    'monthly': {'rule': 'ME', 'windows': (1, 3, 12)},
}


def score(y: np.ndarray, p: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    yv, pv = np.exp(2 * y), np.exp(2 * p)
    ratio = yv / pv
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
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
        'mz_r2': float(1.0 - ssr / sst),
    }


def build_period_data(daily: pd.DataFrame, scale: str) -> pd.DataFrame:
    cfg = SCALES[scale]
    x = daily[['date', 'close']].dropna().copy()
    x['log_return'] = np.log(x['close']).diff()
    x = x.dropna(subset=['log_return'])

    if cfg['rule'] is None:
        out = x[['date', 'log_return']].copy()
        out['rv'] = out['log_return'].abs()
    else:
        g = x.set_index('date')['log_return'].resample(cfg['rule'])
        out = pd.DataFrame({
            'rv': g.apply(lambda s: float(np.sqrt(np.square(s.dropna()).sum()))),
            'n_returns': g.count(),
        }).reset_index()
        out = out.loc[out['n_returns'] > 0].copy()

    out['log_rv'] = np.log(out['rv'].where(out['rv'] > 0))
    w1, w2, w3 = cfg['windows']
    out[f'har_{w1}'] = out['log_rv'].rolling(w1, min_periods=w1).mean()
    out[f'har_{w2}'] = out['log_rv'].rolling(w2, min_periods=w2).mean()
    out[f'har_{w3}'] = out['log_rv'].rolling(w3, min_periods=w3).mean()
    out['target_log_rv_t_plus_1'] = out['log_rv'].shift(-1)
    return out


def make_reservoir(n_inputs: int, n_nodes: int, spectral_radius: float,
                   input_scale: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    win = rng.uniform(-input_scale, input_scale, size=(n_nodes, n_inputs + 1))
    w = rng.normal(size=(n_nodes, n_nodes))
    w *= rng.random((n_nodes, n_nodes)) < 0.10
    radius = float(np.max(np.abs(np.linalg.eigvals(w))))
    if radius <= 0:
        raise ValueError('Reservoir spectral radius is zero')
    w *= spectral_radius / radius
    return win, w


def run_esn(X: np.ndarray, win: np.ndarray, w: np.ndarray, leak: float) -> np.ndarray:
    state = np.zeros(w.shape[0], dtype=float)
    states = np.empty((len(X), w.shape[0]), dtype=float)
    for i, u in enumerate(X):
        cand = np.tanh(win @ np.concatenate([[1.0], u]) + w @ state)
        state = (1.0 - leak) * state + leak * cand
        states[i] = state
    return states


def ridge_fit(Z: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = np.eye(Z.shape[1]) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(Z.T @ Z + reg, Z.T @ y)


def evaluate_scale(df: pd.DataFrame, scale: str, train_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    windows = SCALES[scale]['windows']
    features = [f'har_{w}' for w in windows]
    model_df = df[['date', 'log_rv', 'target_log_rv_t_plus_1', *features]].dropna().copy()
    if len(model_df) < 100:
        raise ValueError(f'{scale}: too few complete rows ({len(model_df)})')

    split = int(np.floor(len(model_df) * train_fraction))
    train = model_df.iloc[:split].copy()
    test = model_df.iloc[split:].copy()
    if len(test) < 20:
        raise ValueError(f'{scale}: test set too small ({len(test)})')

    ytr = train['target_log_rv_t_plus_1'].to_numpy(float)
    yte = test['target_log_rv_t_plus_1'].to_numpy(float)

    preds = []
    metrics = []

    persistence = test['log_rv'].to_numpy(float)
    preds.append(pd.DataFrame({'date': test['date'], 'scale': scale, 'model': 'persistence', 'y_true_log_rv': yte, 'y_pred_log_rv': persistence}))
    metrics.append({'scale': scale, 'model': 'persistence', 'n_train': len(train), 'n_test': len(test), **score(yte, persistence)})

    har = LinearRegression().fit(train[features], ytr)
    ph = har.predict(test[features])
    preds.append(pd.DataFrame({'date': test['date'], 'scale': scale, 'model': 'har', 'y_true_log_rv': yte, 'y_pred_log_rv': ph}))
    metrics.append({'scale': scale, 'model': 'har', 'n_train': len(train), 'n_test': len(test), **score(yte, ph)})

    scaler = StandardScaler()
    Xall = scaler.fit_transform(model_df[features].iloc[:split])
    Xtest = scaler.transform(model_df[features].iloc[split:])
    Xseq = np.vstack([Xall, Xtest])

    win, w = make_reservoir(len(features), 50, 0.9, 0.5, seed)
    states = run_esn(Xseq, win, w, 0.3)
    Z = np.column_stack([np.ones(len(Xseq)), Xseq, states])
    washout = min(24, max(1, split // 10))
    beta = ridge_fit(Z[washout:split], ytr[washout:], 100.0)
    pe = Z[split:] @ beta
    preds.append(pd.DataFrame({'date': test['date'], 'scale': scale, 'model': 'esn_fixed', 'y_true_log_rv': yte, 'y_pred_log_rv': pe}))
    metrics.append({'scale': scale, 'model': 'esn_fixed', 'n_train': len(train), 'n_test': len(test), **score(yte, pe)})

    acf = {f'acf_lag_{lag}': float(model_df['target_log_rv_t_plus_1'].autocorr(lag=lag)) for lag in (1, 2, 3)}
    meta = {
        'scale': scale,
        'windows': windows,
        'rows_complete': int(len(model_df)),
        'train_rows': int(len(train)),
        'test_rows': int(len(test)),
        'train_end': str(train['date'].max()),
        'test_start': str(test['date'].min()),
        **acf,
    }
    return pd.concat(preds, ignore_index=True), pd.DataFrame(metrics), meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, default=Path('data/interim/paper_monthly/daily_source_prepared.parquet'))
    p.add_argument('--outdir', type=Path, default=Path('results/challenge_primary/temporal_scale_headroom'))
    p.add_argument('--train-fraction', type=float, default=0.70)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    if not 0.5 <= args.train_fraction < 0.9:
        raise ValueError('train-fraction must be in [0.5, 0.9)')

    daily = pd.read_parquet(args.input)
    daily['date'] = pd.to_datetime(daily['date'], errors='raise')
    if daily['date'].duplicated().any() or not daily['date'].is_monotonic_increasing:
        raise ValueError('Daily source dates must be unique and increasing')

    all_preds, all_metrics, scale_meta = [], [], []
    for scale in SCALES:
        period = build_period_data(daily, scale)
        pred, metrics, meta = evaluate_scale(period, scale, args.train_fraction, args.seed)
        all_preds.append(pred)
        all_metrics.append(metrics)
        scale_meta.append(meta)

    predictions = pd.concat(all_preds, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)

    pivot = metrics.pivot(index='scale', columns='model', values='rmse_log_rv')
    metrics = metrics.merge(
        (pivot['har'] - pivot['esn_fixed']).rename('esn_rmse_gain_over_har'),
        left_on='scale', right_index=True, how='left',
    )
    metrics['esn_relative_rmse_gain_over_har'] = metrics['esn_rmse_gain_over_har'] / metrics['rmse_log_rv'].where(metrics['model'].eq('har')).groupby(metrics['scale']).transform('max')

    args.outdir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.outdir / 'predictions.csv', index=False)
    metrics.sort_values(['scale', 'rmse_log_rv']).to_csv(args.outdir / 'metrics.csv', index=False)
    pd.DataFrame(scale_meta).to_csv(args.outdir / 'scale_summary.csv', index=False)

    manifest = {
        'purpose': 'diagnostic map of HAR explainability and ESN headroom across natural temporal aggregation scales',
        'input': str(args.input),
        'scales': SCALES,
        'target': 'next natural period log realized volatility; future periods do not overlap',
        'features': 'same three HAR-style volatility-state summaries for HAR and ESN at each scale',
        'split': f'first {args.train_fraction:.0%} chronological train, remaining out-of-sample test; no test refit',
        'models': {
            'persistence': 'current period log RV',
            'har': 'OLS on three scale-specific volatility-state means',
            'esn_fixed': '50 nodes, spectral radius 0.9, input scale 0.5, leak 0.3, input+state readout, Ridge alpha 100, seed fixed',
        },
        'limitation': 'screening diagnostic only; final challenge protocol and target remain unfrozen',
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(metrics.sort_values(['scale', 'rmse_log_rv']).to_string(index=False))
    print('\nScale summary:')
    print(pd.DataFrame(scale_meta).to_string(index=False))


if __name__ == '__main__':
    main()
