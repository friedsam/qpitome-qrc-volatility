#!/usr/bin/env python3
"""Run the frozen Phase 2 TFIM control on the frozen monthly target.

Declared temporal adaptation:
- Phase 2 physics/readout are frozen;
- the old 40-daily-step window becomes a 12-month window, matching the annual
  volatility-memory scale of the monthly task;
- compact7 inputs are used first because the scientific question is whether the
  reservoir exploits external-state interactions beyond univariate volatility memory.

The runner supports selected-fold probes and checkpointed full walk-forward execution.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from qpitome_qrc.qrc.tfim_reservoir import TFIMQRCConfig, build_qrc_feature_matrix, safe_feature_target_correlations

TARGET = 'target_log_rv_t_plus_1'
COMPACT7 = [
    'vol_state_1m', 'vol_state_3m_mean', 'market_mkt_excess', 'market_str',
    'credit_default_spread_baa_minus_aaa_level', 'macro_ip_growth_lag1',
    'macro_inflation_growth_lag1',
]
LOOKBACK = 12
PCA_COMPONENTS = 6
TOP_K = 240
RIDGE_ALPHA = 1000.0


def score(y: np.ndarray, p: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    yv, pv = np.exp(2 * y), np.exp(2 * p)
    ratio = yv / pv
    qlike = float(np.mean(ratio - np.log(ratio) - 1.0))
    X = np.column_stack([np.ones(len(p)), p])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    ssr = float(np.sum((y - fitted) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {
        'rmse_log_rv': rmse,
        'mae_log_rv': mae,
        'qlike_variance': qlike,
        'mz_intercept': float(beta[0]),
        'mz_slope': float(beta[1]),
        'mz_r2': float(1.0 - ssr / sst),
    }


def make_windows(X: np.ndarray, y: np.ndarray, dates: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(X) != len(y) or len(X) != len(dates):
        raise ValueError('X, y, and dates lengths differ')
    windows = np.asarray([X[i - lookback + 1:i + 1] for i in range(lookback - 1, len(X))])
    return windows, y[lookback - 1:], dates[lookback - 1:]


def leaky_integrate_windows(X: np.ndarray, leak: float = 0.3) -> np.ndarray:
    out = np.empty_like(X, dtype=float)
    for i, window in enumerate(X):
        h = np.zeros(window.shape[1], dtype=float)
        for t, u in enumerate(window):
            h = (1.0 - leak) * h + leak * u
            out[i, t] = h
    return out


def fit_phase2_readout(H: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict, StandardScaler, Ridge]:
    lower = np.percentile(H, 1.0, axis=0)
    upper = np.percentile(H, 99.0, axis=0)
    clipped = np.clip(H, lower, upper)
    corr = safe_feature_target_correlations(clipped, y)
    k = min(TOP_K, H.shape[1])
    idx = np.argsort(np.abs(corr))[-k:]
    scaler = StandardScaler()
    Z = scaler.fit_transform(clipped[:, idx])
    model = Ridge(alpha=RIDGE_ALPHA).fit(Z, y)
    return idx, {'lower': lower, 'upper': upper}, scaler, model


def predict_phase2_readout(H: np.ndarray, idx: np.ndarray, meta: dict, scaler: StandardScaler, model: Ridge) -> np.ndarray:
    clipped = np.clip(H, meta['lower'], meta['upper'])
    return model.predict(scaler.transform(clipped[:, idx]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, default=Path('data/processed/paper_monthly/features/preliminary_features.parquet'))
    p.add_argument('--folds', type=Path, default=Path('results/paper_monthly/protocol/paper_rolling_one_step/folds.csv'))
    p.add_argument('--outdir', type=Path, default=Path('results/challenge_primary/models/tfim_monthly_control'))
    p.add_argument('--only-folds', nargs='*', type=int, default=None)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    df = pd.read_parquet(args.features)
    df['date'] = pd.to_datetime(df['date'])
    folds = pd.read_csv(args.folds, parse_dates=['forecast_date', 'prediction_origin', 'train_calendar_start', 'train_calendar_end'])
    if args.only_folds:
        requested = set(args.only_folds)
        folds = folds.loc[folds['fold_id'].isin(requested)].copy()
        missing = requested - set(folds['fold_id'].astype(int))
        if missing:
            raise ValueError(f'Unknown fold ids: {sorted(missing)}')

    config = TFIMQRCConfig(
        qubits=6, lookback_steps=LOOKBACK, anchor_count=10, anchor_policy='recent',
        observable_mode='zxzz', trotter_steps_per_anchor=3, virtual_nodes_per_anchor=3,
        topology='full', coupling_scale=0.7, transverse_field=0.5, evolution_time=0.5,
        angle_max=np.pi / 2, seed=42, collect_anchor_features=True,
        use_disorder=True, disorder_strength=0.20,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    pred_path = args.outdir / 'predictions.csv'
    metrics_path = args.outdir / 'per_fold_metrics.csv'
    existing_pred = pd.read_csv(pred_path, parse_dates=['forecast_date', 'prediction_origin']) if pred_path.exists() and not args.force else pd.DataFrame()
    existing_metrics = pd.read_csv(metrics_path) if metrics_path.exists() and not args.force else pd.DataFrame()
    completed = set(existing_metrics['fold_id'].astype(int)) if len(existing_metrics) else set()
    pred_rows = existing_pred.to_dict('records') if len(existing_pred) else []
    metric_rows = existing_metrics.to_dict('records') if len(existing_metrics) else []

    for _, fold in folds.iterrows():
        fold_id = int(fold['fold_id'])
        if fold_id in completed and not args.force:
            print(f'SKIP completed fold {fold_id}')
            continue

        train_mask = (
            (df['date'] >= fold['train_calendar_start']) &
            (df['date'] <= fold['train_calendar_end']) &
            df[COMPACT7].notna().all(axis=1) & df[TARGET].notna()
        )
        train = df.loc[train_mask, ['date', TARGET, *COMPACT7]].copy()
        pred_idx = df.index[df['date'].eq(fold['prediction_origin'])]
        if len(pred_idx) != 1:
            raise ValueError(f'Prediction origin not found once: {fold["prediction_origin"]}')
        pred_pos = int(pred_idx[0])
        context = df.iloc[pred_pos - LOOKBACK + 1:pred_pos + 1][['date', TARGET, *COMPACT7]].copy()
        if len(context) != LOOKBACK or context[COMPACT7].isna().any().any():
            raise ValueError(f'Incomplete prediction context for fold {fold_id}')

        input_scaler = StandardScaler()
        pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
        Xtr_pca = pca.fit_transform(input_scaler.fit_transform(train[COMPACT7]))
        Xpred_pca = pca.transform(input_scaler.transform(context[COMPACT7]))
        Xtr_win, ytr, _ = make_windows(Xtr_pca, train[TARGET].to_numpy(float), train['date'].to_numpy(), LOOKBACK)
        Xpred_win = Xpred_pca[np.newaxis, :, :]
        Xtr_win = leaky_integrate_windows(Xtr_win, leak=0.3)
        Xpred_win = leaky_integrate_windows(Xpred_win, leak=0.3)

        print(f'\n=== TFIM monthly control fold {fold_id}: train windows={len(Xtr_win)} ===')
        start = time.perf_counter()
        Htr = build_qrc_feature_matrix(Xtr_win, config, verbose=True)
        Hp = build_qrc_feature_matrix(Xpred_win, config)
        feature_seconds = time.perf_counter() - start

        idx, meta, reservoir_scaler, readout = fit_phase2_readout(Htr, ytr)
        pred = float(predict_phase2_readout(Hp, idx, meta, reservoir_scaler, readout)[0])
        truth = float(context.iloc[-1][TARGET])

        pred_rows = [r for r in pred_rows if int(r['fold_id']) != fold_id]
        pred_rows.append({
            'fold_id': fold_id, 'forecast_date': fold['forecast_date'],
            'prediction_origin': fold['prediction_origin'], 'model': 'tfim_monthly_phase2_control',
            'y_true_log_rv': truth, 'y_pred_log_rv': pred,
            'train_windows': int(len(Xtr_win)), 'n_raw_qrc_features': int(Htr.shape[1]),
            'n_selected_qrc_features': int(len(idx)), 'feature_seconds': feature_seconds,
        })
        metric_rows = [r for r in metric_rows if int(r['fold_id']) != fold_id]
        metric_rows.append({
            'fold_id': fold_id, 'forecast_date': fold['forecast_date'],
            'squared_error': float((truth - pred) ** 2),
            'absolute_error': float(abs(truth - pred)), 'feature_seconds': feature_seconds,
        })
        pd.DataFrame(pred_rows).sort_values('fold_id').to_csv(pred_path, index=False)
        pd.DataFrame(metric_rows).sort_values('fold_id').to_csv(metrics_path, index=False)
        print(f'Checkpointed fold {fold_id}: truth={truth:.6f}, pred={pred:.6f}, feature_seconds={feature_seconds:.1f}')

    pred_df = pd.DataFrame(pred_rows).sort_values('fold_id').reset_index(drop=True)
    if len(pred_df) >= 2:
        aggregate = {'model': 'tfim_monthly_phase2_control', 'n_forecasts': len(pred_df), **score(
            pred_df['y_true_log_rv'].to_numpy(float), pred_df['y_pred_log_rv'].to_numpy(float)
        )}
        pd.DataFrame([aggregate]).to_csv(args.outdir / 'aggregate_metrics.csv', index=False)
        print('\n' + pd.DataFrame([aggregate]).to_string(index=False))

    manifest = {
        'role': 'frozen Phase 2 TFIM quantum control/reference on frozen monthly target',
        'feature_set': 'compact7 paper-informed multivariate external-state proxy',
        'features': COMPACT7,
        'temporal_adaptation': {
            'phase2_original': '40 daily steps', 'monthly_control': '12 monthly steps',
            'reason': 'preserve an annual-scale memory horizon instead of blindly converting 40 days to 40 months',
        },
        'input_preprocessing': 'train-only StandardScaler -> PCA6', 'input_leak': 0.3,
        'configuration': config.__dict__,
        'readout': {
            'clip_percentiles': [1.0, 99.0],
            'feature_selection': 'top 240 by absolute train feature-target correlation',
            'scaler': 'train-only StandardScaler on selected QRC features',
            'ridge_alpha': RIDGE_ALPHA, 'target': 'log RV directly',
        },
        'probe_note': 'Use --only-folds for runtime probes before committing to all 245 folds.',
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == '__main__':
    main()
