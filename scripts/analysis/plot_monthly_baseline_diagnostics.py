#!/usr/bin/env python3
"""Generate Phase-2-style diagnostics for the frozen monthly challenge target.

The script consumes existing prediction CSVs, aligns models on common forecast dates,
verifies identical target values, and writes figures plus model-error diagnostics.

Example:
    python scripts/analysis/plot_monthly_baseline_diagnostics.py \
      --model harx=results/paper_monthly/models/har_harx/predictions.csv:harx_public_available \
      --model har=results/paper_monthly/models/har_harx/predictions.csv:har \
      --model esn=results/paper_monthly/models/esn_fixed/<run>/predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATE_CANDIDATES = ('forecast_date', 'date')
TRUE_CANDIDATES = ('y_true_log_rv', 'y_true')
PRED_CANDIDATES = ('y_pred_log_rv', 'y_pred', 'prediction')


def first_present(columns: set[str], candidates: tuple[str, ...], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f'Could not find {label}; tried {candidates}')


def parse_model_spec(spec: str) -> tuple[str, Path, str | None]:
    if '=' not in spec:
        raise ValueError(f'Invalid --model spec {spec!r}; expected label=path[:row_model]')
    label, rhs = spec.split('=', 1)
    if not label:
        raise ValueError(f'Empty model label in {spec!r}')
    row_model = None
    path_text = rhs
    if ':' in rhs:
        path_text, row_model = rhs.rsplit(':', 1)
    return label, Path(path_text), row_model


def load_predictions(label: str, path: Path, row_model: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if row_model is not None:
        if 'model' not in df.columns:
            raise KeyError(f'{path} has no model column for selector {row_model!r}')
        df = df.loc[df['model'].astype(str).eq(row_model)].copy()
        if df.empty:
            raise ValueError(f'No rows with model={row_model!r} in {path}')

    columns = set(df.columns)
    date_col = first_present(columns, DATE_CANDIDATES, 'forecast date')
    true_col = first_present(columns, TRUE_CANDIDATES, 'true target')
    pred_col = first_present(columns, PRED_CANDIDATES, 'prediction')

    out = df[[date_col, true_col, pred_col]].rename(columns={
        date_col: 'date',
        true_col: 'y_true',
        pred_col: f'pred__{label}',
    })
    out['date'] = pd.to_datetime(out['date'], errors='raise')
    out['y_true'] = pd.to_numeric(out['y_true'], errors='raise')
    out[f'pred__{label}'] = pd.to_numeric(out[f'pred__{label}'], errors='raise')
    if out['date'].duplicated().any():
        raise ValueError(f'Duplicate forecast dates for {label} in {path}')
    return out.sort_values('date').reset_index(drop=True)


def align_models(models: list[tuple[str, Path, str | None]]) -> tuple[pd.DataFrame, list[str], dict]:
    labels = [m[0] for m in models]
    loaded = [(label, load_predictions(label, path, selector), path, selector) for label, path, selector in models]

    common = set(loaded[0][1]['date'])
    for _, df, _, _ in loaded[1:]:
        common &= set(df['date'])
    if not common:
        raise ValueError('No common forecast dates across model prediction files')

    common_dates = pd.Series(sorted(common), name='date')
    aligned = pd.DataFrame({'date': common_dates})
    truth_reference = None
    source_meta = {}

    for label, df, path, selector in loaded:
        x = df.loc[df['date'].isin(common)].copy()
        if len(x) != len(common_dates):
            raise ValueError(f'Unexpected alignment loss for {label}')
        x = x.sort_values('date')
        if truth_reference is None:
            truth_reference = x[['date', 'y_true']].copy()
            aligned = aligned.merge(truth_reference, on='date', how='left', validate='one_to_one')
        else:
            check = truth_reference.merge(x[['date', 'y_true']], on='date', suffixes=('_ref', '_new'), validate='one_to_one')
            max_diff = float((check['y_true_ref'] - check['y_true_new']).abs().max())
            if max_diff > 1e-10:
                raise ValueError(f'Truth mismatch for {label}; max absolute difference={max_diff}')
        aligned = aligned.merge(x[['date', f'pred__{label}']], on='date', how='left', validate='one_to_one')
        source_meta[label] = {'path': str(path), 'row_model_selector': selector}

    return aligned.sort_values('date').reset_index(drop=True), labels, source_meta


def transition_states(df: pd.DataFrame) -> pd.Series:
    """Descriptive development-period states only; not final causal evaluation labels."""
    q_low = float(df['y_true'].quantile(1 / 3))
    q_high = float(df['y_true'].quantile(2 / 3))
    prev = df['y_true'].shift(1)
    current = df['y_true']

    prev_state = pd.Series('middle', index=df.index, dtype='object')
    curr_state = pd.Series('middle', index=df.index, dtype='object')
    prev_state.loc[prev <= q_low] = 'calm'
    prev_state.loc[prev >= q_high] = 'elevated'
    curr_state.loc[current <= q_low] = 'calm'
    curr_state.loc[current >= q_high] = 'elevated'

    state = prev_state + '_to_' + curr_state
    state.loc[prev.isna()] = 'initial'
    return state


def save_target_plot(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(df['date'], df['y_true'])
    ax.set_title('Frozen monthly target: realized log volatility')
    ax.set_xlabel('Forecast month')
    ax.set_ylabel('Log realized volatility')
    fig.tight_layout()
    fig.savefig(outdir / '01_target_over_time.png', dpi=180)
    plt.close(fig)


def save_prediction_plot(df: pd.DataFrame, labels: list[str], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df['date'], df['y_true'], linewidth=2, label='actual')
    for label in labels:
        ax.plot(df['date'], df[f'pred__{label}'], label=label, alpha=0.85)
    ax.set_title('Actual versus predicted monthly log volatility')
    ax.set_xlabel('Forecast month')
    ax.set_ylabel('Log realized volatility')
    ax.legend(ncol=min(4, len(labels) + 1))
    fig.tight_layout()
    fig.savefig(outdir / '02_actual_vs_predicted_time.png', dpi=180)
    plt.close(fig)


def save_residual_plot(df: pd.DataFrame, labels: list[str], outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    for label in labels:
        residual = df['y_true'] - df[f'pred__{label}']
        ax.plot(df['date'], residual, label=label, alpha=0.85)
    ax.axhline(0.0, linewidth=1)
    ax.set_title('Forecast residuals through time')
    ax.set_xlabel('Forecast month')
    ax.set_ylabel('Actual minus prediction')
    ax.legend(ncol=min(4, len(labels)))
    fig.tight_layout()
    fig.savefig(outdir / '03_residuals_over_time.png', dpi=180)
    plt.close(fig)


def save_scatter(df: pd.DataFrame, labels: list[str], outdir: Path) -> None:
    low = float(min(df['y_true'].min(), *(df[f'pred__{label}'].min() for label in labels)))
    high = float(max(df['y_true'].max(), *(df[f'pred__{label}'].max() for label in labels)))
    for label in labels:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(df[f'pred__{label}'], df['y_true'], alpha=0.65)
        ax.plot([low, high], [low, high], linewidth=1)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_title(f'Predicted versus actual: {label}')
        ax.set_xlabel('Predicted log RV')
        ax.set_ylabel('Actual log RV')
        fig.tight_layout()
        fig.savefig(outdir / f'04_scatter_{label}.png', dpi=180)
        plt.close(fig)


def save_pairwise_error_plot(df: pd.DataFrame, labels: list[str], outdir: Path, window: int) -> pd.DataFrame:
    rows = []
    if len(labels) < 2:
        return pd.DataFrame(rows)

    reference = labels[0]
    ref_sq = np.square(df['y_true'] - df[f'pred__{reference}'])
    for label in labels[1:]:
        model_sq = np.square(df['y_true'] - df[f'pred__{label}'])
        gain = ref_sq - model_sq
        rolling = pd.Series(gain).rolling(window, min_periods=max(3, window // 3)).mean()

        fig, ax = plt.subplots(figsize=(13, 4.5))
        ax.plot(df['date'], rolling)
        ax.axhline(0.0, linewidth=1)
        ax.set_title(f'Rolling squared-error gain: {label} over {reference}')
        ax.set_xlabel('Forecast month')
        ax.set_ylabel(f'{reference} squared error − {label} squared error')
        fig.tight_layout()
        fig.savefig(outdir / f'05_rolling_gain_{label}_over_{reference}.png', dpi=180)
        plt.close(fig)

        rows.append({
            'reference_model': reference,
            'comparison_model': label,
            'mean_squared_error_gain': float(np.mean(gain)),
            'fraction_months_comparison_better': float(np.mean(gain > 0)),
        })
    return pd.DataFrame(rows)


def save_transition_error_plot(df: pd.DataFrame, labels: list[str], outdir: Path) -> pd.DataFrame:
    x = df.copy()
    x['transition_state'] = transition_states(x)
    rows = []
    for state, group in x.groupby('transition_state'):
        for label in labels:
            err = group['y_true'] - group[f'pred__{label}']
            rows.append({
                'transition_state': state,
                'model': label,
                'n': int(len(group)),
                'rmse_log_rv': float(np.sqrt(np.mean(np.square(err)))),
                'mae_log_rv': float(np.mean(np.abs(err))),
            })
    summary = pd.DataFrame(rows)
    states = [s for s in summary['transition_state'].drop_duplicates() if s != 'initial']
    if states:
        fig, ax = plt.subplots(figsize=(11, 5.5))
        width = 0.8 / len(labels)
        xpos = np.arange(len(states), dtype=float)
        for i, label in enumerate(labels):
            vals = []
            for state in states:
                row = summary.loc[(summary['transition_state'] == state) & (summary['model'] == label)]
                vals.append(float(row['rmse_log_rv'].iloc[0]) if len(row) else np.nan)
            ax.bar(xpos + (i - (len(labels) - 1) / 2) * width, vals, width=width, label=label)
        ax.set_xticks(xpos)
        ax.set_xticklabels(states, rotation=35, ha='right')
        ax.set_title('Development-period RMSE by descriptive transition state')
        ax.set_ylabel('RMSE in log RV')
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / '06_transition_state_rmse.png', dpi=180)
        plt.close(fig)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--model', action='append', required=True, help='label=path[:row_model]; repeat for each model')
    p.add_argument('--outdir', type=Path, default=Path('results/challenge_primary/diagnostics/monthly_baselines'))
    p.add_argument('--rolling-window', type=int, default=12)
    args = p.parse_args()

    models = [parse_model_spec(spec) for spec in args.model]
    labels = [m[0] for m in models]
    if len(set(labels)) != len(labels):
        raise ValueError(f'Duplicate model labels: {labels}')
    if args.rolling_window < 2:
        raise ValueError('rolling-window must be >= 2')

    df, labels, source_meta = align_models(models)
    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / 'aligned_predictions.csv', index=False)

    save_target_plot(df, args.outdir)
    save_prediction_plot(df, labels, args.outdir)
    save_residual_plot(df, labels, args.outdir)
    save_scatter(df, labels, args.outdir)
    gain_summary = save_pairwise_error_plot(df, labels, args.outdir, args.rolling_window)
    transition_summary = save_transition_error_plot(df, labels, args.outdir)

    gain_summary.to_csv(args.outdir / 'pairwise_error_gain_summary.csv', index=False)
    transition_summary.to_csv(args.outdir / 'transition_state_error_summary.csv', index=False)

    manifest = {
        'purpose': 'Phase-2-style diagnostics for frozen monthly next-period log realized volatility target',
        'models_in_order': labels,
        'reference_model_for_pairwise_gain': labels[0],
        'prediction_sources': source_meta,
        'common_forecast_dates': int(len(df)),
        'date_min': str(df['date'].min()),
        'date_max': str(df['date'].max()),
        'rolling_error_window_months': args.rolling_window,
        'transition_state_rule': 'development-period descriptive terciles of realized target; not valid for final causal evaluation',
        'truth_consistency_check': 'required exact agreement within 1e-10 across model files',
    }
    (args.outdir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print('\nPairwise error gain summary:')
    print(gain_summary.to_string(index=False) if len(gain_summary) else 'Need at least two models')
    print('\nTransition-state error summary:')
    print(transition_summary.to_string(index=False))


if __name__ == '__main__':
    main()
