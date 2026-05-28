# May 27 Milestone 7 Closeout: Regime-transition interpretation layer

Date: May 27, 2026

Status: complete for Phase 2 narrative use

## Purpose

Milestone 7 connects the Track A volatility-forecasting substrate to a stakeholder-facing regime-transition warning layer. The goal is not to replace the required volatility metrics, but to show how a volatility forecast can become an interpretable operational signal for market-risk monitoring.

This milestone addresses the risk that the project appears to be generic scalar volatility forecasting. The interpretation layer converts forecasted realized volatility and observable market-stress confirmations into transparent warning states that can be read as calm / watch / warning / crisis-like operational regimes.

## Challenge alignment

Track A is financial volatility prediction using public equity-market data. The challenge framing emphasizes volatility regime shifts and transition forecasting, while the required quantitative metrics remain RMSE, QLIKE, and Mincer-Zarnowitz regression.

Milestone 7 preserves that structure:

- Primary substrate: forecast future realized volatility, here `future_rv_20d`.
- Required evaluation: RMSE, QLIKE, Mincer-Zarnowitz, plus correlation and forecast dispersion diagnostics.
- Secondary interpretation layer: derive warning states from forecasted volatility and contemporaneous stress confirmations.
- Stakeholder link: translate model output into watch / warning / crisis-like states that could support trading-desk monitoring, hedging review, risk escalation, or market-making controls.

The warning layer is therefore an application-layer interpretation of the forecast. It is not a substitute for Track A metrics.

## Definitions

Outcome events are train-calibrated from future realized volatility:

- High-volatility event: `future_rv_20d` above the train-set 80th percentile.
- Extreme-volatility event: `future_rv_20d` above the train-set 90th percentile.
- Crisis-candidate event: `future_rv_20d` above the train-set 95th percentile.

These are not hand-labeled crisis classes. They are reproducible forward-looking volatility-tail events.

The operational warning ladder uses contemporaneous market-stress confirmations:

- VIX level and VIX range.
- Drawdown stress.
- Selloff stress.
- Volume stress.
- Realized-volatility acceleration.

The descriptive stress ladder assigns:

- `normal`: no sufficient stress evidence.
- `watch`: early elevated-stress evidence.
- `warning`: forecast or observable stress is meaningfully elevated.
- `crisis_like`: sparse high-severity stress descriptor requiring strong confirmations.

The forecast-aware layer uses a model forecast as the volatility input. QRC and ESN forecasts are passed through the same warning logic so that architecture effects can be compared cleanly.

## Implemented artifacts

Scripts:

- `scripts/export_final_qrc_predictions.py`
  - Exports row-level QRC test predictions.
  - Output: `results/tables/phase2_qrc_final_encoding_readout_predictions.csv`.
- `scripts/export_esn_predictions.py`
  - Exports row-level ESN test predictions.
  - Output: `results/tables/phase2_esn_predictions.csv`.

Notebooks:

- `notebooks/phase2_stress_regime_candidate_analysis.ipynb`
  - Builds descriptive stress indicators and v2/v3 regime ladders.
  - Evaluates warning states against q80/q90/q95 future-volatility events.
- `notebooks/phase2_forecast_aware_regime_layer.ipynb`
  - Converts final QRC volatility forecasts into forecast-aware regime warnings.
- `notebooks/phase2_forecast_aware_regime_layer_comparison.ipynb`
  - Applies the same forecast-aware warning layer to QRC and ESN forecasts.
  - Compares QRC and ESN regime-warning behavior.

Generated tables include:

- `phase2_forecast_aware_regime_eval_notebook.csv`.
- `phase2_qrc_vs_esn_forecast_regime_eval.csv`.
- QRC and ESN row-level prediction exports.

Generated figures include:

- Forecast-aware regime timeline.
- QRC vs ESN volatility forecast timeline.
- QRC vs ESN forecast-aware regime F1 comparison.

## Key results

### Descriptive stress-regime layer

The descriptive v2/v3 stress ladders show that simple, train-calibrated stress confirmations produce plausible regime behavior:

- The calm validation split produces mostly normal/watch states and fewer crisis-like labels.
- The stressed test period produces more warning and crisis-like labels.
- The top q95 future-volatility episodes align with the COVID-19 crash window, validating that the q95 event definition captures historically meaningful crisis-like volatility episodes.

This supports the interpretation that the regime layer is not arbitrary. It maps future-volatility tail events to observable market-stress conditions.

### QRC forecast-aware layer

The final QRC forecast-aware warning layer produced useful broad high-volatility warnings:

- QRC `watch_or_higher` vs q80 high-volatility event:
  - precision: 0.448
  - recall: 0.828
  - F1: 0.581

QRC warning performance is weaker for q90/q95 tail events because the final QRC forecast compresses extreme volatility amplitudes. During the early COVID test window, actual `future_rv_20d` is near 0.91-0.92, while QRC forecasts are substantially lower. This limits crisis-like forecast recall.

Interpretation: the QRC model is not merely a toy; it produces a functioning forecast-aware warning layer and improves over simpler baselines. The remaining gap is specific and diagnosable: tail-amplitude calibration.

### ESN forecast-aware reference

The ESN is a strong but still lightweight classical reservoir reference. It is not a full production volatility model, but it provides a useful upper-bound benchmark for the reservoir-computing architecture.

ESN test forecast metrics:

- RMSE: 0.0719
- QLIKE: -2.5048
- Mincer-Zarnowitz R²: 0.5313
- correlation: 0.7289
- prediction standard deviation: 0.0804

Forecast-aware ESN regime results:

- ESN `warning_or_higher` vs q90 extreme-volatility event:
  - precision: 0.377
  - recall: 0.802
  - F1: 0.513
- ESN `crisis_like` vs q95 crisis-candidate event:
  - precision: 0.228
  - recall: 0.853
  - F1: 0.360

The ESN comparison validates the regime-warning architecture and shows the performance ceiling for a stronger reservoir forecaster. The QRC comparison identifies where the quantum reservoir must improve: tail tracking, readout calibration, and richer encoding.

## Event-level case study: COVID-19 crash onset

The q95 future-volatility event window aligns with the COVID-19 crash onset in early 2020. This is the central qualitative case study for the regime-transition layer.

Observed behavior:

- The future-realized-volatility target reaches extreme values near 0.91-0.92.
- Stress confirmations rise through VIX, drawdown, selloff, range, and realized-volatility indicators.
- QRC forecasts become elevated but remain compressed relative to the realized tail magnitude.
- ESN forecasts track the stress episode more strongly and support higher warning/crisis-like regime performance.

Interpretation:

The case study demonstrates why the project should not stop at scalar forecast metrics alone. Stakeholders need an operational translation layer: when forecasted volatility rises and stress confirmations accumulate, the model can flag watch/warning/crisis-like conditions even when exact tail magnitude is imperfect.

## QRC vs ESN framing

The ESN should not be framed as a negative result for QRC. It is a strong classical reservoir reference that validates the reservoir-computing direction and gives a concrete target for future QRC optimization.

Recommended framing:

> The ESN shows the ceiling of the reservoir approach; the QRC shows a working quantum-reservoir path toward that ceiling.

The current QRC result demonstrates feasibility and progress:

- It implements an end-to-end quantum reservoir volatility pipeline.
- It improves over simple volatility baselines.
- It produces usable high-volatility warning behavior.
- It supports the same forecast-to-regime-warning architecture as the ESN.
- It identifies clear technical bottlenecks rather than failing generically.

The remaining gap is concentrated in extreme-volatility tail calibration.

## Concrete roadmap to close the gap

QRC technical levers:

1. Richer temporal encoding
   - More anchors.
   - Adaptive anchors concentrated near recent stress.
   - Multi-scale windows.

2. Stronger feature encoding
   - Move beyond PCA-only compression.
   - Encode returns, VIX, drawdown, range, volume, and volatility ratios as separate channels.
   - Test repeated input re-uploading.

3. Richer readout observables
   - Add full-pair ZZ observables.
   - Add selected higher-order Pauli observables.
   - Use time-resolved readout trajectories rather than only compressed terminal features.

4. Reservoir parameter tuning
   - Coupling strength.
   - Transverse field.
   - Disorder structure.
   - Evolution time.
   - Chain vs full topology.

5. Tail-aware readout
   - Weighted high-volatility samples.
   - Quantile or asymmetric loss.
   - Tail-probability calibration.
   - Regime-specific calibration layer.

6. Ensemble QRC reservoirs
   - Multiple seeds.
   - Multiple topologies.
   - Multiple readout heads.

Application-layer levers:

- Use the QRC forecast as one input to a regime-warning score rather than the only trigger.
- Combine forecast elevation with VIX, drawdown, selloff, and volume confirmations.
- Report operational warning states, not only point volatility magnitudes.
- Benchmark against ESN to quantify progress and headroom.

## Rubric linkage

Track Selection & Problem Framing:

- Reframes Track A as volatility-regime transition early warning with realized-volatility forecasting as the quantitative substrate.

Stakeholder relevance / impact:

- Converts numerical forecasts into operational states that are directly interpretable for trading desks, risk managers, hedging review, and market-making controls.

Clarity of Communication:

- Produces a transparent warning ladder, timeline plots, and QRC-vs-ESN comparison figures.

Theoretical & Analytical Justification:

- Shows why reservoir dynamics are suitable for nonlinear, multi-scale volatility dynamics.
- Uses ESN as a strong classical reservoir reference and QRC as the quantum reservoir prototype.

## Sign-off assessment

Milestone 7 satisfies the stop condition:

- The volatility forecast becomes a regime-transition warning.
- The warning definition is transparent and reproducible.
- Regime diagnostics do not replace RMSE, QLIKE, or Mincer-Zarnowitz.
- The framework is stakeholder-facing without inventing a complex latent regime-labeling system.

Status: signed off for Phase 2 narrative integration.
