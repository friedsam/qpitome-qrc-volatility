# Palindromic Rydberg Quantum Reservoir for Volatility-Transition Forecasting

**Team:** QPITOME  
**Track:** Financial Volatility Prediction  
**Phase 3 report source:** five content pages plus references; 11-point serif, single-spaced PDF.

## Abstract

We developed a simulator-first quantum reservoir computing pipeline for forecasting volatility transitions at lead 5. A 40-step two-channel sequence drives a six-atom staggered Rydberg ladder through a symmetric A/4-B/2-A/4 control palindrome. Three probe times yield six density-curvature observables, and a no-intercept Ridge head learns only a correction to a causal HAR forecast. On pooled validation paths the selected interaction-on model improved QLIKE by 0.00792 and RMSE by 0.00204 versus HAR; on transition paths the improvements were 0.01287 and 0.00400. A matched interaction-off control performed better, so the evidence supports a useful nonlinear reservoir representation but not an interaction-specific quantum advantage. Aquila experiments showed high observable-transfer fidelity. MNIST, noise, scaling, and finite-shot studies established expressivity, robustness, non-monotonic resource scaling, and the shot budget required to preserve the transition-direction signal.

## 1. Architecture and forecasting target

The operational target is not merely lower pooled loss. We ask whether the QRC correction changes direction around transition paths at lead 5, where calm-period offsets can otherwise cancel crisis-side gains.

```text
40-step level + instability sequence
        -> train-only robust scaling
        -> six-atom staggered Rydberg ladder
           A/4 -> B/2 -> A/4 per observation
           0.02 microseconds; interaction scale 1.25
        -> probes at steps 10, 20, and 40
        -> six density/curvature observables
        -> StandardScaler + Ridge(alpha=100, no intercept)
        -> HAR forecast + QRC residual correction
```

**Figure 1 source:** architecture schematic in the assembled DOCX/PDF.

## 2. Financial simulator results

| Scope | Model | HAR QLIKE | QRC QLIKE | Delta QLIKE | HAR RMSE | QRC RMSE | Delta RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| All validation paths | Selected QRC, interactions on | 0.784580 | 0.776660 | -0.007920 | 0.519081 | 0.517044 | -0.002037 |
| Transition only | Selected QRC, interactions on | 1.118613 | 1.105739 | -0.012874 | 0.512641 | 0.508642 | -0.003998 |
| All validation paths | Matched interaction-off control | 0.784580 | 0.764721 | -0.019859 | 0.519081 | 0.510224 | -0.008857 |
| Transition only | Matched interaction-off control | 1.118613 | 1.100247 | -0.018365 | 0.512641 | 0.507536 | -0.005104 |

Negative deltas favor the reservoir correction. The selected model gives favorable pooled and transition-stratum deltas, but the matched interaction-off reservoir is stronger. We therefore do not attribute the gain specifically to Rydberg interactions.

**Figure 2 source:** pooled mean transition path from the archived residual no-intercept palindrome result. The correction changes sign across the forecast horizon, demonstrating why pooled metrics can conceal transition-specific behavior.

## 3. Case GE151 and Aquila observable transfer

Case GE151 (MERV, fold 8) was selected before hardware analysis as a strong illustrative transition. It is not presented as a representative average case.

| Window | HAR QLIKE | QRC QLIKE | Delta QLIKE | HAR RMSE | QRC RMSE | Delta RMSE | QRC closer horizons |
|---|---:|---:|---:|---:|---:|---:|---:|
| H1-H10 | 0.969205 | 0.344849 | -0.624356 | 0.505496 | 0.417252 | -0.088244 | 5/10 |
| H5-H10 | 1.476956 | 0.367573 | -1.109382 | 0.599808 | 0.421605 | -0.178203 | 3/6 |

The original simulator schedule could not be submitted unchanged to Aquila. A hardware-native six-atom, 40-step sequence was used to test matched observables at three probe times.

| Probe | Usable shots | Retention | Slope | Correlation | RMSE | Reported cost |
|---|---:|---:|---:|---:|---:|---:|
| Pooled, 63 observables | - | - | 1.068 | 0.943 | 0.0439 | - |
| 10-step | 955/1000 | 95.5% | 1.237 | 0.977 | 0.0547 | 1030 |
| 20-step | 960/1000 | 96.0% | 0.937 | 0.971 | 0.0322 | 1030 |
| 40-step | 966/1000 | 96.6% | 1.116 | 0.964 | 0.0420 | 1030 |

The 3,000-shot hardware study retained 2,881 shots. It supports observable transfer but is not an end-to-end hardware forecast.

**Figure 3 sources:** Case GE151 exact-simulator path and simulator-to-Aquila observable scatter from the archived hardware assembly.

## 4. Required Phase 3 benchmarks

| Study | Protocol | Primary result | Interpretation |
|---|---|---|---|
| MNIST | 2,000 train / 1,000 official test; six atoms; 63 occupation/pair features | QRC accuracy 0.663, macro-F1 0.656; matched two-channel logistic accuracy 0.732, macro-F1 0.731 | Nonlinear expressivity, but no advantage over the matched classical input model |
| Noise | T1, T2, depolarizing, and combined channels; clean readout frozen | Worst relative feature MAE 0.00525; minimum correlation 0.999978; no observed forecast degradation | Strong robustness under the tested local Markovian channels; not a calibrated Aquila model |
| Scaling | Exact 5-12 atoms; resources through 20; fixed six-feature readout | Best QLIKE at 9 atoms (1.08500); best RMSE at 11 atoms (0.51546); 12-atom runtime 11.0 s | Performance is non-monotonic; larger Hilbert space does not guarantee a better compact representation |
| Finite shots | 100-20,000 shots per sample/probe; ten seeds; exact readout frozen | 10,000 shots preserved the aggregate transition/control direction in all seeds; 20,000 gave mean correction correlation 0.982 | Below 1,000 shots is unreliable; 20,000 is preferred for horizon-resolved warning profiles |

**Figure 4 sources:**

```text
results/transition_forecasting/qrc/palindrome_noise_assay/palindrome_noise_primary_001/plots/
results/transition_forecasting/qrc/palindrome_scaling_assay/palindrome_scaling_primary_001/plots/
results/transition_forecasting/qrc/palindrome_shot_assay/palindrome_shots_primary_001/plots/
```

The optional position-encoded MNIST study improved QRC accuracy from 0.650 at five atoms to 0.765 at nine atoms, but atom count and input dimension co-varied and a random-tanh baseline remained stronger. It is diagnostic, not evidence of quantum advantage.

## 5. Resources, limitations, and conclusion

### Resource envelope

- Exact statevector scaling was measured from 5 to 12 atoms.
- The 20-atom raw batched state array is approximately 0.75 GiB for 48 samples; actual peak memory is higher.
- Empirical runtime extrapolation for the same panel is approximately 7,695 seconds at 20 atoms.
- Ten thousand shots per sample/probe is the observed aggregate transition-direction threshold; twenty thousand is preferred for the horizon profile.

### Limitations

- The interaction-off control performs better than the selected interaction-on model.
- The financial forecast remains primarily simulator-based.
- Aquila validates observables under an adapted schedule, not the complete forecasting pipeline.
- Noise, scaling, and shot assays use one bounded development fold and no reserved test rows.
- The transition-control path mean is small because early and later horizon effects have opposite signs.
- MNIST demonstrates expressivity but not superiority over matched classical baselines.

### Conclusion

The completed system is a reproducible simulator-first QRC pipeline with a neutral-atom hardware-transfer demonstration and explicit failure boundaries. It modestly improves HAR on pooled and transition-stratum simulator metrics, produces interpretable horizon-dependent corrections in selected crises, remains stable under the tested noise channels, exhibits non-monotonic reservoir-size scaling, and requires approximately 10,000-20,000 shots per probe for reliable transition-direction recovery. The strongest defensible claim is not generic quantum advantage: it is that a compact palindromic Rydberg reservoir can generate stable nonlinear temporal observables that complement a causal volatility baseline and transfer with high fidelity to neutral-atom hardware.

## Primary run IDs

```text
mnist_palindrome_primary_001
palindrome_noise_primary_001
palindrome_scaling_primary_001
palindrome_shots_primary_001
```

## References

1. Fujii, K. and Nakajima, K. (2017). Harnessing disordered-ensemble quantum dynamics for machine learning. *Physical Review Applied* 8, 024030.
2. Kornjaca, M. et al. (2024). Large-scale quantum reservoir learning with an analog quantum computer. arXiv:2407.02553.
3. Li, Q., Mukhopadhyay, C., Bayat, A., and Habibnia, A. (2026). Quantum Reservoir Computing for Realized Volatility Forecasting. arXiv:2505.13933v2.
4. Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics* 160, 246-256.
5. Zhu, Y. et al. (2025). Practical few-atom quantum reservoir computing. *Physical Review Research* 7, 023290.
