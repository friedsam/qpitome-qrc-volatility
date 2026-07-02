# Phase 3 Rydberg reservoir results summary

## Main frozen configuration

- total_time_us = 0.55
- anchors = 8
- anchor_policy = even
- reverse_anchors = true
- geometry = dual_chain
- encoding = plateau unless otherwise stated

## Purged walk-forward: Rydberg vs raw/product baselines

Median test metrics:

| model | RMSE | QLIKE | MZ R2 | q90 AUC | q90 AP | q90 F1 | q95 AUC | q95 AP | q95 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw_baseline | 0.069684 | -2.323642 | 0.303785 | 0.752793 | 0.361835 | 0.392157 | 0.640833 | 0.086933 | 0.146418 |
| raw_products | 0.078999 | -2.276560 | 0.269376 | 0.702338 | 0.346364 | 0.381356 | 0.664414 | 0.090995 | 0.110089 |
| rydberg_memoryless | 0.075006 | -2.298153 | 0.230818 | 0.787803 | 0.440956 | 0.335106 | 0.711080 | 0.143995 | 0.174840 |
| rydberg_shuffled | 0.076660 | -2.199081 | 0.156816 | 0.789564 | 0.273671 | 0.362460 | 0.644679 | 0.130854 | 0.141219 |
| rydberg_temporal | 0.074866 | -2.289833 | 0.194562 | 0.816297 | 0.439800 | 0.350877 | 0.665797 | 0.212225 | 0.163604 |

Interpretation:

- Explicit second-order raw anchor products do not explain away the temporal Rydberg result.
- Temporal Rydberg substantially improves q90 AUC/AP and q95 AP relative to raw_products.
- The strongest temporal Rydberg market-side result is rare-event AP, especially q95 AP.
- Temporal Rydberg does not dominate AUC, regression metrics, or the memoryless ablation.
- Memoryless remains competitive and is stronger on median q95 AUC.

## Mechanism diagnostic: cross-time nonlinear memory

The synthetic temporal-memory diagnostic validates the mechanism:

- classical_additive: zero cross-time product capacity
- plateau_memoryless: zero cross-time product capacity
- plateau_temporal: positive cross-time product capacity with fading dependence on anchor separation
- classical_products: high/product-capacity ceiling control

Interpretation:

Temporal Rydberg dynamics can generate cross-time nonlinear features that memoryless plateau evolution cannot. This supports the reservoir-memory mechanism, but does not by itself establish market predictive advantage.

## Dual-chain geometry diagnostic

The dual-chain diagnostic supports a two-timescale geometry interpretation:

- mean within-slow absolute connected correlation = 0.0210
- mean within-fast absolute connected correlation = 0.0025
- slow / fast connected-correlation ratio = 8.39
- mean cross slow-fast absolute connected correlation = 0.00184
- cross / within connected-correlation ratio = 0.156

Interpretation:

The slow sublattice develops much stronger connected correlations than the fast sublattice, while cross-chain correlations remain nonzero but weaker. The geometry behaves like a weakly coupled two-timescale reservoir rather than a homogeneous atom array.

## Ramp encoding check

Ramp temporal median metrics:

| model | RMSE | QLIKE | MZ R2 | q90 AUC | q90 AP | q90 F1 | q95 AUC | q95 AP | q95 F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rydberg_temporal ramp | 0.075370 | -2.276503 | 0.182510 | 0.805721 | 0.406853 | 0.348525 | 0.665441 | 0.197093 | 0.164663 |

Compared with plateau temporal:

- plateau q90 AP = 0.440 vs ramp q90 AP = 0.407
- plateau q95 AP = 0.212 vs ramp q95 AP = 0.197

Interpretation:

Ramp encoding is viable and physically motivated, but under the frozen tt055/a8/reverse setting it does not improve rare-event AP over plateau encoding. It is a mechanistic extension/control, not the primary market result.

## Safe final claim

The temporal Rydberg reservoir has demonstrable cross-time nonlinear memory in controlled diagnostics and improves selected warning metrics, especially q95 AP, under purged walk-forward validation. The product-augmented raw baseline does not reproduce this result, suggesting that the useful signal is not captured by simple second-order anchor products. However, the evidence does not establish quantum advantage: ESN remains stronger overall, the memoryless ablation is competitive, and temporal Rydberg does not dominate AUC or regression metrics.
