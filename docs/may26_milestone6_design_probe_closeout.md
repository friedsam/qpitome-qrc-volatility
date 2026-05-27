# May 26 — Milestone 6 Closeout: Focused QRC Design Probe

## Goal

Run one focused design probe that directly supports a QRC architecture claim for the Phase 2 volatility-forecasting prototype.

The original preferred probe was an observable ladder (`Z` versus `Z+X` versus `Z+X+ZZ`). After the May 25 baseline and subsequent ESN comparison, the most informative design weakness shifted to **encoding and within-window trajectory processing**. The focused design probe was therefore reframed around trajectory-aware QRC input encoding and readout selection.

## Design question

Does QRC underperform because the reservoir dynamics are weak, or because the input encoding fails to expose the relevant volatility trajectory structure?

The reset-window ESN provided the critical diagnostic: it also resets state for each 40-day sample window, yet substantially outperforms the reset-window QRC. Therefore, cross-window carryover is not the immediate explanation. The more direct issue is that ESN processes the full within-window path recurrently, while QRC initially saw sparse raw anchor snapshots.

## Probe performed

The final focused probe tested trajectory-aware QRC encoding while keeping the Hamiltonian and reservoir structure fixed:

```text
6-qubit full-topology TFIM-QRC
PCA-6 volatility features
40-day lookback
10 recent temporal anchors
leaky-integrated PCA input, leak = 0.3
3 Trotter steps per anchor
3 virtual nodes per anchor
evolution_time = 0.5
ZXZZ observables
disorder_strength = 0.20
train-only winsorization
train-only feature selection
ridge readout on log target
```

A final bounded sensitivity pass varied only:

```text
angle map: linear clipped versus tanh angle maps
selected features: top_k ∈ {80, 120, 160, 240}
ridge alpha ∈ {1000, 3000, 10000}
```

This remained within the intended Milestone 6 scope: one focused design axis, no broad architecture search.

## Compact result table

| Model / setting | Test RMSE ↓ | Test QLIKE ↓ | Test MZ R² ↑ | Test corr ↑ | Test pred std | High-vol recall ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Earlier QRC reference: 6 even anchors, top-120, alpha=3000 | 0.1005 | -2.0454 | 0.0961 | 0.3099 | 0.0404 | 0.3238 |
| Leaky input QRC: 10 recent anchors, top-120, alpha=3000 | 0.0961 | -2.2246 | 0.1664 | 0.4079 | 0.0505 | 0.5492 |
| Final QRC: leaky input, linear angle, top-240, alpha=1000 | **0.0951** | **-2.2297** | **0.1889** | **0.4346** | **0.0548** | **0.5861** |
| ESN reference | 0.0881 | -2.4553 | 0.4475 | 0.6690 | 0.1100 | 0.6885 |

## Interpretation

The probe supports a clear design claim:

```text
For this volatility task, QRC performance is strongly limited by input encoding and within-window trajectory representation. Leaky trajectory encoding makes the sparse QRC anchor inputs more informative and substantially improves high-volatility detection.
```

The improvement was largest on the failure mode identified by the ESN comparison: compressed predictions and low recall for high-volatility regimes.

Key changes from earlier QRC to final QRC:

```text
RMSE:             ~0.1005 → ~0.0951
MZ R²:            ~0.096  → ~0.189
correlation:      ~0.310  → ~0.435
prediction std:   ~0.040  → ~0.055
high-vol recall:  ~0.324  → ~0.586
```

The final QRC still does not beat the ESN, especially on continuous calibration and dynamic range. ESN prediction variance is still approximately twice the QRC prediction variance. However, QRC now recovers much more of the high-volatility regime signal and is no longer merely a flat compressed predictor.

## Architecture justification

### Hamiltonian

The selected Hamiltonian remains a full-topology TFIM-style reservoir with fixed disorder:

```text
ZZ interaction layer + transverse X field + deterministic disorder
```

Rationale:

- TFIM is a standard interacting quantum many-body model.
- Full topology increases feature mixing relative to a chain.
- Fixed disorder breaks symmetry without training the reservoir.
- Short evolution time (`t = 0.5`) was empirically supported: longer evolution increased rank but degraded forecast signal.

### Encoding

Encoding is the main design lesson from Milestone 6.

The initial raw angle encoding was too sparse in time. The final encoding uses:

```text
train-only PCA-6 → 40-day windows → leaky integration → recent anchor selection → angle encoding
```

Rationale:

- Recent-biased anchors improved high-volatility recall relative to evenly spaced extra anchors.
- Leaky integration approximates within-window trajectory memory while preserving the six-dimensional input and six-qubit resource target.
- Tanh-angle encoding did not outperform clipped-linear angles in the final bounded probe, so the final model keeps the simpler clipped-linear map.

### Readout

The final readout uses virtual-node `ZXZZ` observables, train-only winsorization, train-only top-feature selection, and ridge regression on the log target.

The final setting selected more useful features than earlier probes:

```text
top_k = 240
alpha = 1000
```

This suggests that leaky input made a larger fraction of the reservoir features useful, so the earlier top-120 readout was too restrictive.

### Hybrid integration

The resulting pipeline is a hybrid classical/QRC architecture:

```text
market features → chronological split → train-only PCA → leaky trajectory preprocessing → QRC reservoir → classical readout → benchmark diagnostics
```

This is not a quantum-advantage claim. It is a controlled architecture study showing which QRC design components matter and where the remaining gap to classical ESN remains.

## Resource and platform notes

The final QRC remains small enough for exact statevector simulation:

```text
6 qubits
state dimension = 64
10 anchors
3 Trotter steps per anchor
3 virtual-node readouts per anchor
ZXZZ readout = 17 observables per virtual node
raw QRC features = 10 × 3 × 17 = 510
selected readout features = 240
```

This supports the Phase 2 light-touch prototype boundary. A Phase 3 resource plan should examine:

- shot-based readout stability;
- circuit export through qBraid-compatible provider workflows;
- noise sensitivity;
- streaming/carryover variants;
- larger or hardware-aware reservoir layouts.

## Rubric criteria addressed

### Theoretical & analytical justification

Addressed by linking observed failures to specific architecture choices:

- sparse raw anchors lost trajectory information;
- leaky trajectory encoding improved high-volatility recall;
- final QRC still underpredicts high-volatility magnitudes, defining the next theoretical bottleneck.

### Platform justification & resources

Addressed by keeping the final prototype within a six-qubit exact-simulation resource envelope and documenting the next shot/noise/provider steps for Phase 3.

### QRC architecture design

Addressed through controlled evolution from raw angle-encoded TFIM-QRC to a trajectory-aware leaky-input TFIM-QRC with virtual-node readout and train-only selected ridge readout.

## Stop condition

Milestone 6 is closed.

The probe supports the design choice that trajectory-aware encoding is necessary for this task. It also gives a clear Phase 3 fallback: if stronger encodings or streaming QRC do not close the remaining calibration gap, the QRC contribution should be framed as complementary regime detection or residual modeling rather than standalone ESN replacement.

## Final milestone conclusion

```text
The Phase 2 QRC prototype is working and materially improved. It does not beat ESN, but it now has a rigorous design narrative: ESN diagnosed the missing within-window trajectory processing; leaky QRC encoding repaired much of the high-volatility recall failure; remaining gaps define concrete Phase 3 extensions.
```
