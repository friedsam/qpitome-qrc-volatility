# Phase 2 TFIM equivalent — retained reference

The active reset branch must retain one faithful TFIM stream equivalent to the final Phase 2 model. This is a reference/control model, not a new architecture search.

Historical source on GitHub archive branch:

- branch: `archive/pre-reset-20260705`
- runner: `scripts/run_canonical_tfim.py`
- core implementation: `src/qpitome_qrc/qrc/tfim_reservoir.py`
- original notebook lineage: `phase2-volatility-regression-qrc:notebooks/phase2_qrc_final_encoding_readout_probe.ipynb`
- historical run name: `linear_clip_top240_alpha1000`

## Exact retained configuration

- full input feature set -> train-only `StandardScaler` -> PCA6;
- 40-step input window;
- leaky integration with leak = 0.3;
- clipped linear angle encoding;
- 10 recent temporal anchors;
- 6-qubit fully connected TFIM;
- 3 Trotter steps per anchor;
- 3 virtual nodes per anchor;
- Z/X/nearest-neighbor ZZ observable readout across anchors;
- fixed disorder strength = 0.20;
- train-only 1st/99th percentile clipping of reservoir features;
- train-only absolute feature-target correlation ranking;
- top 240 reservoir features;
- train-only `StandardScaler` on selected reservoir features;
- `Ridge(alpha=1000)` readout;
- log target;
- seed = 42.

## Reset rule

The active TFIM implementation will be rewritten against the new common dataset/protocol interfaces. The physics and readout above remain frozen unless a change is explicitly declared as a new ablation.

The old runner will not be copied wholesale because it depends on removed Phase 3 comparison infrastructure and legacy dataset paths. The archive remains the source of truth for implementation parity.

## Role in the rebuilt project

1. cheap classical sanity baseline;
2. ESN — primary classical reservoir baseline;
3. **Phase 2-equivalent TFIM — frozen quantum control/reference**;
4. Rydberg QRC — primary quantum architecture, moved to finite-shot and Aquila testing early.
