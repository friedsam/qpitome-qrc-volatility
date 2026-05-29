# May 28 — Milestone 8: Platform, Resource, and Phase 3 Execution Plan

Date: May 28, 2026  
Track: A — Financial volatility and regime-transition forecasting  
Status: planning artifact; no Phase 3 studies implemented here

## Rubric target

The relevant rubric item is:

> Appropriateness of selected backends, qubit/depth/shot estimates, and integration plan.

This document therefore does four things explicitly:

1. explains why each selected backend class is appropriate for this specific QRC design;
2. gives concrete qubit/atom, depth/evolution, circuit-count, and shot estimates;
3. defines an integration plan from the current Phase 2 code to each backend path;
4. states fallback paths if hardware access, translation, or scaling fails.

The goal is not to request access to attractive hardware in general. The goal is to make clear what Phase 3 would run, why the selected hardware is needed, what resource scale is expected, and how the work remains reproducible through qBraid.

## 1. Hardware thesis

Phase 2 produced a working digital TFIM-QRC prototype for realized-volatility forecasting and a regime-warning layer. The strongest Phase 3 question is not simply whether the existing code can be executed somewhere. The question is which physical realization of a quantum reservoir is most appropriate for an industry-relevant volatility-warning task.

The proposed hardware hierarchy is:

1. **QuEra/Aquila-class neutral atoms — primary hardware-native QRC route.**  
   Neutral atoms best match the core QRC hypothesis: the processor's native many-body dynamics can act directly as the reservoir.

2. **IonQ-class trapped ions — targeted dense digital-QRC comparator.**  
   Trapped ions test whether high-connectivity gate-model hardware improves compact dense digital reservoir circuits relative to sparse superconducting layouts.

3. **IBM/Qiskit superconducting resources — accessibility and reproducibility benchmark.**  
   IBM is the most direct benchmark path from the current Qiskit-style circuit workflow and provides broad user accessibility, fake-backend resource estimates, and limited real-QPU validation.

This is not a three-way hardware wish list. Each backend answers a different technical question:

| Backend class | Question answered |
|---|---|
| Neutral atoms | Does native analog many-body dynamics provide a better QRC reservoir substrate? |
| Trapped ions | Does high connectivity help the dense digital QRC implementation? |
| IBM/superconducting | What is the accessible, reproducible gate-model baseline and routing/depth cost? |

## 2. Backend-specific justification

### 2.1 QuEra / Aquila-class neutral atoms: primary route

**Why this backend is appropriate.**  
QRC is fundamentally a reservoir-computing method: a fixed dynamical system transforms input histories into nonlinear high-dimensional features, followed by a simple classical readout. A neutral-atom/Rydberg platform is therefore not just another backend. It is the backend class most naturally aligned with the QRC mechanism, because programmable Rydberg/Ising dynamics can supply the reservoir directly.

**Specific match to this project.**

- Our Phase 2 QRC was motivated as a spin-system reservoir.
- The bottleneck observed in Phase 2 was limited feature expressivity / tail-amplitude compression.
- Neutral-atom dynamics offer a plausible way to test richer hardware-native reservoir dynamics without digitally compiling every pairwise interaction.
- The volatility-regime task benefits from nonlinear temporal feature maps; analog many-body dynamics are a scientifically motivated feature generator for this purpose.

**What must be translated.**  
The current TFIM-QRC code is not a direct Aquila program. Phase 3 must define:

- atom geometry;
- which market-history features control geometry, local detunings, Rabi parameters, pulse duration, or initial states;
- which measured bitstring statistics become reservoir features;
- how many market windows are submitted in the QPU pilot;
- how emulator and QPU features are compared.

**Why it is still the primary route.**  
The translation work is precisely the research value. If QRC is to show a real hardware advantage, the strongest route is not simply to compile a small digital circuit. It is to test whether hardware-native many-body dynamics can provide a useful reservoir for a nontrivial financial risk signal.

### 2.2 IonQ-class trapped ions: dense digital-QRC comparator

**Why this backend is appropriate.**  
The final Phase 2 QRC uses compact qubit counts but dense/full interaction options. On sparse superconducting hardware, dense interactions create routing and SWAP overhead. Trapped-ion hardware is therefore a relevant comparator because high connectivity can reduce routing overhead for dense digital reservoir circuits.

**Specific match to this project.**

- Tests the same digital QRC idea without requiring an analog Rydberg reformulation.
- Provides a realistic QPU path for 6-10 qubit compact reservoirs.
- Directly probes whether connectivity is a key resource for QRC.
- Lets us compare digital QRC feature stability under finite shots and real trapped-ion execution.

**Limit.**  
IonQ does not test the analog-reservoir hypothesis. It tests the best near-term digital dense-circuit variant of the QRC design.

### 2.3 IBM / Qiskit superconducting resources: benchmark and reproducibility path

**Why this backend is appropriate.**  
IBM is the most direct continuation of the current workflow because the Phase 2 circuit thinking is Qiskit-like. IBM fake/hardware-aware backends also let us estimate qubit layout, native gates, transpiled depth, two-qubit count, and routing overhead before spending QPU time.

**Specific match to this project.**

- Immediate resource characterization from the current implementation.
- Limited real-QPU validation with free or low-cost IBM access.
- Broad reproducibility: many reviewers/users can reproduce IBM/Qiskit resource estimates.
- Useful benchmark against which IonQ connectivity and QuEra analog dynamics can be compared.

**Limit.**  
Sparse superconducting topology may be a poor match for dense reservoir interactions. That is not a reason to exclude IBM; it is exactly why IBM is an informative baseline.

## 3. Resource estimates derived from the Phase 2 QRC design

The current final QRC configuration is a 6-qubit TFIM-style digital reservoir with PCA-6 input features, 40-day lookback windows, 10 temporal anchors, 3 Trotter steps per anchor, dense/full topology, and Z/X/ZZ observables. That configuration is acceptable for exact simulation but too large for a first QPU batch if naively compiled.

### 3.1 Naive digital circuit cost

For a 6-qubit full-topology ZZ layer:

- number of pair interactions: `6 * 5 / 2 = 15`;
- common gate-model decomposition per ZZ interaction: approximately `2 CNOT + 1 RZ` or backend-native equivalent;
- therefore one full ZZ Trotter step is approximately `30 two-qubit gates + 15 RZ`, plus single-qubit transverse-field rotations;
- with 10 anchors and 3 Trotter steps per anchor: `30 Trotter steps`;
- naive full digital version: approximately `900 two-qubit gates` before routing and transpiler optimization.

This estimate explains why Phase 3 should not submit the full final simulator configuration directly to a QPU. The QPU study must use representative reduced circuits.

### 3.2 Representative QPU circuit sizes

| Circuit class | Qubits | Anchors / layers | Approx. dense pair interactions | Approx. two-qubit gates before routing | Intended use |
|---|---:|---:|---:|---:|---|
| Minimal hardware check | 6 | 1 anchor, 1 step | 15 | ~30 | verify execution path and observable extraction |
| Small QRC pilot | 6 | 2 anchors, 1 step | 30 | ~60 | compare simulator vs hardware features |
| Main digital QPU pilot | 6-8 | 3-4 anchors, 1-2 steps | 45-112 | ~90-224 | targeted feature-stability study |
| Simulator-only digital sweep | 6-10 | 6-10 anchors, 1-3 steps | 90-405 | ~180-810 | scaling/resource estimate, not first QPU batch |

For IBM/superconducting hardware, routing overhead may increase two-qubit counts substantially. For IonQ/trapped ions, the same pair-interaction count is still relevant, but the layout/routing penalty should be smaller because connectivity is the resource being tested.

### 3.3 Neutral-atom analog estimates

Neutral-atom resource estimates are not expressed as gate depth. They are expressed as atom count, geometry, pulse-program count, evolution duration, and shots per program.

| Analog study | Atom count | Programs/windows | Shots/program | Purpose |
|---|---:|---:|---:|---|
| Formulation pilot | 16-25 atoms | 5-10 encoded windows | emulator first | test encoding/readout design |
| Small QPU reservoir pilot | 25-50 atoms | 10-20 windows | 500-1000 | compare emulator/hardware features |
| Expanded analog reservoir study | 50-100+ atoms | 20-50 windows | 1000-2000 | evaluate regime separability and stability |

The first neutral-atom QPU goal is not a full time-series backtest. It is to determine whether analog reservoir features are stable and informative for selected calm, warning, and crisis-like market states.

### 3.4 Shot-budget estimates

| Study type | Shots | Rationale |
|---|---:|---|
| Exact simulator | 0 / exact expectations | clean reference signal |
| Shot simulator, exploratory | 512 | quick estimate of sampling variance |
| Shot simulator, standard | 2048 | default feature-stability estimate |
| Shot simulator, high precision | 8192 | selected final circuits only |
| IBM limited QPU check | 1024-4096 | feasible for a few representative circuits |
| IonQ targeted QPU subset | 500-4000 | balance feature precision against queue/cost |
| QuEra/Aquila pilot | 500-2000 | estimate bitstring-feature stability per analog program |

Shot budgets will not be applied to every test sample. Phase 3 QPU runs will use selected windows representing calm, watch/warning, and crisis-like regimes.

## 4. Platform/resource table

| Role | Platform/backend class | Why appropriate | Qubit/atom estimate | Depth/evolution estimate | Shot estimate | Integration route | Fallback |
|---|---|---|---:|---|---:|---|---|
| Primary hardware-native QRC | QuEra/Aquila-class neutral atoms | Native Rydberg/Ising many-body dynamics can act directly as reservoir | pilot 16-50 atoms; expanded 50-100+ | analog pulse schedule; no gate depth; few programs first | 500-2000/program | qBraid/Braket/QuEra-compatible analog workflow; emulator before QPU | simplify geometry; emulator-only; digital QRC comparator |
| Dense digital comparator | IonQ-class trapped ions | Tests whether high connectivity improves compact dense digital QRC | 6-10 qubits | 1-4 anchors, 1-2 steps for QPU; deeper on simulator | 500-4000/circuit | Qiskit circuit source -> qBraid/Braket/IonQ execution or provider-native route | shot simulator; IBM resource baseline |
| Accessibility benchmark | IBM/Qiskit superconducting | Direct from current workflow; broad reproducibility; fake backend resource estimates | 6-10 qubits | report transpiled depth, two-qubit count, routing overhead | 1024-4096 for limited QPU check | Qiskit circuits -> IBM fake backend / Runtime if available | fake backend only; reduce dense interactions |
| Exact reference | Statevector simulator | clean QRC feature baseline | 6-10 qubits; 12 if feasible | full final config possible for small n | exact | current Python/QRC code | reduce qubits/anchors |
| Sampling reference | Shot simulator | finite-shot feature variance before QPU | 6-10 qubits | representative digital circuits | 512/2048/8192 | qBraid/Aer/Braket simulator depending backend | selected windows only |
| Selective noisy pilot | Density-matrix/noisy simulator | limited degradation estimate | 4-6 qubits | shallow circuits only | 512-2048 equivalent | simulator-specific | skip if O(4^n) cost dominates |

## 5. Integration plan

### 5.1 Shared artifacts across all backend paths

Phase 3 will define a common backend-facing interface:

```text
phase3_backend_config.yaml
scripts/export_representative_qrc_inputs.py
scripts/export_digital_qrc_circuits.py
scripts/run_backend_resource_report.py
scripts/collect_backend_observables.py
scripts/compare_backend_features.py
```

Common metadata saved for every backend run:

- backend name and provider;
- simulator/QPU flag;
- qubits or atom count;
- circuit depth or analog pulse/evolution description;
- native gate counts or analog-program parameters;
- shot count;
- selected market-window IDs;
- random seeds;
- package versions;
- raw counts/bitstrings;
- derived observables/features;
- downstream forecast/regime metrics for the selected subset.

### 5.2 IBM integration path

Purpose: resource and reproducibility benchmark.

1. Export representative QRC circuits in Qiskit.
2. Transpile to IBM fake/hardware-aware backend.
3. Record depth, two-qubit gate count, SWAP/routing overhead, basis gates, and layout.
4. Run 1-3 representative circuits on free/limited IBM QPU access if feasible.
5. Compare simulator and QPU-derived Z-basis observables.

Expected output: `results/backend_metadata/ibm_resource_report.csv` and a limited `ibm_qpu_observable_check.csv` if QPU access is used.

### 5.3 IonQ integration path

Purpose: high-connectivity digital QRC comparator.

1. Start from the same representative QRC circuits used in IBM resource analysis.
2. Convert/rebuild circuits for IonQ-compatible execution through qBraid/Braket/provider route.
3. Use a small fixed subset of market windows.
4. Run exact/shot simulator first.
5. Submit selected circuits to IonQ-class QPU if access/credits permit.
6. Compare feature stability, sampling noise, and warning-layer degradation.

Expected output: `ionq_digital_qrc_feature_stability.csv` and `ionq_vs_ibm_resource_comparison.csv`.

### 5.4 QuEra/Aquila integration path

Purpose: primary analog reservoir path.

1. Define analog encoding options: geometry-based, local-detuning-based, pulse-parameter-based, or reduced market-state class encoding.
2. Construct small representative analog programs for selected market windows.
3. Run emulator/simulator first.
4. Derive bitstring-statistics features: excitation counts, spatial correlations, local observables, pair correlations, or distribution summaries.
5. Train/read out on selected-window features or evaluate class separability between calm/warning/crisis-like states.
6. Submit small QPU program set if access permits.

Expected output: `quera_analog_reservoir_features.csv`, `quera_emulator_vs_qpu_feature_stability.csv`, and an analog reservoir feasibility summary.

## 6. Phase 3 milestone sequence

### Milestone 3.1 — Freeze reproducible dataset and benchmark package

- Freeze public data sources, preprocessing, train/validation/test splits, and feature definitions.
- Freeze classical baselines: persistence, HAR-like ridge, ESN reference, and optional GARCH-family comparator if time permits.
- Freeze Phase 2 QRC reference runs: QRC v0, final QRC, and regime-warning layer.

**Output:** reproducible qBraid-compatible repository with data instructions, fixed seeds, run scripts, and saved metric tables.

### Milestone 3.2 — Digital circuit resource characterization

- Build representative digital QRC circuits from the final Phase 2 architecture.
- Evaluate exact and shot simulator behavior.
- Transpile representative circuits to IBM/fake backend constraints.
- Report qubit count, native gates, depth, two-qubit gate count, routing overhead, shot budget, and runtime estimates.

**Output:** resource table showing what the current digital QRC requires before live QPU execution.

### Milestone 3.3 — Targeted trapped-ion digital QRC validation

- Run a small fixed subset on IonQ-class trapped-ion hardware if available.
- Use 6-10 qubits and selected calm, warning, and crisis-like market windows.
- Compare exact simulator, shot simulator, and QPU-derived reservoir features.
- Evaluate feature stability and downstream regime-warning degradation, not full production performance.

**Output:** trapped-ion digital QRC feasibility report.

### Milestone 3.4 — Neutral-atom analog reservoir formulation

- Translate the TFIM-QRC design into a Rydberg/Ising reservoir formulation.
- Define atom geometry, encoding variables, pulse schedule, and measured feature construction.
- Run emulator/simulator studies before QPU use.
- Test whether analog reservoir features separate calm, warning, and crisis-like windows.

**Output:** Aquila-class analog QRC formulation and emulator results.

### Milestone 3.5 — Neutral-atom QPU validation

- Submit a small number of neutral-atom reservoir programs if access permits.
- Compare hardware-derived reservoir features to emulator features.
- Evaluate whether analog dynamics reduce the expressivity/tail-compression bottleneck observed in digital QRC.

**Output:** primary Phase 3 hardware-native QRC result.

### Milestone 3.6 — Cross-platform comparison and final selection

- Compare IBM/superconducting, IonQ/trapped-ion, and QuEra/neutral-atom resource profiles.
- Compare feature stability, warning-layer performance, implementation cost, and reproducibility.
- Decide whether the most promising next route is analog neutral atoms, trapped-ion digital QRC, or simulator-first QRC refinement.

**Output:** final Phase 3 paper, qBraid workflow, and reproducibility package.

## 7. Risk and fallback table

| Risk | Impact | Detection point | Fallback |
|---|---|---|---|
| Neutral-atom translation is slower than expected | Primary hardware-native path delayed | Milestone 3.4 | Deliver emulator-level analog formulation; use IonQ digital validation as QPU result |
| Aquila/QPU access limited | Cannot run full analog hardware study | Milestone 3.5 | Use QuEra/Bloqade-style emulator and specify exact QPU-ready programs |
| Analog encoding weak | Neutral-atom features do not separate regimes | Milestone 3.4 | Try alternative encodings; reduce task to feature stability/class separability before full forecasting |
| Trapped-ion access limited or too costly | Digital dense-QRC QPU run reduced | Milestone 3.3 | Use shot simulator and IBM resource benchmark; run fewer windows/shots |
| IBM routing overhead too large | Superconducting path poor fit | Milestone 3.2 | Use this as evidence favoring trapped ions/neutral atoms; reduce interaction density |
| Noisy simulation too expensive | Full noise scaling infeasible | Milestone 3.2/3.3 | Use selected small circuits only; prioritize empirical shot/QPU comparisons |
| QRC remains tail-compressed | Weak crisis-like amplitude response | All modeling milestones | Add tail-aware readout/calibration; test richer encoding and observables; compare analog reservoir features |
| Classical reservoirs dominate | Harder to claim practical advantage | Cross-platform comparison | Reframe as hardware feasibility and feature-map study; identify where QRC features add complementary signal |
| qBraid conversion fails for one provider | Workflow reproducibility risk | Integration test per backend | Use provider-native SDK route and document backend-specific command path |

## 8. qBraid reproducibility plan

Phase 3 will package the work as a qBraid-executable workflow. The goal is for a reviewer to re-run headline results without reconstructing private environment assumptions.

Planned structure:

```text
repo/
  notebooks/
    phase3_01_data_and_baselines.ipynb
    phase3_02_digital_qrc_resource_characterization.ipynb
    phase3_03_ionq_digital_qrc_validation.ipynb
    phase3_04_quera_analog_qrc_formulation.ipynb
    phase3_05_platform_comparison.ipynb
  scripts/
    run_phase3_baselines.py
    export_representative_qrc_inputs.py
    export_digital_qrc_circuits.py
    run_backend_resource_report.py
    collect_backend_observables.py
    compare_backend_features.py
    export_reproducibility_bundle.py
  results/
    tables/
    figures/
    backend_metadata/
  docs/
    platform_resource_plan.md
    run_instructions_qbraid.md
    phase3_reproducibility_manifest.md
```

The workflow will be simulator-first. QPU cells will be isolated and parameterized so the same notebook can run in one of three modes:

1. exact/ideal simulator;
2. shot/noisy simulator or emulator;
3. live backend if credentials/access are available.

## 9. Submission-ready hardware paragraph

> Our Phase 3 hardware plan is hypothesis-driven. QuEra/Aquila-class neutral atoms are the primary route because QRC is naturally a hardware-native analog reservoir problem: programmable Rydberg many-body dynamics can serve directly as the reservoir feature map, rather than being digitally compiled into deep gate sequences. IonQ-class trapped ions provide a targeted digital comparator for compact dense reservoir circuits, testing whether high connectivity reduces the routing burden that affects superconducting devices. IBM/Qiskit backends provide the accessibility benchmark: they allow immediate resource characterization, reproducible compilation studies, and limited QPU validation for the current circuit formulation. This three-part plan directly addresses backend appropriateness, resource estimates, and integration: analog neutral atoms test the strongest QRC hypothesis; trapped ions test the best dense digital-circuit alternative; IBM provides the broadly reproducible superconducting baseline.

## 10. Sign-off criteria for Milestone 8

A judge should be able to see:

- why each selected backend is appropriate;
- which platform is primary versus comparator versus accessibility benchmark;
- the qubit/atom counts, depth/evolution estimates, circuit-count scale, and shot budgets;
- what will run first in Phase 3;
- what qBraid/provider integration path will be used;
- what happens if scaling, access, translation, or noise limits appear.

This plan deliberately avoids implementing Phase 3 studies in Phase 2. It defines the executable roadmap and resource justification needed to advance.
