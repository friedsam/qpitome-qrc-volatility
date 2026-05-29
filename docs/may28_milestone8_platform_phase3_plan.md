# May 28 — Milestone 8: Platform, Resource, and Phase 3 Execution Plan

Date: May 28, 2026  
Track: A — Financial volatility and regime-transition forecasting  
Status: planning artifact; no Phase 3 studies implemented here

## 1. Purpose

This document turns the Phase 2 QRC prototype into a concrete Phase 3 execution plan. The goal is not to request generic access to attractive hardware platforms. The goal is to justify each platform by the specific technical question it answers for quantum reservoir computing (QRC) applied to volatility-regime forecasting.

The Phase 2 results established four facts that drive the hardware plan:

1. The current QRC implementation is a working circuit-level spin-reservoir prototype.
2. Performance improved materially from the initial QRC to the final calibrated QRC.
3. The remaining bottleneck is not merely model plumbing; it is whether richer reservoir dynamics and hardware-native dynamics can produce more expressive, less amplitude-compressed features.
4. The strongest Phase 3 question is therefore hardware-specific: which physical realization of a quantum reservoir is most promising for an industry-relevant volatility-warning task?

The platform plan is staged but not timid. Neutral atoms are the primary scientific route because they best match the QRC hypothesis. Trapped ions are a targeted digital comparator because their connectivity may help the dense-reservoir circuit version. IBM/superconducting resources are included as an accessible benchmark and reproducibility path because the current implementation is Qiskit-aligned.

## 2. Hardware thesis

### Primary route: neutral atoms / QuEra Aquila-class analog reservoir

QRC is most naturally implemented on hardware whose own many-body dynamics act as the reservoir. A programmable neutral-atom/Rydberg system is therefore the most direct physical match to the QRC idea. Instead of digitally compiling every interaction into gate layers, the Phase 3 neutral-atom path asks whether native Rydberg/Ising dynamics can provide a useful reservoir feature map for volatility and transition-risk forecasting.

**Scientific question:**

Can hardware-native neutral-atom dynamics produce reservoir features that improve volatility-regime warning relative to digitally compiled reservoir circuits and strong classical reservoirs?

**Why this platform is needed:**

- It tests QRC in its strongest physical form: an analog dynamical system used directly as the reservoir.
- It may avoid deep digital Trotter circuits and routing overhead.
- It aligns with published neutral-atom QRC motivation and with the challenge-provided QuEra/Aquila QRC tutorials.
- It is the clearest route to a non-toy hardware story: the device dynamics are the model component being tested.

**Main technical challenge:**

The current TFIM-QRC code is not a direct Aquila program. Phase 3 must translate the digital TFIM reservoir idea into a Rydberg/Ising analog formulation: atom geometry, detuning/Rabi schedule, input encoding, measurement features, and readout protocol.

### Secondary scientific comparator: IonQ-class trapped-ion digital QRC

Trapped-ion hardware is not the same hypothesis as neutral atoms. It remains a gate-model implementation. Its value is that high connectivity may make compact dense digital QRC circuits much more natural than on sparse superconducting topologies.

**Scientific question:**

Does high-connectivity trapped-ion hardware preserve QRC feature stability and regime-warning signal for compact dense digital reservoir circuits better than topology-constrained superconducting hardware?

**Why this platform is needed:**

- The final QRC prototype uses a dense/full interaction option and repeated reservoir evolution; trapped-ion connectivity is directly relevant to this design.
- It keeps the circuit-level QRC formulation close to the current implementation while avoiding excessive routing overhead.
- It gives a fairer digital-QRC hardware test than using only sparse superconducting connectivity.
- It provides a near-term QPU execution path while the neutral-atom analog formulation is developed.

**Main technical challenge:**

The Qiskit-based circuit must be converted or rebuilt for IonQ/qBraid/Braket execution, and the hardware study must be limited to a fixed subset of circuits/windows because full QRC test-set evaluation would be shot- and queue-expensive.

### Accessibility and reproducibility benchmark: IBM superconducting / Qiskit-aligned path

IBM-style superconducting resources are not the primary scientific target for this QRC design. They are included because they provide the most direct, widely accessible resource-characterization path from the current Qiskit-like circuit representation.

**Scientific question:**

What are the depth, two-qubit gate count, routing overhead, shot budget, and reproducibility characteristics of the current digital QRC circuit on widely available superconducting tooling?

**Why this platform is needed:**

- It is the most direct continuation of the current circuit workflow.
- Fake/hardware-aware IBM backends provide useful topology, native-gate, and transpilation constraints.
- Limited free IBM QPU access can support representative hardware execution.
- IBM/Qiskit makes the reproducibility story stronger for judges and future users, because the tooling is widely known and accessible.

**Main technical challenge:**

Sparse superconducting topology may penalize dense QRC circuits through SWAP/routing overhead. That is a feature of the benchmark: it helps quantify why trapped-ion or neutral-atom paths may be better suited to QRC.

## 3. Platform and resource table

| Role | Platform/backend class | Purpose | Qubit / atom range | Depth / evolution estimate | Shot budget | Noise/error plan | Fallback |
|---|---|---|---:|---|---:|---|---|
| Algorithmic reference | Statevector simulator | Exact QRC feature generation and clean baseline | 6–10 qubits initially; 12 if feasible | 6–10 anchors; 1–3 Trotter/evolution steps; Z/X/ZZ or platform-native observables | exact expectations first | no noise; algorithmic signal only | reduce anchors/features/qubits |
| Sampling reference | Shot-based simulator | Estimate finite-shot degradation of reservoir observables | 6–10 qubits | same representative circuits as exact reference | 512, 2048, 8192 | compare observable variance and downstream warning metrics | restrict to selected windows |
| Hardware-native primary | QuEra/Aquila-class neutral atoms | Analog Rydberg/Ising reservoir feature map | small pilot: 16–50 atoms; scaling target: 50–100+ atoms if feasible | analog pulse/evolution schedules rather than gate depth; one or few pulse programs per encoded sample | 500–2000 shots per program initially | hardware noise assessed empirically by repeated runs and simulator/emulator comparison | reduce atom count; simplify geometry; return to digital TFIM-QRC |
| Digital dense-circuit comparator | IonQ-class trapped ions | Compact dense digital QRC with lower routing overhead | 6–10 qubits for targeted QPU subset | 1–3 reservoir layers / anchors in QPU study; simulator may test deeper | 500–4000 shots per circuit | compare exact/shot simulator/QPU feature stability; no full noise tomography | use shot simulator or IBM benchmark |
| Accessibility benchmark | IBM superconducting / fake backend + limited QPU | Qiskit-aligned transpilation, routing, depth, and reproducibility benchmark | 6–10 qubits | transpiled depth and two-qubit count reported for representative circuits | 1024–4096 shots for limited QPU check if feasible | fake backend/hardware-aware constraints; optional mthree/ZNE only if justified | use fake backend only; reduce interaction density |
| Optional noisy pilot | Density-matrix / noisy simulator | Small degradation study only | 4–6 qubits | shallow representative circuit only | 512–2048 equivalent shots | selective noise model; avoid full O(4^n) scaling | skip if runtime dominates; use empirical QPU comparison |
| Cross-framework execution | qBraid SDK / qBraid Lab | Provider-agnostic workflow and reproducibility | platform-dependent | records backend, circuit/resource metadata, shots | platform-dependent | common run metadata and fallback routing | direct SDK route if conversion fails |

## 4. Phase 3 milestone sequence

### Milestone 3.1 — Freeze reproducible dataset and benchmark package

- Freeze public data sources, preprocessing, train/validation/test splits, and feature definitions.
- Freeze classical baselines: persistence, HAR-like ridge, ESN reference, and optional GARCH-family comparator if time permits.
- Freeze Phase 2 QRC reference runs: QRC v0, final QRC, and regime-warning layer.

**Output:** reproducible qBraid-compatible repository with data instructions, fixed seeds, run scripts, and saved metric tables.

### Milestone 3.2 — Circuit-level resource characterization

- Build representative digital QRC circuits from the final Phase 2 architecture.
- Evaluate on ideal and shot simulators.
- Transpile representative circuits to IBM/fake backend constraints.
- Report qubit count, native gates, depth, two-qubit gate count, routing overhead, shot budget, and runtime estimates.

**Output:** resource table showing what the current digital QRC requires before live QPU execution.

### Milestone 3.3 — Targeted digital QPU validation

- Run a small fixed subset on IonQ-class trapped-ion hardware if available.
- Use 6–10 qubits and a small number of representative market windows, selected to include calm, warning, and crisis-like regimes.
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

- Compare digital superconducting, digital trapped-ion, and analog neutral-atom resource profiles.
- Compare feature stability, warning-layer performance, and implementation cost.
- Decide whether the most promising Phase 4 route is analog neutral atoms, trapped-ion digital QRC, or simulator-first QRC refinement.

**Output:** final Phase 3 paper, qBraid workflow, and reproducibility package.

## 5. Risk and fallback table

| Risk | Impact | Detection point | Fallback |
|---|---|---|---|
| Neutral-atom translation is slower than expected | Primary hardware-native path delayed | Milestone 3.4 | Deliver emulator-level analog formulation; use IonQ digital validation as QPU result |
| Aquila/QPU access limited | Cannot run full analog hardware study | Milestone 3.5 | Use QuEra/Bloqade-style emulator and specify exact QPU-ready programs |
| Trapped-ion access limited or too costly | Digital dense-QRC QPU run reduced | Milestone 3.3 | Use shot simulator and IBM resource benchmark; run fewer windows/shots |
| IBM routing overhead too large | Superconducting path poor fit | Milestone 3.2 | Use this as evidence favoring trapped ions/neutral atoms; reduce dense interactions |
| Noisy simulation too expensive | Full noise scaling infeasible | Milestone 3.2/3.3 | Use selected small circuits only; prioritize empirical shot/QPU comparisons |
| QRC remains tail-compressed | Weak crisis-like amplitude response | All modeling milestones | Add tail-aware readout/calibration; test richer encoding and observables; compare analog reservoir features |
| Classical reservoirs dominate | Harder to claim practical advantage | Cross-platform comparison | Reframe as hardware feasibility and feature-map study; identify where QRC features add complementary signal |
| qBraid conversion fails for one provider | Workflow reproducibility risk | Integration test per backend | Use provider-native SDK route and document backend-specific command path |

## 6. qBraid reproducibility plan

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
    export_digital_qrc_circuits.py
    run_backend_resource_report.py
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

Each backend-facing run will save:

- backend name and provider;
- simulator/QPU flag;
- qubit count or atom count;
- circuit depth or analog evolution schedule;
- native gate counts or pulse-program metadata;
- shot count;
- random seeds;
- date/time;
- package versions;
- resulting counts/observables;
- downstream forecast/regime metrics.

The workflow will be simulator-first. QPU cells will be clearly isolated and parameterized so the same notebook can run in one of three modes:

1. exact/ideal simulator;
2. shot/noisy simulator or emulator;
3. live backend if credentials/access are available.

## 7. Phase 2 submission framing

Use this as the concise submission version:

> Our Phase 3 hardware plan is hypothesis-driven. QuEra/Aquila-class neutral atoms are the primary route because QRC is naturally a hardware-native analog reservoir problem: programmable Rydberg many-body dynamics can serve directly as the reservoir feature map. IonQ-class trapped ions provide a targeted digital comparator for compact dense reservoir circuits, testing whether high connectivity reduces the routing burden that would affect superconducting devices. IBM/Qiskit backends provide the accessibility benchmark: they allow immediate resource characterization, reproducible compilation studies, and limited QPU validation for the current circuit formulation. This three-part plan separates the primary scientific claim from execution controls: analog neutral atoms test the strongest QRC hypothesis; trapped ions test the best digital dense-circuit alternative; IBM provides the broadly reproducible superconducting baseline.

## 8. Sign-off criteria for Milestone 8

A judge should be able to see:

- which hardware is requested and why;
- which platform is primary versus comparator versus accessibility benchmark;
- what qubit/atom counts, depths/evolution schedules, and shot budgets are expected;
- what will run first in Phase 3;
- what happens if scaling, access, or noise limits appear;
- how the workflow will be made qBraid-reproducible.

This plan deliberately avoids implementing Phase 3 studies in Phase 2. It defines the executable roadmap and resource justification needed to advance.
