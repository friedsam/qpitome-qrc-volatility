# Hardware Access Audit

Purpose: determine which qBraid-accessible platforms are usable beyond simulation and choose a primary + backup path.

| Provider | Access status | Real QPU? | Simulator? | SDK/path | Native model | Fit for QRC | Cost/credits | Queue | Notes |
|---|---|---|---|---|---|---|---|---|---|
| QuEra | TBD | TBD | TBD | TBD | neutral atom | high if real access | TBD | TBD | conceptually strong for QRC |
| IonQ | TBD | TBD | TBD | TBD | trapped ion gates | good | TBD | TBD | likely safe first hardware demo |
| Rigetti | TBD | TBD | TBD | TBD | superconducting gates | good noise study | TBD | TBD | hardware-noise relevance |
| OQC | TBD | TBD | TBD | TBD | superconducting gates | plausible | TBD | TBD | check availability |
| IQM | TBD | TBD | TBD | TBD | superconducting gates | plausible | TBD | TBD | check availability |
| IBM | external | yes with IBM account | yes | Qiskit | superconducting gates | backup only | external | TBD | not assumed through qBraid |

Decision rule:
1. Real QPU access enabled
2. Tiny circuit can run cheaply
3. Compatible with QRC circuit model
4. Queue friction acceptable
5. Strong portfolio value
6. Conceptual fit
