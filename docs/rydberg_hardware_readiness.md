# Rydberg / qBraid hardware-readiness workflow

Scope: operational readiness only. This does **not** redesign the volatility QRC model. It verifies that qBraid organization context, credentials, Aquila access, AHS program construction, discretization, job submission, and result retrieval work.

## Current working hypothesis

Translate the Phase 2 reservoir idea into the neutral-atom setting only far enough to learn the hardware surface:

1. Run a tiny AHS chain/lattice locally.
2. Discretize against QuEra Aquila device properties.
3. Submit one low-shot hardware job only after qBraid org/credit context is confirmed.
4. Compare simulator vs hardware readout features: occupation counts, empty-site/loss fraction, mean Rydberg excitation, and shot variability.
5. Treat noise as measured reservoir behavior, not automatically as an error source. Do not claim benefit until repeated runs show stable regularization or useful state dispersion.

## Required qBraid checks before hardware

In qBraid Lab or local shell:

```bash
qbraid --version
qbraid account --help
qbraid devices --help
qbraid jobs --help
```

Account-side checks:

- switch to the GIC/qBraid challenge organization, not personal `qBraid Default`;
- verify the 10,000-credit balance is visible under that organization;
- verify QuEra/Aquila is visible in available devices or via Amazon Braket integration;
- generate/configure API key if running locally: `qbraid configure`;
- do not submit hardware from the wrong organization.

## Minimal probe

```bash
python scripts/hardware/qbraid_ahs_readiness_probe.py --check-qbraid --simulate --shots 50
```

Then device property check, still no quantum task:

```bash
python scripts/hardware/qbraid_ahs_readiness_probe.py --check-qbraid --device-check
```

Only after the above works:

```bash
python scripts/hardware/qbraid_ahs_readiness_probe.py --hardware --shots 50 --out artifacts/ahs_probe_hw.json
```

The script refuses more than 100 hardware shots in readiness mode.

## What the probe records

- qBraid CLI availability and help surface;
- Aquila ARN and device-property excerpts;
- AHS IR preview;
- local simulator counts;
- hardware metadata if submitted;
- top measured bitstrings over `{d,u,e}`:
  - `d`: ground/down;
  - `u`: Rydberg/up;
  - `e`: empty/lost site;
- empty-site shot fraction;
- mean Rydberg excitations.

## Aquila / AHS constraints that matter for the project

- Aquila is analog Hamiltonian simulation, not gate-circuit QRC.
- User controls atom positions, global Rabi amplitude, global phase, global detuning, and, when available, experimental local detuning.
- Rydberg interaction strength is fixed by device physics and pairwise distance.
- Programs must be discretized against device resolution before hardware submission.
- Local detuning is experimental/request-gated and can worsen decoherence; omit it unless needed.
- Empty-site/preparation imperfections are part of the hardware readout and should be logged.

## Translation target from Phase 2

Minimal translation should preserve only the reservoir idea:

- input encoding -> geometry and/or pulse parameter schedule;
- reservoir state -> sampled Rydberg occupation distribution/features;
- readout -> classical regression/classification already used in Phase 2;
- memory -> must be explicit, probably via classical sequential feature construction or repeated AHS calls, because one Aquila AHS shot is not an automatically persistent recurrent state.

Do not attempt broad model redesign until the probe establishes what is operationally cheap, stable, and retrievable.
