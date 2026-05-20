# Phase 2 execution guardrails

Date: May 20, 2026

## Core Phase 2 interpretation

Phase 2 is judged primarily on design rigor and depth of justification, not final benchmark dominance.

The challenge expects light-touch prototyping only: small simulator experiments, e.g. 7–12 qubit reservoirs, sufficient to substantiate the core design claims.

Detailed benchmarking, full scaling studies, and full noise analysis belong to Phase 3.

## Practical operating rule

Do enough prototyping to support the design argument. Do not turn Phase 2 into a full performance-optimization sprint.

A useful Phase 2 prototype should answer one of these questions:

- Does the proposed Hamiltonian/encoding/readout pipeline execute end-to-end?
- Does the QRC feature map produce nontrivial predictive signal on the volatility forecasting target?
- Does the result plausibly compete with a simple classical baseline?
- Does a small reservoir-size or encoding-density pilot support the resource plan?
- Does a small shot/noise pilot expose feasibility constraints for Phase 3?

## Stop rule

Stop expanding an experiment once it supports or falsifies the design claim needed for Phase 2.

Do not spend Phase 2 time on:

- exhaustive hyperparameter searches;
- large qubit-count scaling studies;
- full noise-model characterization;
- final benchmark polishing;
- extensive LSTM/GARCH implementation unless the narrative requires it.

## Minimum useful Phase 2 prototype

Target: realized-volatility forecasting.

Classical baselines:

- persistence or HAR-like ridge;
- one compact ESN/regression baseline if feasible.

QRC prototype:

- 7–12 qubit TFIM/spin-system simulator, or smaller if runtime requires;
- compact multivariate input encoding;
- ridge/linear readout;
- RMSE/QLIKE/Mincer-Zarnowitz where feasible.

Resource pilot:

- at most one small comparison across reservoir size, encoding density, shot budget, or simple noise setting.

## Reminder phrase

Phase 2 asks: "Is this a rigorous, well-justified, executable QRC design worth scaling in Phase 3?"

It does not ask: "Did we already complete the final Phase 3 benchmark?"
