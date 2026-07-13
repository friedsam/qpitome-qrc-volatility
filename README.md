# QPITOME QRC Volatility

Phase 3 Global Industry Challenge project for qBraid / MITRE / JonesTrading.

## Objective

Build a reproducible prototype for financial time-series intelligence using compact classical baselines, quantum reservoir computing, and hardware-aware validation. The repository includes volatility forecasting, regime analysis, transition modeling, Day 5 branching experiments, Rydberg reservoir studies, diagnostics, and reproducibility infrastructure.

## Core pipeline

1. Public financial time-series data
2. Causal rolling-window preprocessing
3. Volatility, regime, and transition targets
4. Compact classical baselines
5. QRC and Rydberg reservoir feature extraction
6. Classical readout and warning heads
7. Purged and walk-forward evaluation
8. Noise, shot-budget, and hardware-feasibility checks
9. Mechanism and falsification diagnostics
10. Reproducible result manifests and documentation

## Repository status

The project is currently consolidating Phase 3 results and repository structure on branch `phase3-refactor`.

Result areas already flattened and script-aligned:

```text
results/modeling/weekly_regimes/
results/modeling/transition_onset/
results/modeling/transition_destination/
results/modeling/day5_branching/baseline/
```

The Day 5 static Rydberg outputs remain present but have not yet been flattened. Several historical Day 5 output families, including the distributed spatial Rydberg shard and merge outputs, are missing or have unverified historical locations.

## Documentation

- [Project milestones](docs/milestones.md)
- [Results layout, provenance, and known missing artifacts](docs/results_layout_and_provenance.md)
- [Phase 3 Rydberg results summary](docs/phase3/results/phase3_rydberg_results_summary.md)
- [Rydberg temporal reservoir design note](docs/phase3/rydberg_temporal_reservoir_design_note.md)

## Provenance rule

Current script defaults may have been introduced during later refactoring and must not automatically be treated as evidence of where historical runs were originally stored. Expensive or distributed reruns should record the full command, Git commit, machine, shard identity, output directory, and merge manifest.
