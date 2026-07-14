# Data policy

This repository is intended to run from a fresh clone on local machines, SSH hosts, qBraid, and agent-controlled environments.

## Versioned in ordinary Git

Commit every small, essential, redistributable dataset required by an active canonical run, including:

- frozen raw public-market snapshots;
- canonical processed model inputs;
- transition labels and weekly panels;
- curated episode, landmark, and crisis-cluster tables;
- machine-readable manifests and checksums.

These files must not depend on undocumented copies from a particular computer.

## Not versioned in ordinary Git

Keep the following under ignored local directories or external versioned artifact storage:

- regenerable caches and intermediate matrices;
- large reservoir-state tensors;
- model checkpoints;
- bootstrap samples;
- bulky shot-level hardware output;
- restricted or licensed datasets;
- credentials and secrets.

Every external required artifact must still have a committed manifest containing its expected path, retrieval procedure, schema, and checksum.

## Directory roles

- `data/raw/`: immutable source snapshots used by retained pipelines.
- `data/processed/`: canonical tables consumed by retained model runners.
- `data/manifests/`: provenance, schema, row-count, date-range, producer, and SHA-256 records.
- `data/cache/`, `data/tmp/`, `data/downloads/`, `data/intermediate/`, `data/local/`: ignored machine-local material.

## Reproducibility requirement

A fresh clone must be able to validate the canonical data without manual copying. Processed tables should have deterministic producer scripts where possible, but the exact canonical processed inputs used for reported results should also be versioned when compact.
