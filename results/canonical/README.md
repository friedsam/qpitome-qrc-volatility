# Canonical comparison results

Only common-protocol comparisons belong here.

Required properties:

- same purged walk-forward folds
- explicit valid-fold policy for q90 and q95
- common metric implementation
- per-fold outputs plus aggregate summaries
- resource/cost metadata
- exact distinction between exact-state, finite-shot, and hardware results

## In-repo canonical outputs

The authoritative in-repo outputs are:

- aggregate metrics
- per-fold metrics
- run manifests
- missing-model summaries where applicable

These files are retained because they capture the canonical scientific comparison and its provenance compactly.

## Row-level prediction exports

Canonical row-level prediction CSVs are reproducible generated artifacts and are not retained in the active Git tree.

On July 7, 2026, 12 tracked prediction CSVs were archived off-repo as:

`canonical_predictions_202607.tar.gz`

The archive was created from repository commit:

`0e517a1c7a26907ceb9d386838f6763f2cd3f0dd`

It contains:

- `results/canonical/current/predictions_*.csv`
- `results/canonical/segments/*/predictions_*.csv`

The archive includes an internal manifest and an external SHA-256 checksum.

Untracked experimental outputs were excluded from that archive.

## Scope

No historical fixed-split result should be copied here merely because it is strong.

Canonical promotion requires evaluation under the common protocol and explicit provenance.
