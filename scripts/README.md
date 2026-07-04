# Script organization

The `scripts/` tree is organized by purpose rather than model family.

## Active root entry points

Scripts kept directly under `scripts/` are current Phase 3 operational runners or preparation tools.

## Subdirectories

- `reference/` — Phase 2 reference runs and prediction exports retained for comparison.
- `exploratory/` — model-development experiments that are not canonical benchmark entry points.
- `diagnostics/` — mechanism, scaling, ordering, geometry, and ablation diagnostics.

Future canonical comparison and parameter-search entry points should remain at the root. Experimental scripts should be placed in the appropriate subdirectory from the start.
