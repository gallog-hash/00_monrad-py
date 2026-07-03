---
name: per-axis-recovery-rejected
description: Per-axis telescope hit recovery was evaluated and dropped — not worth the complexity
metadata:
  type: project
---

**Decided 2026-06-17: per-axis hit recovery is NOT worth keeping.** The local
`feat/per-axis-recovery` branch (was `f878d9e`, never pushed) was force-deleted
from `main` after evaluation. The shipped behaviour is the **single-axis**
recovery already on `main` (PR #9, merge `c92f2f5`).

## What it was
Generalised `disambiguate_telescope_hits` from plane-level to **per-axis**:
X and Y recovered independently on their own anchors, so two planes that each
failed on the OPPOSITE axis could both recover (each axis keeps 2 anchors).
Required a data-model change in `decode_position` (a resolved axis of an
`unresolved` hit stored as a coordinate with `candidates=None` ⇒ anchor).

## Why dropped — diminishing returns
- Inliers 933 → **999 (+7%)** vs the single-axis step's 3.1×. Pool grew
  12358 → 18215 (+47%) but ~96% of the extra opposite-axis tracks die at the
  Stage-5 track-χ² gate: each axis pinned by only 2 anchors → geometrically
  weak, no χ² redundancy on the filled coordinate.
- The candidate match is **nearest-only** within a 15 mm window. On real
  `testLab_20210723`: 528 / 56398 recoverable axes (~0.9%) had a 2nd candidate
  *also* inside the window — predictions landing midway between two physical
  channels 20 mm apart, resolved by a sub-mm coin-flip (e.g. coincidence #592,
  Y/plane2, gap 0.45 mm). A gap-guard could clean these but the total upside is
  tiny. (Measured 2026-06-17 with a one-off corrected-frame `_fill_axis`
  replay over the coincidence stream; script not retained.)

## Cluster-width extension is branch-independent (the deciding insight)
`cluster` quality = **one ribbon × ≥2 ADJACENT fibers** (contiguous run);
multiple adjacent *ribbons* break contiguity → `unresolved` instead. Three
adjacent fibers ALREADY decodes as a width-3 cluster on `main` today — no code
change needed, and `_decode_axis`/`_reconstruct_coord` are identical on both
branches. The per-axis vs per-plane choice only governs **unresolved-hit
recovery**, never cluster formation. So widening accepted fiber clusters does
not require (or favour) the per-axis branch. Widening the *ribbon* dimension
would change what counts as unresolved, but is a `_reconstruct_coord` change,
still orthogonal to per-axis vs per-plane.

See [[dataset-testlab-20210723]] and [[testlab-20210723-plane-z-order]]
(`--z-tel 0 -1340 -670`).
