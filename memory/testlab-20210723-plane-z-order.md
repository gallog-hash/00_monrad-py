---
name: testlab-20210723-plane-z-order
description: Telescope file-columns are NOT stacked in z order — col 1 is the far plane
type: project
---

For dataset [[dataset-testlab-20210723]], the three telescope planes are
**not** stacked in file-column order. Physical z-order of the columns is
**0 → 2 → 1**: column 0 = z 0, column 2 = the true middle, column 1 = the
far plane. Confirmed real gap is 670 mm, so the correct CLI is
`--z-tel 0 -1340 -670`, not the default `0 400 800`. (Any same-ordered
even spacing, e.g. `0 800 400`, recovers the identical in-plane pose; only
the gap sets the absolute z_p — see [[dataset-testlab-20210723]].)

**Why (the original bug, now fixed):** `stage4.fit_telescope_alignment`
originally hardcoded the tiltable middle plane as file-column index 1
(`if k == 1`), and the docstring assumed hits arrive "in z order". With the
default
`z=0 400 800`, column 1 is told it sits at z=400; the real hits don't
match, so the middle-plane fit invents a nonphysical `delta_z ≈ +1090 mm`
(stable across all `tot_thresh`), the track model is wrong, and Stage 5
starves (<30 clean coincidences, SKIPPED). Passing `0 800 400` first
revealed this: even pre-fix (`k==1`) Stage 5 converged (n_inliers≈132) but
with `delta_z`≈−35 mm misattributed to the far plane (col 1). The committed
`k==mid` fix instead puts a near-zero offset on the true middle (col 2);
the real-geometry result is in [[dataset-testlab-20210723]]. Stage 3 hit
quality is byte-identical regardless of z, proving the difference is
geometry only.

**How to apply:** Run this dataset with `--z-tel 0 -1340 -670`. The code fix
is done and merged to `main`: Stage 4 now picks the middle plane by z-order
(`mid = argsort(z)[1]`) instead of hardcoding index 1, so column order and z
order need not coincide (PR #8, merge commit 3805500; test
`TestMiddlePlaneByZOrder`).
