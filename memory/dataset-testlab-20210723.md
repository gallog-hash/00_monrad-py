---
name: dataset-testlab-20210723
description: The 0_testLab_20210723 real-data acquisition — layout, scale, quirks
type: reference
---

`data/0_testLab_20210723/` — a full-day (≈11:40–23:30) lab acquisition used
for testing the pipeline against real detector files.

Layout:
- `Base/`    — telescope, 3 planes (headers `[J11] [J13] [J12]`),
               148 `*.bin` + 148 `*_GPS.bin` pairs, header `20210723_113529_header.txt`.
- `Probe_0/` — probe, 1 plane, 148 pairs, header `20210723_113555_header.txt`.

Scale / behaviour observed (default `tot_thresh=1`, `tot_weights=False`):
- Stage 1: 1,461,697 telescope / 1,307,941 probe events, ~all GOOD;
  tel/probe ratio 1.118.
- Stage 2: 106,355 coincidences, mean cluster size 2.00.
- 20 of 148 files report a 1–2 block GPS-vs-position count mismatch
  (the file-boundary split-block case; stitched + warned, not fatal).
- Stage 3 hit quality is dominated by `unresolved` (~64–78% per telescope
  plane) — genuine fiber×ribbon multiplicity, NOT single-row crosstalk:
  sweeping `tot_thresh` 1→4 barely moves it and never unblocks Stage 5.
  The real Stage 5 blocker is plane geometry, not hit quality — see
  [[testlab-20210723-plane-z-order]].

Real geometry (confirmed): plane gap = 670 mm, columns NOT in file order —
correct mapping is `--z-tel 0 -1340 -670` (col 0 = z 0, col 2 = middle at
z −670, col 1 = far at z −1340). See [[testlab-20210723-plane-z-order]].

Correct invocation + result (post-fix `k==mid` stage 4):
`python scripts/run_pipeline.py --telescope data/0_testLab_20210723/Base \
  --probe data/0_testLab_20210723/Probe_0 --z-tel 0 -1340 -670`
→ probe at **t_x≈+177 mm, t_y≈+237 mm, θ≈−0.9°, z_p≈+844 mm, n_inliers≈150**.
Telescope Z is near-perfect: middle-plane (col 2) delta_z≈−0.10 mm,
tilts ~1–10 mrad.

Coincidence-loss funnel (Stage 2 → Stage 5, instrumented via
`scripts/track_coincidence_loss.py`): 106,355 coincidences → 150 inliers
(0.141% survival). Almost all loss is ONE gate: **telescope hit quality
kills 105,139 (98.86%)** — a coincidence survives only if all 3 tel planes
are golden/cluster (per-plane golden+cluster ≈25/15/19%, joint ≈1%). The
rest is minor: track χ² −717, probe quality −301, RANSAC −20. So yield is
throughput-limited by the telescope's per-plane hit-resolution (the
`unresolved` fiber×ribbon multiplicity), NOT by cuts/geometry/`tot_thresh`.
Raising inliers needs better gate2 hit reconstruction.

UPDATE — gate2 recovery wired into Stage 5: `disambiguate_telescope_hits`
(stage3, two-plane line projection §6.3b) existed and was used in Stage 4
but was MISSING from `stage5.PoseFitter._decode_cluster`. Adding it there
(promote a single `unresolved` plane from the other two) recovered 889
coincidences at gate2 → clean coincidences 170→319, **Stage 5 inliers
150→272 (+81%)**, pose unchanged within errors, σ_zp tighter (3.6→2.4 mm).
gate2 still dominates (98.02%): most lost events have ≥2 unresolved planes
or no candidate within the 1.5-strip window. Measured with
`scripts/track_coincidence_loss.py` (prints the gate-by-gate funnel).

UPDATE 2 — corrected-frame prediction (realized the refinement above):
`disambiguate_telescope_hits` gained an optional `offsets` arg (per-plane
delta_x/delta_y); Stage 5 now predicts AND selects candidates in the
alignment-corrected frame (Stage 4 still defaults to raw → unchanged).
Strictly better than raw-frame: it recovers FEWER candidates (603 vs 889)
because the sharper prediction rejects the ~286 wrong matches the raw
±5 mm-misaligned window let in (those were failing the track χ² cut —
gate3 dropped 1186→838). Net **Stage 5 inliers 150→272 (raw) →297
(corrected, +98% over baseline)**, clean coincidences 319→343, z_p
843.9→836.3 mm (σ 3.6→2.3). Lesson: in Stage 5 always predict in the
corrected frame; the raw frame trades real inliers for χ²-gate pollution.

UPDATE 3 — single-axis recovery (biggest win, commit a871fd5): a plane is
`unresolved` if EITHER x or y fails; **51.1% of unresolved telescope
readings are single-axis** (one axis cleanly resolved, measured by
`scripts/investigate_single_axis.py`). decode_position used to discard both
coords. Now it keeps the resolved axis as a one-element candidate so
`disambiguate_telescope_hits` fills only the failed axis from the projection.
**Stage 5 inliers 297→933 (3.1×)**, recovered-at-gate2 603→2996, clean
coincidences 343→1086, σ_zp 2.3→1.3 mm, pose still consistent (t_x +179.5,
t_y +234.5, θ −0.6°, z_p +839.0). Topology: of 106,355 coincidences only
1.1% have all 3 planes natively resolved, 10.5% have 2 (the recoverable
"one-bad-plane" set), 34.6% have 1, 53.7% have 0 — so recovery needs ≥2 good
planes and the ceiling is the 2-resolved bucket. KNOWN GAP:
`run_pipeline.py`'s Stage-3 hit-quality display still decodes WITHOUT
disambiguation, so those counts understate the real recovered yield.

UPDATE 4 — full combinatorial track finder (branch
`feat/combinatorial-track-finder`, post `9c08787`/`2660fb9`/`ebd0178`/
`df80e1d`, run 2026-06-22): re-ran the canonical invocation (`--z-tel 0
-1340 -670`, default `tot_thresh=1`, default `tot_weights=True`) via
`uv run python scripts/run_pipeline.py ... --plot`. **n_inliers = 8604**
(vs. 933 in UPDATE 3 — the combinatorial candidate-triple χ² search
(`7aff0e5`–`b37ac08`), mirror-fold candidate splitting (`c98d8a2`), and
default `tot_weights` together account for most of the gap; UPDATE 3 predates
all of them). Pose unchanged within error: t_x +180.3±0.1, t_y +233.2±0.1,
θ −0.6°, z_p +840.5±0.4 mm. Gate funnel: 106355 coincidences →
zero_candidate_plane −21608 → no_anchor_plane −39658 → chi2_track_cut
−13354 → probe_quality −20634 → 11073 accepted. Run via plain `python`
fails on `--plot` (`ModuleNotFoundError: plotly`, a `dev` dependency-group
package) — must use `uv run python scripts/run_pipeline.py` to get plotly.
The 20/148-file ±1-block GPS/position mismatch warning is the same
known-benign quirk noted above, unaffected by this rewrite. Output:
`pipeline_out/testLab_20210723_run/{summary.txt,pose_3d.html}`.

Robustness notes (verified across z=400/640/670 mm even spacings):
- t_x, t_y, θ, n_inliers are spacing-INVARIANT (identical to the digit) —
  the in-plane probe pose does not depend on the assumed plane gap.
- z_p is exactly linear in the gap: z_p ≈ 1.260 × gap_mm (probe sits 1.26
  plane-gaps beyond the col-0 reference). So z_p follows directly from the
  true gap; no re-run needed if the gap changes.
