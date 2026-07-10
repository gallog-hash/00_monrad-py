# Post-fit continuity gate: root-caused the burst-tail degradation, implemented and drop-mode validated, threshold not yet finalized

Written 2026-07-09. Branch `feat/probe-monitoring`. Follow-up to
`docs/handoffs/2026-07-09-post-fix-monitor-run-transition-window.md` (same
day, earlier session) — that doc left off with the user asking whether to
try `--max-off-probe-mm` or a tighter `--max-rigidity-resid-mm` on the
`17:29:21–17:36:3x` burst-tail window. This session took a different path
instead: investigated *why* neither existing gate could fix that window and
implemented a new one.

## What this session found (investigation, no code yet)

Isolated the `17:29:21–17:36:3x` window (using ad hoc scripts, not saved —
each was a short throwaway analysis re-running `stream_coincidences` /
`fit_probe_pose` directly, cheap to reproduce) and found:

- Of the 100 coincidences that pass `--max-rigidity-resid-mm 100`, **83 are
  rejected only by `fit_probe_pose`'s own Mahalanobis cut** — the rigidity
  gate is barely discriminating in this window.
- The 83-strong "outlier" group is not noise: fit alone, it is **internally
  self-consistent** (0 self-rejects) and converges to its own coherent pose.
  This window contains **two mutually-consistent geometric clusters**, not
  signal + scatter.
- Neither existing gate can separate them: rigidity scores overlap almost
  completely between the two clusters (outlier-group minimum score, 13.8mm,
  is even *below* the inlier-group minimum, 14.2mm). The footprint gate
  (checked against the *previous* accepted pose) is completely blind here —
  100/100 survivors land inside the probe footprint even at a 10mm
  tolerance, because near this shower/pile-up burst essentially every
  candidate track intersects the probe footprint when projected back to the
  last known-good z.
- Neither cluster is the truth: windows immediately before (842.5mm, θ=−1.01°)
  and after (832.8–851.6mm, θ≈−0.6° to −1.1°) agree tightly on a stable
  trajectory around **t_x≈179, t_y≈234, θ≈−1°, z_p≈840mm** — both the
  reported 17-inlier cluster (θ=9.5°, z_p=965.9) and the alternate 83-cluster
  (θ=11.0°, z_p=1037.9) are ~120mm/~10° off from it.

**User-proposed alternative angle, confirmed empirically:** growing the raw
batch (bigger `--min-fit` / `--window-s`) dilutes the contaminated cluster
until `fit_probe_pose`'s own robust fit naturally treats it as the minority.
Replaying the same window with a much larger raw batch showed a sharp
transition around raw≈200–300 (window span ~7min → ~12.5min), converging
cleanly to the known-stable trajectory and staying stable out to raw=2000.
Full `--min-fit 250` CLI run over the whole 12h `data/0_testLab_20210723`
dataset confirmed this generalizes: max resid RMS drops from 133.1mm
(`--min-fit 100` baseline) to 63.2mm, at the cost of window count
159→63 (coarser time resolution everywhere, not just at bursts).

## What was implemented

**1. Committed (`f7c82b4`):** a missing log line. `monrad-monitor` had no
visibility into `fit_probe_pose`'s own Mahalanobis inlier/outlier split —
`n_inliers` appeared in the CSV with no denominator. `_record` (then
`_fit_and_record`) in `src/monrad/monitor/timeseries.py` now logs
`"Window %s–%s: fit accepted %d/%d gate-survivor(s), rejected %d via
Mahalanobis cut."` whenever `pose.outliers` is non-empty.

**2. Not yet committed:** a new post-fit continuity gate,
`max_pose_jump_mm`/`max_pose_jump_deg` (CLI: `--max-pose-jump-mm`,
`--max-pose-jump-deg`), in `src/monrad/monitor/timeseries.py`. Unlike the two
pre-fit gates (`filter_rigidity`, `filter_off_probe` in
`src/monrad/pose/optimize.py`, which filter individual coincidences *before*
the fit), this gate runs **after** `fit_probe_pose`, compares the candidate
pose against the previous accepted window's pose via a new helper
`_pose_jump()`, and rejects it if it moved more than `max_pose_jump_mm`
(Euclidean distance of the raw fit corner `(t_x, t_y, z_p)` — deliberately
*not* the probe centre, since a rotation can cancel a corner translation
there; see `_pose_jump`'s docstring for the exact testLab numbers that bit
us) or rotated more than `max_pose_jump_deg`.

**Current behavior (per explicit user request this session, revising an
earlier grow-and-retry design):** rejection is terminal. The whole raw batch
is discarded immediately (`logger.warning("Dropping window %s–%s:
post-fit continuity gate rejected the fitted pose ...")`, same style/level as
the other two drop paths) and the next window starts fresh from the
following coincidence. An earlier version of this gate instead kept growing
the same batch and retrying (like the pre-fit gates do) — that version was
implemented, tested, and then explicitly replaced with the current
discard-on-failure behavior; there is no growth/retry logic left for this
gate. `CONTINUITY_REFIT_STRIDE` (the throttling constant from the
grow-and-retry version) was removed along with it.

Both changes: `ruff check`/`ruff format` clean, full `uv run pytest` suite
(226 tests) passes.

## Validation runs (all in untracked `pipeline_out/`, not committed)

Same base command each time (`data/0_testLab_20210723`, `--z-tel 0 -1340
-670 --n-probe-ch 40 --tot-thresh 2 --min-anchor-planes 0
--max-rigidity-resid-mm 100`):

| Run (dir under `pipeline_out/`) | Extra flags | Windows | RMS min/median/max (mm) |
|---|---|---|---|
| `monitor_window_run_2026-07-09` | `--min-fit 100` (baseline) | 159 | 24.9/41.5/**133.1** |
| `monitor_window_run_minfit250` | `--min-fit 250` | 63 | 30.2/41.7/**63.2** |
| `monitor_window_run_minfit100_continuity` | `--min-fit 100 --max-pose-jump-mm 80 --max-pose-jump-deg 3` (grow-and-retry version) | 159 | 25.9/41.5/**103.9** |
| `monitor_window_run_minfit100_continuity_tight` | `--min-fit 100 --max-pose-jump-mm 40 --max-pose-jump-deg 1.5` (grow-and-retry version) | 154 | 25.4/41.0/**79.9** |
| `monitor_window_run_minfit100_drop_tight` | `--min-fit 100 --max-pose-jump-mm 40 --max-pose-jump-deg 1.5` (**current discard-on-failure code**) | 149 | 22.7/41.4/**79.3** |

The `drop_tight` run is the one that reflects the code currently in the
working tree. Confirmed directly in its log: the known burst window
(`17:29:21–17:36:38`) is dropped with `Δpos=139.6mm, Δθ=10.55°`, and the
*next* window (`17:36:41–17:41:55`) is fully clean (n_inliers=80, rms=36.2,
matching the established baseline trajectory) — no contamination carried
forward.

## Open, unresolved — threshold choice

The 40mm/1.5° thresholds recover the known burst well but also drop **8
other windows** elsewhere in the 12h run that look like ordinary θ jitter
(Δθ 1.6–2.0°, Δpos 10–17mm) rather than contamination — with discard-on-
failure these are now simply **lost** (no data point for that stretch),
whereas the grow-and-retry version used to recover them fine on regrowth.
The looser 80mm/3° thresholds don't have this false-positive cost but also
don't fully recover the burst (converges to an intermediate, still-off
pose). No threshold in between (e.g. ~2–2.5°) has been tried yet.

**One new, unexplained finding, not investigated further:** the
`drop_tight` run's log also shows a drop at `21:48:21–21:54:55` with
`Δpos=74.3mm` but `Δθ=only 0.89°` — position-dominant, unlike either known
burst (both rotation-dominant, Δθ ~10° with more modest Δpos). Worth a look
before assuming the gate's false-positive/true-positive characterization
above is complete.

## Not done / next steps

1. Decide on final `--max-pose-jump-mm`/`--max-pose-jump-deg` values —
   possibly loosen θ to ~2–2.5° and re-run the whole-run comparison to see
   if that avoids the 8 false-positive drops while still fully recovering
   the known burst.
2. Investigate the `21:48:21` window — is it a third genuine anomaly, or
   an artifact of the position-vs-rotation split in `_pose_jump`?
3. Once thresholds are settled, commit the continuity-gate code (currently
   uncommitted in `src/monrad/monitor/timeseries.py`).
4. Update `docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md`'s
   "recommended thresholds" section and the
   `rigidity-footprint-gates-shipped` memory once a final config ships.
5. Consider whether the two-cluster-degeneracy finding (see "What this
   session found" above) is worth its own memory entry — it refines
   `testlab-20210723-anomaly-root-cause` (still correct: geometric
   consistency gate, not timing) with a more precise mechanism (two
   internally-consistent clusters defeating both existing gates
   simultaneously) that the existing memory doesn't capture.

## Suggested skills for next session

- **`/verify`** — after any threshold change, drive `monrad-monitor`
  end-to-end on `data/0_testLab_20210723` and check both the known burst and
  the 8 currently-dropped "false positive" windows before declaring a
  threshold final.
- **`code-review`** — before committing, a careful pass on the new
  continuity-gate block in `monitor_probe`'s main loop (and the `_pose_jump`
  helper) is warranted: this file's gate-growth logic has already had two
  real-data-only bugs found post-hoc in earlier sessions (see
  `rigidity-footprint-gates-shipped` memory), and the discard-on-failure
  rewrite touched the same loop.

## Key files

- `src/monrad/monitor/timeseries.py` — `monitor_probe` (main loop is where
  the continuity gate lives), `_pose_jump`, `_record`, `_run_gates`,
  `RAW_CAP_MULTIPLIER`, `COLD_START_REFIT_STRIDE`. Currently has one
  uncommitted diff (the continuity gate).
- `src/monrad/pose/optimize.py` — `fit_probe_pose`, `filter_rigidity`,
  `filter_off_probe` (unchanged this session; referenced for contrast with
  the new post-fit gate).
- `pipeline_out/monitor_window_run_minfit100_drop_tight/` +
  matching `.log` — the run reflecting the current code; start here to
  inspect the 8 false-positive drops and the 21:48 window.
- `pipeline_out/monitor_window_run_minfit250/` — the dilution-only
  alternative (no continuity gate), kept for comparison.
