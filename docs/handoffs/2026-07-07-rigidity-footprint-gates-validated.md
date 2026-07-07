# Off-probe track gates: implemented, two real bugs found and fixed, validated over 17:00–18:30 UTC

Written 2026-07-07. Branch `feat/probe-monitoring`. Implements
`docs/handoffs/2026-07-07-off-probe-track-gate-strategy.md`'s two gates and
runs its validation plan against real data. **Closes the testLab 20210723
z_p burst thread.**

## What shipped

- `filter_rigidity(coincs, z_ref, max_resid_mm)` and
  `filter_off_probe(coincs, ref_pose, probe_size_mm, max_off_probe_mm)` in
  `src/monrad/pose/optimize.py`, exported from `monrad.pose`.
- Wired into `monitor_probe`/`monrad-monitor`
  (`src/monrad/monitor/timeseries.py`): rigidity first, footprint second,
  both opt-in via `--max-rigidity-resid-mm` / `--max-off-probe-mm` (default
  `None` = off). `prev_pose` tracks the last *accepted* window's pose as the
  `z_ref` / footprint anchor; footprint is skipped on window 0.
- `tests/test_pose_gates.py` (new, 7 tests) — hand-built clean clusters +
  injected cross-particle tracks, per gate.

## Two bugs found only by running real data (neither showed up in synthetic tests)

**1. The "never drop below 3 survivors" floor-guard was backwards.** Both
gate functions originally bypassed to a no-op (kept everything) whenever
applying the cut would leave fewer than 3 coincidences. On the real testLab
17:14:31–17:19:31 window — which the prior root-cause analysis established
is **100% contaminated** (0 genuine coincidences, every track wide-angle) —
every single coincidence scores above any sane threshold, so `mask.sum()`
hits 0 and the old guard made the gate a **no-op on exactly the case it
exists to catch**. Fixed: the gates now honestly drop down to 0 survivors;
`monitor_probe`'s existing `len(working) < min_fit` check (min_fit defaults
to 30, always ≥ 3) is what protects `fit_probe_pose` from ever seeing < 3
coincidences — the gates don't need their own floor beyond the `n < 3` input
guard (can't vote on rigidity with fewer than 3 to compare against). See
`filter_rigidity`/`filter_off_probe` docstrings and
`tests/test_pose_gates.py::test_drops_everything_in_fully_contaminated_window`
/ `test_drops_everything_when_no_track_lands_on_probe`.

**2. The strategy doc's cold-start `z_ref` fallback (`mean(z_corr)`) is
catastrophically wrong for this detector's real geometry.** The doc claimed
"the gate tolerates being off by tens of mm" and suggested `mean(tel_z)` for
window 0. Measured on real data: `mean(z_corr) ≈ -670 mm` vs true
`z_p ≈ +840 mm` — **over 1500 mm off**, not tens. Since the rigidity residual
scales with `|z_ref - z_p|`, this flagged every genuine coincidence in
window 0 too, dropping it entirely — and because `prev_pose` only updates on
an *accepted* window, it never left `None`, so **every subsequent window for
the rest of the run inherited the same broken z_ref** (confirmed: a run with
this fallback dropped essentially 60–100% of every single window all day,
not just the two known bursts). Fixed in `monitor_probe._emit`: cold start
now runs one ungated `fit_probe_pose` on the window's own coincidences and
uses *that* z_p as `z_ref` — cheap (one extra fit, once per run) and far
better anchored (826.6 mm on the real first window, vs -670 mm).

**Lesson for future gate/threshold work in this codebase:** synthetic tests
validate the math; only a real-data run surfaces cascading failures from a
bad default/fallback. Always run the `/verify`-style real-data pass before
declaring a gate done, even when unit tests are green.

## Validation results (full run, `data/0_testLab_20210723`, `--z-tel 0 -1340 -670 --n-probe-ch 40 --min-anchor-planes 1 --window-s 300`)

Full-day run required (not just 17:00–18:30) because stage-1 time
reconstruction anchors to the *first PPS record of the files it's given* —
feeding it a file subset starting mid-day silently reinterprets that
subset's first tick as `t=utc0`, producing wrong absolute times. Windows
outside 17:00–18:30 in the printed table below are for context only.

**Baseline (no gates), 143 windows fitted:**
- `17:14:31–17:19:31`: t_x=705.1, t_y=359.3, **z_p=-360.6 mm**, rms=287.0
- `18:11:06–18:16:10`: t_x=330.4, t_y=311.1, **z_p=1846.2 mm**, rms=833.5
- Whole-run resid RMS: min=19.8 median=84.8 max=833.5 mm

**`--max-rigidity-resid-mm 100` alone, 140 windows fitted:**
- Both burst windows **vanish from the output** — every coincidence in each
  is flagged (66/66 and 73/73 dropped), leaving 0 survivors, so
  `min_fit` skips them. This is the *correct* outcome, not a partial
  recovery: these two windows have zero genuine coincidences to fit (matches
  `memory/testlab-20210723-anomaly-root-cause.md`'s "0 good / 5–9 bad" per
  30 s sub-bin) — there is nothing to recover a pose *from*. The strategy
  doc's "z_p → ~840 mm" validation wording assumed a partial-recovery outcome
  that isn't physically available for a 100%-contaminated window; dropping
  the window entirely is the safe/correct behavior.
- Every other window in 17:00–18:30 stays at sane z_p (827–855 mm),
  `n_inliers` down by only a handful per window (the "ever-present wild-track
  baseline" the code already documents), and RMS drops sharply everywhere —
  not just a no-op, a net improvement: whole-run resid RMS min=15.9
  median=35.1 max=129.7 mm (was 19.8 / 84.8 / 833.5).
- Cold-start check passed: the true window 0 (`09:36:21`) now succeeds
  (z_ref bootstrapped to 826.6 mm), and only 3 windows total are lost across
  the whole day (143→140) — the two known bursts plus one previously
  unidentified window elsewhere in the day (not investigated further; a
  candidate third burst worth a look if this thread reopens).
- Two windows just after each burst (`17:30:00–17:35:02` n=5 rms=118.9,
  `18:47:17–18:52:32` n=8 rms=129.7) remain marginal but not catastrophic —
  consistent with "step structure" transition contamination bleeding across
  the fixed 300 s monitor-window boundaries (which don't align to the
  underlying 5-min *file* boundaries where the step was originally
  characterized).

**`--max-rigidity-resid-mm 100 --max-off-probe-mm 100` together, 140
windows:** identical printed table to rigidity-alone — the footprint gate
fired exactly once across the entire ~12 h run (1/88 in one window,
09:36–…) with no visible effect on that window's `n_inliers`/z_p. Confirms
the strategy doc's characterization: footprint is "rare, cheap insurance,"
not the workhorse; rigidity alone does essentially all the work on this
dataset. No regression from adding it.

## Recommended thresholds for this setup

`--max-rigidity-resid-mm 100` cleanly separates clean-window scores
(median-of-median pairwise residual ~8–30 mm, occasional single-outlier max
up to a few hundred mm) from the two known-bad windows (min per-coincidence
score 147.6 mm / 232.7 mm — i.e. even their *best* coincidence is well above
100). As with `--max-resid-rms`, this is setup-specific — tune from a
whole-run score distribution, not a universal constant. `--max-off-probe-mm`
can reuse the same order of magnitude; its effect was negligible here but it
is cheap insurance against a rigidly-but-wrongly-drifted window rigidity
can't see (see strategy doc's rationale — not exercised by this dataset).

## Not done / open follow-ups

- No CLI diagnostic prints the rigidity pairwise-score distribution (the
  strategy doc's validation step 1 asked for one "like the existing RMS
  distribution"). Threshold-picking here was done with an ad hoc scratch
  script, not shipped code — if this gate becomes a standing part of the
  workflow, consider adding an opt-in `--print-rigidity-dist`-style summary
  analogous to the existing end-of-run RMS print.
- The third dropped window (143→140, only 2 of which are the known bursts)
  was not identified or investigated.
- `filter_rigidity`/`filter_off_probe` are not wired into `PoseFitter`'s
  streaming pipeline path (`scripts/run_pipeline.py`) — per the strategy
  doc, the monitor was the deliverable; `PoseFitter`'s default behavior is
  intentionally unchanged.
