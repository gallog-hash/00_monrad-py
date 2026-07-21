# Geometric-cut scan harness — built and validated, real-data runs not yet done

Implements the harness half of the plan "Real-data scan of `_CHI2_TRACK` and
the geometric event cuts (testLab_20210723)". The harness is complete, tested
and exercised against real data; the actual Tier-A/Tier-B production runs, the
figures and `reports/cut_scan_testLab_20210723/report.md` are **not** done and
are the next session's job.

Branch: `study/geometric-cut-scan`.

## What shipped

**`src/monrad/pose/types.py` + `fitter.py`** — additive only:
`TelescopeTrackResult.best_cands` now carries the winning candidate triple
(defaulted `None`, so every existing caller and positional construction is
untouched). Needed so offline diagnostics can recompute per-plane mm residuals
without re-running the combinatorial search.

**`scripts/scan_geometric_cuts.py`** — the two-tier scan.

- *Tier A* (`decode_pass`) streams stages 1→3 once per
  `(tot_thresh, tot_weights, max_cluster_width)` at `chi2_track=inf`, caching
  one record per stage-2 cluster to `.npz`: funnel reason, `cand_counts`, best
  χ², probe/telescope quality, the winning triple's positions/σ, its per-plane
  mm residuals, and the full `Coincidence` payload.
- *Tier B* (`replay`, `evaluate`, `run_grid`) reproduces any **tighter**
  `(chi2_track, min_anchor_planes)` and any post-fit gate combination offline,
  including a MATLAB-`ALIGNDIST`-style absolute-mm cut in place of the
  σ-adaptive χ². Refuses to replay looser than the cache was decoded at.
- Figures of merit per grid point: funnel, `n_inliers`, σ per parameter,
  **σ_zp·√N**, `resid_rms` over all fed coincidences, Mahalanobis inlier
  fraction, an accidental-pedestal **on-probe purity/signal-count** estimate,
  and the even/odd half-split spread.

**`scripts/scan_plots.py`** — the 11 figures, including `sigma_vs_n.png` (the
decision plot: real gain tracks 1/√N, junk rises above it).

**`tests/test_scan_geometric_cuts.py`** — 40 tests. The linchpin is
`TestReplayEquivalence`, which asserts the offline replay reproduces a **live**
`PoseFitter` run exactly (funnel counts, accepted `Coincidence` list, and
`n_inliers`) across 7 `(chi2_track, min_anchor_planes)` points.

## The blocker this uncovered: mid-stream slices are silently mistimed

The plan's step 5 asked to verify mid-acquisition slicing rather than assume
it. It does **not** work, and the failure is silent.

`reconstruct_stream` anchors a stream's *first* PPS to the header's `utc0` and
then counts PPS edges. Hand it a slice that starts mid-acquisition and every
event comes back timed as though the slice were the start of the run. The shift
is per detector and the two detectors do not share it — on `testLab_20210723`
the telescope's `20210724_0000*` files begin 12:20:02 after its own first file
and the probe's begin 12:20:00 after its own. That is a 2 s skew against a
200 ns coincidence window: **zero coincidences, no error raised, a
perfectly healthy-looking decode of 0 clusters.**

### The fix: never slice the file list

`decode_pass` requires the **whole** acquisition and takes a `file_range`
instead. It streams from file 0 and gates each cluster on the telescope's
`PosRef.file_idx`, so times are correct *by construction* — no estimation, no
calibration, no heuristic — and stops as soon as it leaves the range. Only
stage 1 and the coincidence merge are paid for the skipped prefix; no position
decoding happens outside the window.

This is cheap: **3.2 s** of stage 1 for both detectors across the CLEAN
window's entire 154-file prefix. A day-3 window (~600 files) would be ~15 s.

`slice_detector` survives only for the two places absolute time is irrelevant:
the no-`--alignment` fallback fit (position decoding only) and the
window-pairing sanity check.

### The check that it worked

`decode_pass` then tallies, free from the events it already streamed, how many
raw coincidences survive each whole-second probe shift (`window_shift_scan`).
Whole seconds only: both detectors' PPS are the same physical GPS pulses on
whole UTC seconds, so any misanchoring is an exact integer number of them. A
correctly-timed pair peaks sharply at zero, and the CLI aborts if it does not:

| window | files | shift 0 | next best |
|---|---|---|---|
| `20210724_0000`–`0030` | [148, 154) | 110 484 | 25 |
| `20210723_1900`–`2040` | [88, 108) | 77 846 | 19 |

(The tally covers the streamed range, prefix included, so it exceeds the cached
cluster count — deliberately, since it checks the clocks agree run-wide.)

An earlier revision instead re-anchored the sliced lists and *measured* the
residual offset by maximising coincidence yield. That worked — it found +3 s on
CLEAN where the file-name estimate said +2 s, i.e. **trusting file names alone
would have produced zero coincidences** — but streaming from file 0 is exact,
simpler and cheaper, so it replaced it. The two agree: identical `accepted`
(385) on the CLEAN 6-pair window, differing only by ±2–4 clusters at the window
edges.

Synthetic data cannot exercise the check's success path: `synthetic.generate`
emits tracks on an exact 0.1 s grid, so whole-second shifts alias perfectly and
every shift ties. The check correctly reports that as inconclusive
(`--no-window-check` is the documented escape hatch, used by the CLI tests) and
`TestWindowCheck::test_periodic_data_is_reported_as_inconclusive` pins it.

## Real-data sanity checks passed

Reproduces the known funnel shape on the 6-pair CLEAN sample
(`--tot-weights`, in-run alignment), with `no_anchor_plane` dominating exactly
as `2026-07-20-chi2-track-max-cluster-width-flags-shipped.md` reported:

| χ² | ambiguous | zero-cand | no-anchor | χ²-cut | probe-qual | accepted |
|---|---|---|---|---|---|---|
| 4.0 | 1 | 880 | 1530 | 642 | 682 | 385 (9.34 %) |
| 37.0 | 1 | 880 | 1530 | 158 | 1011 | 540 (13.11 %) |

Timing: Tier-A decode runs ~800 clusters/s (20 file pairs → 14.3k clusters in
17.8 s, including ~4 s of stage 1 over the 88-file prefix), so the 72-pair CLEAN
window is ~1–2 min, not the ~30 min the plan budgeted. The `--decode-anchor 0`
variant will be far slower and is still uncalibrated. Tier-B replay costs ~0.7 s
per grid point at 419 coincidences and scales with the coincidence count; the
default grid is ~170 points per cache, so budget accordingly at CLEAN's much
larger coincidence count.

## Test status

`ruff check` / `ruff format --check` clean. **358 tests pass**, covering every
file in the suite.

**10 tests were not verified**: the `test_real_*` tests in
`tests/test_monitor_timeseries.py`, which stream the whole
`data/0_testLab_20210723/` acquisition (`-k "not real"` gives 39 passed, 10
deselected). Each attempt to run them was killed part-way by this environment's
reaping of long-running processes. They are untouched by this branch and their
runtime is pre-existing — but they have not been seen green here, so re-run
them before merging.

Worth noting: `CLAUDE.md` states "All tests use synthetic data from
`monrad.synthetic.generate()`; no real detector files are required." That is not
true of this file, and those real-data tests are what make the suite ~20 min
rather than ~2.

## Not done / next session

1. Steps 6, 8, 9, 10: fix the alignment once with `monrad-align`, run Tier A on
   CLEAN + ANOMALY, the B1→B2→B3 grid, the figures, and `report.md`.
2. **Verification still outstanding**: the plan's cross-check that the harness
   reproduces raw **2201** / accepted **197** on the 3-file-pair MATLAB subset.
   Not yet run.
3. `--start/--end` are half-open, so the plan's `…2035*` window needs
   `--end 20210723_204000` to include the 20:35 batch; that gives the intended
   20 pairs (files [88, 108), 14 307 clusters).
4. Absolute UTC is good to a few minutes only, which matters for the ANOMALY
   veto: the 17:15 and 18:10 UTC bad bins must be located empirically from the
   reconstructed stream, **not** by subtracting 2 h from file names — the error
   is enough to land on the adjacent 5-minute bin. Two independent effects: the
   header's UBX-TM2 frame (which seeds `utc0`) was captured **5m03s before the
   first data file opened**, identically on both detectors, so every
   reconstructed time is ~5 min early; and the DAQ
   file-name clock is **not** exactly UTC+2 — the first file is named
   `11:40:32` while its `utc0` is `09:36:19`, an offset of 2 h 04 m 13 s.
   (`monrad.monitor.align._daq_utc_offset` already compensates for this when
   labelling alignment windows.)

## Checked and dismissed: GPS/position block-count warnings

Decoding the ANOMALY window logs GPS-vs-position block-count mismatches on a
few files, two of which sit in the 18:10 UTC bad bin. That looked briefly like
the first raw-file signature of the anomaly. It is not — a 19-file comparison
shows they are a normal, low-rate artifact:

| window | files | mismatch warnings |
|---|---|---|
| `20210723_1900`–`2040` (ANOMALY) | 19 | 4 |
| `20210724_0000`–`0140` (CLEAN)   | 19 | 2 |
| `20210723_1500`–`1640` (neither) | 19 | 0 |

They also always come in adjacent pairs, one file one block short and the next
one block long (`…191534` −1 / `…192034` +1; `…201033` −2 / `…201534` +2;
`…011033` −1 / `…011534` +1) — the signature of a 16-row block straddling a
file rotation, which the pipeline already stitches. Benign; no follow-up
needed. This does **not** overturn
`testlab-20210723-anomaly-no-raw-telescope-signature`.
