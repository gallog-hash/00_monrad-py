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

**`tests/test_scan_geometric_cuts.py`** — 36 tests. The linchpin is
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

Fix, in `reanchor_window`:

1. *Absolute* — advance each detector by its own file-name elapsed time. Good
   to a few seconds, which is ample for 5-minute bins and multi-hour alignment
   windows.
2. *Relative* — the part that must be exact, and is **measured, not assumed**.
   Both detectors' PPS are the same physical GPS pulses on whole UTC seconds,
   so the residual is an integer number of seconds; `calibrate_shift` finds it
   by maximising raw coincidence yield over whole-second probe shifts
   (`count_matches`, two `searchsorted` calls per candidate, one extra stage-1
   pass total).

Measured on real data — the peaks are unambiguous, and **the shift differs per
window**, so a single global constant would not have worked:

| window | files | file-name estimate | measured | coincidences at best | next best |
|---|---|---|---|---|---|
| `20210724_0000`–`0030` | 6 | +2 s | **+3 s** | 4120 | 2 |
| `20210723_1900`–`2035` | 19 | — | **+1 s** | 13603 | 5 |

Note the CLEAN window's file-name estimate was **off by one second** — trusting
file names alone (`--trust-file-names`) would have produced zero coincidences.
`ShiftCalibration.is_confident` requires the winner to stand 5× above every
other shift, and the CLI aborts rather than scan cuts against accidentals.

Synthetic data cannot exercise the success path here: `synthetic.generate`
emits tracks on an exact 0.1 s grid, so whole-second shifts alias perfectly and
every shift ties. The calibration correctly reports that as inconclusive, and
`TestReanchoring::test_periodic_data_is_reported_as_inconclusive` pins it.

## Real-data sanity checks passed

Reproduces the known funnel shape on the 6-pair CLEAN sample
(`--tot-weights`, in-run alignment), with `no_anchor_plane` dominating exactly
as `2026-07-20-chi2-track-max-cluster-width-flags-shipped.md` reported:

| χ² | ambiguous | zero-cand | no-anchor | χ²-cut | probe-qual | accepted |
|---|---|---|---|---|---|---|
| 4.0 | 1 | 880 | 1530 | 642 | 682 | 385 (9.34 %) |
| 37.0 | 1 | 880 | 1530 | 158 | 1011 | 540 (13.11 %) |

Timing: Tier-A decode runs ~280 clusters/s (19 file pairs → 13.6k clusters in
15 s), so the 72-pair CLEAN window is ~1 min, not the ~30 min the plan budgeted
— the `--decode-anchor 0` variant will be far slower and is still uncalibrated.
Tier-B replay costs ~0.7 s per grid point at 419 coincidences and scales with
the coincidence count; the default grid is ~170 points per cache.

## Test status

`ruff check` / `ruff format --check` clean. **354 tests pass**, covering every
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
   `--end 20210723_204000` to include the 20:35 batch (19 pairs came back, not
   20).
4. Absolute UTC after re-anchoring is good to a few seconds only. The DAQ
   file-name clock is **not** exactly UTC+2 on this dataset — the acquisition's
   first file is named `11:40:32` while its `utc0` is `09:36:19`, i.e. an offset
   of 2 h 04 m 13 s. Any CLEAN/ANOMALY window boundary quoted in UTC from a
   file name is therefore ~4 minutes out.

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
