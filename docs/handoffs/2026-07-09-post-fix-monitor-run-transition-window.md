# Post-fix `monrad-monitor` reproduction: transition-window degradation explained, not yet acted on

Written 2026-07-09. Branch `feat/probe-monitoring`. Follow-up to
`docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md` and
`docs/handoffs/2026-07-07-off-probe-track-gate-strategy.md`.

## What this session did

Re-ran `monrad-monitor` against `data/0_testLab_20210723` on current `HEAD`
(includes `dab37df`..`14f4ebe`, seven commits merged after the archived
`pipeline_out/monitor_window_run_rigidity_fixed` baseline) and diffed the
two runs window-by-window. No code was changed this session — this was
pure investigation.

**Command run** (output in `pipeline_out/monitor_window_run_2026-07-09/`,
full log in `pipeline_out/monitor_window_run_2026-07-09.log`):

```
uv run monrad-monitor \
  --telescope data/0_testLab_20210723/Base \
  --probe data/0_testLab_20210723/Probe_0 \
  --z-tel 0 -1340 -670 \
  --n-probe-ch 40 \
  --tot-thresh 2 \
  --min-anchor-planes 0 \
  --max-rigidity-resid-mm 100 \
  --min-fit 100 \
  --out pipeline_out/monitor_window_run_2026-07-09
```

`--z-tel`/`--n-probe-ch` taken from [[testlab-20210723-plane-z-order]] /
[[testlab-20210723-probe-size]] memory, not from user input — the user's
original ask omitted them but they're load-bearing for this dataset.

## Result summary

| | old (`monitor_window_run_rigidity_fixed`) | new (`monitor_window_run_2026-07-09`) |
|---|---|---|
| windows | 162 | 159 |
| resid RMS | min=21.2 median=42.1 max=128.3 | min=24.9 median=41.5 max=133.1 |

Sequence-aligned 159/162 windows 1:1 (3 old windows absorbed into
neighbors post-fix, 0 new-only windows). Central values flat within fit
noise (Δt_x mean +0.06 mm, Δt_y mean −0.06 mm, Δz_p mean +1.2 mm, all
stdev ≈ 3–10 mm). The comparison script used inline Python (csv +
`difflib.SequenceMatcher` aligning on rounded `utc_start`) — not saved as
a file; rerun ad hoc if needed, it's ~40 lines.

**The one real finding:** window `17:29:21–17:36:3x` (tail of the known
17:15 burst, see [[testlab-20210723-anomaly-root-cause]]) is the worst
window in both runs and got measurably worse post-fix:
- old: n_inliers=22, z_p=902.4, rms=128.3
- new: n_inliers=17, z_p=965.9, rms=133.1

## Root cause (verified against the new run's own logs)

This window sits in a **partially**-contaminated transition tail, not the
100%-contaminated burst core (which both runs correctly drop entirely via
`min_fit`). Rigidity residuals here straddle the 100 mm threshold: the
gate rejects only ~41% of raw coincidences (69/169) vs ~12% (13/112) in an
adjacent clean window — confirmed via the new run's per-step
`"rigidity gate dropped N/M"` log lines (`grep "17:29:21" ... | grep
"rigidity gate dropped"`). `z_ref` stayed correctly anchored at the prior
window's pose (842.5 mm) throughout — not a poisoned-anchor bug.

The degradation vs. the old run is attributed to `dab37df` ("Decouple
monitor min-fit floor from raw batch size"): pre-fix, a window's raw batch
was fixed at exactly `--min-fit` coincidences and gated once; if survivors
fell short the *whole window silently dropped*. Post-fix, a short-of-floor
window keeps pulling in more raw coincidences and re-gating until
survivors clear `min_fit` (cap `5x`). For a partially-contaminated stretch
this is correct in general (recovers windows that would've vanished
outright) but means this specific window had to reach for 69 extra raw
coincidences instead of closing on a smaller/luckier sample, dragging in
proportionally more sub-threshold contamination for `fit_probe_pose`'s own
Mahalanobis inlier selection to fight — hence fewer trusted inliers (17
vs 22) and higher rms.

**This is a known, already-documented limitation**, not a new bug — see
"two windows just after each burst... remain marginal but not
catastrophic" in
`docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md`. The fix
commits didn't introduce contamination; they made the monitor try harder
(correctly) to fit through marginal windows instead of silently discarding
them, which incidentally surfaces this pre-existing edge case more
visibly in the printed table.

## Open caveat — old baseline provenance is not fully verified

`pipeline_out/monitor_window_run_rigidity_fixed.log` has **zero**
gate-activity log lines (`grep -c "rigidity gate dropped"` → 0) and no
`=== Run configuration ===` header, yet its `log.warning(...)` lines
(GPS/position-block mismatches) print unprefixed — implying
`logging.basicConfig(format="%(message)s")` was active. Current code only
calls `basicConfig` once, in `main()`, added by `cac14b5` (which also adds
the `=== Run configuration ===` header) — so the log's formatting and its
missing header are mutually inconsistent with any single point in this
repo's git history. Likely explanation: that baseline was invoked via a
different entry point (ad hoc script calling `monitor_probe()` directly
with its own `basicConfig(level=WARNING)`, suppressing `logger.info` gate
messages) rather than the `monrad-monitor` CLI — but this wasn't
confirmed. Git commit timestamps for this branch also don't reliably
bracket when the baseline file was actually produced. Treat exact
old-run parameters as reconstructed/best-guess, not certain.

## Not done / next steps

The user asked, and this session ended before acting on it:
> "want me to try either `--max-off-probe-mm` or tightening
> `--max-rigidity-resid-mm` below 100 for this stretch specifically?"

Next session should:
1. Try `--max-off-probe-mm` (footprint gate, stacks after rigidity) and/or
   a lower `--max-rigidity-resid-mm` (e.g. 60–80mm) on the same testLab
   run, focused on whether the `17:29:21–17:36:3x` window either drops
   cleanly (correct, since it's genuinely mostly bad) or recovers a tight
   fit (also fine) — either is better than the current wide-error middle
   ground.
2. Check whether tightening the threshold trades off against the
   "ever-present wild-track baseline" — i.e. does it start dropping
   otherwise-clean windows elsewhere in the 12h run? Compare whole-run RMS
   distribution before/after, same as prior validation passes.
3. If a new threshold is adopted, update
   [[rigidity-footprint-gates-shipped]] memory and the "recommended
   thresholds" section of
   `docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md`.

## Suggested skills for next session

- **`/verify`** — after any threshold change, drive `monrad-monitor`
  end-to-end on `data/0_testLab_20210723` and confirm the target window's
  behavior before declaring it fixed.
- **`code-review`** (if `_run_gates`/`monitor_probe` logic changes) — the
  cold-start / raw-batch-growth logic in `src/monrad/monitor/timeseries.py`
  has already had two real-data-only bugs found post-hoc
  ([[rigidity-footprint-gates-shipped]]); worth a careful pass on any
  further changes there.

## Key files

- `src/monrad/monitor/timeseries.py` — `monitor_probe`, `_run_gates`,
  `COLD_START_REFIT_STRIDE`, `RAW_CAP_MULTIPLIER`.
- `src/monrad/pose/optimize.py` — `filter_rigidity`, `filter_off_probe`.
- `pipeline_out/monitor_window_run_2026-07-09/pose_timeseries.csv` — this
  session's new-code run, kept for future diffing (untracked, like the
  rest of `pipeline_out/`).
- `pipeline_out/monitor_window_run_rigidity_fixed/pose_timeseries.csv` —
  archived baseline, provenance uncertain (see caveat above).
