# `--max-abs-resid` removed: rigidity gate made it redundant, real-data run confirmed no benefit

Written 2026-07-07. Branch `feat/probe-monitoring`. Follow-up to
`docs/handoffs/2026-07-07-rigidity-footprint-gates-validated.md` (rigidity/
footprint gates, shipped and validated same day) and
`docs/handoffs/2026-07-06-in-fit-abs-mm-recovery-plan.md` (the opt-in
in-fit absolute-mm residual cut, shipped 2026-07-06 in commit `64ec238`).

## What happened this session

1. Re-read the rigidity/footprint validation handoff and asked whether the
   earlier in-fit `max_abs_resid_mm` cut (`--max-abs-resid`) was still
   pulling weight now that the pre-fit rigidity gate exists — both target
   the same failure mode (wide-angle "wild" telescope tracks that evade the
   Mahalanobis `d>4` cut because their inflated `var_b·z_p²` keeps their
   Mahalanobis distance low).
2. Ran the combined config on the full `data/0_testLab_20210723` day
   (`--z-tel 0 -1340 -670 --n-probe-ch 40 --min-anchor-planes 1
   --window-s 300 --max-rigidity-resid-mm 100 --max-off-probe-mm 100
   --max-abs-resid 20`) and diffed it window-by-window against the
   rigidity+footprint-only run (same config minus `--max-abs-resid`).
   Both configs kept the same 140/143 windows (still dropping exactly the
   two known 100%-contaminated bursts).
3. Result: abs-resid fired in 138/140 windows but the effect was
   noise-level — median `|Δz_p|` 2.2 mm, mean 3.0 mm (only 3/140 windows
   moved >10 mm, and those are already-thin/flagged-marginal windows where
   σ_zp is comparably large). Whole-run stats were statistically
   unchanged: resid RMS median 35.1→35.2 mm, max 129.7→129.9 mm; z_p
   mean 840.8→840.7 mm, std 10.2→**10.7** mm (slightly worse dispersion,
   not better). The two marginal post-burst windows it was meant to
   rescue were unaffected (`17:30:00–17:35:02`, completely untouched) or
   a wash (`18:47:17–18:52:32`: n_inliers 8→6, rms 129.7→129.9, z_p moved
   closer to baseline but σ_zp grew 24.9→28.2 mm).
4. Conclusion: the pre-fit rigidity gate (cross-coincidence, anchored to
   the *previous* window's accepted pose) already exiles the same wild
   tracks the in-fit abs-mm cut was designed to catch, without that cut's
   self-referential weakness (it scored against the *current*,
   potentially-still-contaminated fit). Once rigidity/footprint are
   applied, abs-resid adds nothing worth keeping.
5. **Removed `max_abs_resid_mm` / `--max-abs-resid` from the codebase
   entirely** (not just left opt-in-off): the third robust-stage block in
   `fit_probe_pose` (`src/monrad/pose/optimize.py`), the `PoseFitter`
   plumbing (`src/monrad/pose/fitter.py`), the `monrad-monitor` CLI flag
   (`src/monrad/monitor/timeseries.py`), the `run_pipeline.py` CLI flag,
   and the associated tests (`TestAbsResidCut` in `tests/test_stage5.py`;
   two plumb-through tests in `tests/test_monitor_timeseries.py`). Left
   the historical handoff docs (`2026-07-06-in-fit-abs-mm-recovery-plan.md`
   and other docs that mention it in passing) untouched — they're a record
   of what was tried, not code.

## Current state — uncommitted

```
 scripts/run_pipeline.py          |  13 ----
 src/monrad/monitor/timeseries.py |  26 +-------
 src/monrad/pose/fitter.py        |   5 --
 src/monrad/pose/optimize.py      |  60 -----------------
 tests/test_monitor_timeseries.py |  45 -------------
 tests/test_stage5.py             | 135 ---------------------------------------
 6 files changed, 1 insertion(+), 283 deletions(-)
```

Full suite green: `uv run pytest` → 219 passed. `uv run ruff check .` clean
on the touched files. **Not committed** — the user had not yet said whether
to commit, or whether to write this handoff, when the session ended (this
doc was requested; commit was not).

Also present but untouched/unrelated to this change: two large scratch
run directories from the investigation, `pipeline_out/monitor_combined_rigidity_absresid/`
and `pipeline_out/monitor_rigidity_footprint_only/` (each one `pose_timeseries.csv`,
~140 rows), plus a pre-existing untracked `pipeline_out/` and `.claude/` from
before this session. None of these are tracked by git; safe to delete if
disk space matters, or leave for anyone who wants to re-diff without
re-running (see the two `uv run monrad-monitor …` invocations above to
reproduce).

## Not done / open follow-ups

- **Commit not made.** Next session should confirm with the user whether to
  commit this removal (and with what message — a git-log-style summary is
  already implicit in this handoff) before doing so.
- The rigidity/footprint gates themselves remain **opt-in** in
  `monrad-monitor` (`--max-rigidity-resid-mm` / `--max-off-probe-mm`,
  default `None`) and are **not** wired into `PoseFitter`'s streaming
  pipeline path (`scripts/run_pipeline.py`) — this was already true before
  this session (see the validation handoff's "Not done" section) and is
  unaffected by the abs-resid removal.
- No CLI diagnostic prints the rigidity pairwise-score distribution yet
  (still open from the validation handoff).
- The third dropped window from the 143→140 full-day run (only 2 of which
  are the known bursts) is still not identified or investigated.

## Suggested skills

- `verify` — if a future session picks this back up mid-flight (e.g. to
  finish the commit or extend the removal to any missed reference), rerun
  it before considering the change done; a grep-based check
  (`grep -rln "max_abs_resid\|max-abs-resid" --include=*.py .`) was used
  this session and returned clean, but re-verify after any further edits.
- `astral:ruff` — used this session for lint; reuse for any follow-up
  edits.
