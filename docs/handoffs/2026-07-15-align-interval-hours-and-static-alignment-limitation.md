# Handoff: monrad-align --interval-hours + a discovered monrad-monitor limitation

Date: 2026-07-15
Branch: `feat/align-interval-hours` (pushed, in sync with `origin`, PR not yet opened)

## What shipped this session

`monrad-align` was changed from "fit exactly one day" to "process the entire
dataset by default, one refit per `--interval-hours` window (default 24 =
one/day)". Full behavioral detail, flag docs, and examples are in
`README.md`'s "Alignment calibration + hardware-drift monitor" section and
`CLAUDE.md`'s command block — read those rather than this doc for the feature
spec.

Commits on `feat/align-interval-hours` (all pushed):
- `aa5ef74` — core feature: `compute_alignment()` + `select_alignment_windows()`
  / `group_by_interval()` in `src/monrad/monitor/io.py`, new `--interval-hours`
  CLI flag, `--date` extended to also match a day-prefix under sub-day
  intervals. `compute_daily_alignment()` (single-window) kept unchanged as
  the building block — still backs the monitor drivers' auto-fit fallback and
  existing tests.
- `1e7bc1d` — fixed stale "daily calibration" wording left over in
  README.md/CLAUDE.md package-structure comments.
- `b0ebda0` — blank-line separators between windows' log blocks in a
  multi-window run (readability fix requested after visually inspecting output).

Tests: `tests/test_align.py` (26 tests, all passing) covers the new
`group_by_interval`/`select_alignment_windows` unit behavior and
`compute_alignment` end-to-end (whole-dataset default, single-day
restriction, single-plot-per-run). Full suite (290 tests) passes; lint/format
clean.

Not yet done: no PR opened. GitHub offered
`https://github.com/gallog-hash/00_monrad-py/pull/new/feat/align-interval-hours`.

## Real-data validation state (local, untracked)

Ran against `data/0_testLab_20210723/Base` with the dataset's established
correct plane ordering `--z-tel 0 -1340 -670` (see prior-session memory:
`testlab-20210723-plane-z-order`). Two output directories exist locally
under `pipeline_out/` (untracked, not committed — regenerable from the
commands below):

- `pipeline_out/alignment/` — default 24h-interval run, one JSON per day
  (`alignment_20210723.json`, `_20210724.json`, `_20210725.json`).
- `pipeline_out/alignment_interval_test/` — `--interval-hours 6`, whole
  dataset (no `--date`), 11 window JSONs (`alignment_20210723_060000.json`
  ... `_20210725_180000.json`; the day-1 `000000` window is absent because
  that day's acquisition starts at ~11:35 UTC, after the 00:00–06:00 window).
  Regenerate with:
  ```bash
  uv run monrad-align --telescope data/0_testLab_20210723/Base \
      --z-tel 0 -1340 -670 --interval-hours 6 \
      --out pipeline_out/alignment_interval_test
  ```

**Caution for the next session**: this directory was found once already
containing 11 files fit with the *wrong*, positive `z_tel=[0, 1340, 670]`
(should be negative per the project convention) — that state was produced
outside what's visible in this conversation's tool history (not by any
command run in this session up to that point), so its origin is unexplained.
It has since been deleted and regenerated correctly (confirmed
`z_tel=[0.0, -1340.0, -670.0]` in the JSON). If you find `pipeline_out/`
contents that don't match what's described here, verify `z_tel` in the JSON
before trusting the file — `load_alignment(..., expect_z_tel=...)` will
raise on a real mismatch, but only if you pass the matching `--z-tel` to
`monrad-monitor`/`monrad-multiprobe` in the first place.

## Open finding: monrad-monitor has no time-varying alignment

This surfaced while trying to feed a sub-day-window alignment JSON into
`monrad-monitor --alignment` and is the most likely next-session topic.

**The finding**: `monrad-align` can now characterize alignment drift at
arbitrary time resolution (`--interval-hours`), but `monrad-monitor` (and,
by the same code path, `monrad-multiprobe`) has no mechanism to *consume*
that per-window information. Both the `--alignment <file>` path and the
default in-run auto-fit path resolve to exactly **one**
`AlignmentCorrection` object, computed once before the streaming loop starts,
then applied uniformly to every coincidence for the entire run — regardless
of how many days/windows of probe data the run actually covers.

Code pointer: `src/monrad/monitor/timeseries.py:607-614`
```python
if alignment_path is not None:
    alignment = load_alignment(alignment_path, expect_z_tel=z_tel)
else:
    alignment, _ = fit_alignment(tel, z_tel, tot_thresh=tot_thresh, tot_weights=tot_weights)
z_corr = alignment.corrected_z_tel(z_tel)
```
`fit_alignment` (`src/monrad/monitor/io.py`) itself just calls
`select_day_files(tel, date=None, n_files=DAILY_ALIGNMENT_N_FILES)` — the
earliest day's first 3 files, once. This is unchanged by this session's work
and was already true before `--interval-hours` existed; the new flag just
made the mismatch between `monrad-align`'s (now fine-grained) capability and
`monrad-monitor`'s (still single-snapshot) consumption visible.

**Not yet decided** — this was left as an open question to the user
(conversation ended on a clarifying answer, not a decision). Options
discussed but not chosen:
1. Restrict a given `monrad-monitor`/`multiprobe` run's probe-data time range
   to roughly match the alignment window's fit window, so the static
   correction stays valid for that whole run (simplest, no code change,
   just an operational convention).
2. Fall back to the whole-day (`--interval-hours 24`, i.e. default)
   alignment file when monitoring a full day/run, since that's already
   scoped to match a day-long run.
3. Implement actual time-varying alignment in `monrad-monitor`: load/accept
   multiple `alignment_*.json` windows and switch the active
   `AlignmentCorrection` as the streaming loop crosses window boundaries
   (real feature work — touches `_WindowAccumulator`,
   `stream_coincidences`, and the CLI's `--alignment` flag, which currently
   takes one path).

## Suggested skills for the next session

- **`verify`** — before extending `monrad-monitor`/`multiprobe` with any
  time-varying alignment logic, use this to drive the actual CLI end-to-end
  against `data/0_testLab_20210723/` rather than trusting tests alone (this
  session found the real-world z_tel-sign issue only by inspecting JSON
  output directly, not from a test failure).
- **`code-review`** (medium or high effort) — if option 3 above is chosen
  and implemented, this touches streaming state (`_WindowAccumulator`) and
  correctness there is easy to get subtly wrong; worth a review pass before
  merging.
- If the user wants a second opinion on which of the three options to
  pursue, this is a good candidate for **`llm-council`** — it's a genuine
  architecture tradeoff (operational convention vs. new feature) without an
  obviously-correct answer yet.

## Relevant prior-session memory (already in the persistent memory store)

- `testlab-20210723-plane-z-order` — this dataset's correct `--z-tel` is
  `0 -1340 -670` (not z-ordered filenames).
- `testlab-20210723-probe-size` — probe is 400×400mm (`--n-probe-ch 40`),
  not the stale 30cm once in CLAUDE.md.
- `testlab-20210723-probe0-probe1-identical` — `Probe_0`/`Probe_1` in this
  dataset are byte-identical duplicates, not a real second probe; fine for
  exercising the CLI, not for divergence testing.
- `daily-alignment-shipped` — the original `monrad-align` (single-day) plus
  the `#18` fix that made the in-run auto-fit match it; this session's work
  builds directly on top of that.
