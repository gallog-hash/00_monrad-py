# Handoff: time-varying telescope alignment (monrad-monitor/multiprobe)

Date: 2026-07-15
Branch: `feat/monitor-time-varying-alignment` — pushed to `origin`, **PR not yet opened**.
Commit: `94d4b8c` (11 files, +796/−24).

## What shipped

`monrad-monitor` / `monrad-multiprobe` `--alignment` now accepts a **directory**
of `alignment_<label>.json` (what a `--interval-hours` run of `monrad-align`
writes) for time-varying alignment: the driver switches the active
`AlignmentCorrection` per window as the coincidence stream crosses UTC window
boundaries. A single file / one-file directory is unchanged (static).

Full behavioral spec, the design rationale, and the correctness traps are
already captured — do **not** re-derive them, read these instead:
- Commit message on `94d4b8c` — the authoritative summary (feature + the
  filename-vs-UTC fix). `git show 94d4b8c --stat`, `git show 94d4b8c` for the body.
- Persistent memory `time-varying-alignment-shipped` (in the auto-memory store,
  loaded each session) — the non-obvious gotchas: DAQ filenames are local CEST
  (+2h) not UTC; per-window align fits re-anchor to `utc0` so only the earliest
  window's `t_ns` is absolute; `reconstruct_stream` needs contiguous files from
  the header origin (mid-acquisition subsets give garbage timing).
- README.md "Alignment calibration + hardware-drift monitor" section and
  CLAUDE.md command block — updated user-facing docs (directory form + UTC bounds).
- The approved implementation plan lived at `~/.claude/plans/snoopy-dancing-zebra.md`
  (machine-local, will NOT travel to another machine) — its content is now
  subsumed by the commit + memory above, so nothing is lost.

## State: done and verified

- Full test suite green (311 passed + the added `test_daq_utc_offset` unit test);
  ruff check + format clean (pre-commit hook passed on commit).
- New `tests/test_alignment_schedule.py`; additions to `test_align.py`,
  `test_monitor_timeseries.py`, `test_monitor_multiprobe.py`.
- Real-data verify on `data/0_testLab_20210723` (with the dataset's correct
  `--z-tel 0 -1340 -670`): `monrad-align --interval-hours 6` → 11 UTC-bounded
  JSONs; `monrad-monitor --alignment <dir>` loads the schedule, validates
  z_tel per file, and a contiguous run switches `_060000`→`_120000` exactly
  once at the corrected UTC `09:55:47` (was the buggy `12:00` before the fix).

## Not yet done / next steps

1. **Open the PR** (the only strictly-remaining task):
   https://github.com/gallog-hash/00_monrad-py/pull/new/feat/monitor-time-varying-alignment
   — reuse the `94d4b8c` commit body as the description. The user was about to
   decide this when the session handed off.
2. **Optional `/code-review` before merge** — the plan flagged this: the change
   touches streaming state (`_WindowAccumulator`, `stream_coincidences`) and the
   multiprobe shared-search identity invariant (`multiprobe.py:183-189`), where
   correctness is easy to get subtly wrong.

## Known limitations (documented, not bugs to fix now)

- The DAQ→UTC offset is a single constant per acquisition (`_daq_utc_offset` =
  `file_ts[0] − utc0`). A **DST change mid-acquisition** would break it. Fine for
  the July CEST testLab data; note it if a run spans a DST boundary.
- Local `pipeline_out/` scratch (untracked) and the scratchpad symlink subsets
  used for verification are regenerable from the commands in the memory/README;
  nothing there needs committing.

## Suggested skills for the next session

- **`review`** (the GitHub-PR reviewer) once the PR is open, or **`code-review`**
  (medium/high) on the branch diff before opening it — see step 2 above.
- **`verify`** if you extend the feature further (e.g. gap/coverage warnings when
  a run's UTC span falls outside all loaded windows) — drive the real CLI on
  `data/0_testLab_20210723`, slicing **contiguously from the first file** (a
  mid-acquisition subset reconstructs to garbage times; this bit the last session).
