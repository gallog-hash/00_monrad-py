# Handoff — `--mahal-cut` / `--max-per-plane` / `--coincidence-window-ns` shipped + pushed

**Date:** 2026-07-28
**Branch:** `study/geometric-cut-scan` — **pushed** to `origin`, no PR opened yet
**Commit:** `ff9be9f` — "Expose mahal_cut, max_per_plane and window_ns as real parameters"
**PR link (unused):** https://github.com/gallog-hash/00_monrad-py/pull/new/study/geometric-cut-scan

## What this session did

Executed, in full, the plan at
`~/.claude/plans/plan-1-4-and-include-nifty-sprout.md` (all five steps, not
just 1–4 as the filename suggests). That plan holds the complete design
rationale — the measured evidence for both findings, the flag-naming argument,
the acceptance-vs-selection tier distinction, and the per-driver wiring table.
**Read the plan rather than re-deriving any of it.** The commit message on
`ff9be9f` summarises the same ground.

This session's contribution beyond the plan: the tests, the before/after
real-data regression, and the four flag-effect runs below.

Note the plan's stated "251 tests baseline" was stale — the suite is 396 tests.

## Verification results (don't re-run unless something regresses)

- `uv run pytest -q`: **396 passed** in 22:49.
- `uv run ruff check .` + `ruff format --check .`: clean. Pre-commit hooks
  passed on commit.
- **Defaults unchanged.** `run_pipeline.py` on the same input, `main` (via a
  throwaway worktree) vs branch: `diff summary.txt` shows *only* the output-dir
  line and the three new config-echo lines. Every physics number identical.
- **All three flags reach the physics:**

  | Run | Raw coincidences | n_inliers | z_p (mm) |
  |---|---|---|---|
  | baseline | 2201 | 148 | +841.2 ± 3.1 |
  | `--mahal-cut 3.0` | 2201 | 136 | +843.3 ± 3.3 |
  | `--max-per-plane 64` | 2201 | 158 | +841.9 ± 3.1 |
  | `--coincidence-window-ns 100` | 1838 | 121 | +840.6 ± 3.5 |

  `--max-per-plane 64` changing the result is the direct confirmation the cap
  was binding, exactly as the plan's 17.2%-of-searched-events measurement
  predicted.

- **Scan harness:** `--stage decode --max-per-plane 64
  --coincidence-window-ns 200` records both in the cache meta;
  `--stage replay --mahal-grid 3 4 6` gives 183 / 193 / 203 inliers over one
  cache — the axis survived the monkeypatch removal intact.

### How the real-data runs were set up (rebuild if needed)

The full `data/0_testLab_20210723/Base` is 724 file pairs / 2.7 GB — far too
slow for a turnaround. All runs above used a **3-file-pair slice** copied to
scratch (`mini/Base`, `mini/Probe_0`: the header plus the first three
chronological `*_GPS.bin` + matching `*.bin`). Scratch is gone after reboot;
rebuild by copying `*header*.txt` plus the first 3 GPS/bin pairs from each of
`Base`/`Probe_0`. The slice yields **2201 raw coincidences**, matching the
MATLAB reference count (memory `matlab-vs-python-coincidence-yield`), so it is
the same subset that comparison used.

Flags used throughout: `--z-tel 0 -1340 -670 --max-rigidity-resid-mm 100
--n-probe-ch 40` (per memories `testlab-20210723-plane-z-order`,
`testlab-20210723-probe-size`, `rigidity-footprint-gates-shipped`).

## Deviations from the plan worth knowing

1. **Real-data verification ran on the 3-pair slice, not the full acquisition.**
   The slice exercises every path the plan's verification targeted, but it is
   not the literal full-directory comparison the plan wrote. If a
   full-acquisition confirmation matters before merge, that is the gap.
2. **`None`-sentinel resolution happens in each driver's `main()`**, not
   threaded down. `monitor_probe`/`monitor_probes` take concrete defaults
   (`MAX_PER_PLANE_DEFAULT`, `WINDOW_NS_DEFAULT`) because they are also a
   public Python API where a real default reads better than `None`.
   `--mahal-cut` stays `None` end-to-end since `fit_probe_pose` owns that
   default. This differs slightly from the `add_chi2_track_args` precedent,
   which keeps `None` all the way to `PoseFitter`.

## Suggested next steps

- Open the PR against `main`. Nothing is blocking it; the branch also carries
  the two earlier scan-harness commits (`e16413f`, `11ca3f2`) that have never
  been reviewed, so the PR is three commits wide — consider whether to split.
- Optional: the full-acquisition regression from deviation 1.
- The new knobs are now scannable. Per the plan's finding 1, `mahal_cut` is
  "arguably more useful *tightened* than `chi2_track` is loosened" — a
  `--mahal-grid` sweep on testLab/testMili against the shipped rigidity gate is
  the obvious follow-up study, and the harness already supports it with no
  re-decode.
- `--max-per-plane` scanning needs one decode pass per value (it is not a grid
  axis, by design). Budget accordingly.

## Suggested skills

- `/code-review` — for the working diff before opening the PR. Note
  `/code-review ultra` is user-triggered and billed; an agent cannot launch it.
- `/astral:ruff` — this repo mandates Ruff via `uv run` for all lint/format
  (see `CLAUDE.md`). Pre-commit hooks already enforce it.
- `/astral:uv` — all commands go through `uv run`; `uv sync` after any
  dependency change.
- `/security-review` — only if the PR scope widens; this change is
  parameter-threading with no I/O or auth surface.

## Related artifacts

- Plan: `~/.claude/plans/plan-1-4-and-include-nifty-sprout.md` (not in-repo)
- Prior art for the same pattern:
  `docs/handoffs/2026-07-20-chi2-track-max-cluster-width-flags-shipped.md`
- Scan harness origin: `docs/handoffs/2026-07-21-cut-scan-harness.md`
- Diff: `git show ff9be9f` (20 files, +817/−81)
