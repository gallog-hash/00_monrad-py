# Handoff — `--chi2-track` / `--max-cluster-width` gates implemented + committed

**Date:** 2026-07-20
**Branch:** `feat/chi2-track-max-cluster-width-flags` (local only — **not pushed**, no PR yet)
**Commit:** `39c7c7d` — "Add --chi2-track and --max-cluster-width overrides to the pose-fit gates"

## What this session did

Implemented the plan at `~/.claude/plans/revise-the-plan-claude-plans-add-chi2-tr-synchronous-treasure.md`
in full (Parts 1–3): threaded `chi2_track: float | None` and
`max_cluster_width: int | None` through `PoseFitter`, `decode_position`/
`_decode_axis`, `reconstruct_plane_candidates`, and exposed both as CLI flags
on `scripts/run_pipeline.py`, `monrad-monitor`, and `monrad-multiprobe`. Both
default to `None` (4.0 / off) — zero behavior change until a flag is passed.
See the plan doc for the full design rationale (the `None`-sentinel pattern,
why `_CHI2_TRACK` stays a module constant, why Part 3's shared `cli_args.py`
was deferred). Don't re-derive that reasoning here — read the plan.

This session's own contribution beyond the plan: 10 new unit tests
(`tests/test_stage3.py::TestMaxClusterWidth`,
`tests/test_stage5.py::TestChi2TrackOverride` /
`TestMaxClusterWidthOverride`), a real-data verification run, and an 8-angle
parallel code review with two bugs found and fixed before commit.

## Verification results (don't re-run unless something regresses)

- `uv run pytest -q`: **327/327 pass** (317 pre-existing + 10 new).
- Real data, `data/0_testLab_20210723` (`Base` + `Probe_0`,
  `--z-tel 0 -1340 -670`):
  | | default (chi2=4.0, no cap) | `--chi2-track 37 --max-cluster-width 4` |
  |---|---|---|
  | `chi2_track_cut` rejections | 66,241 | 20,175 |
  | `zero_candidate_plane` rejections | 106,172 | 116,808 (cap adds strictness here) |
  | `probe_quality` rejections | 100,241 | 129,123 (cap adds strictness here) |
  | accepted (fed to pose fit) | 53,871 | 68,557 |
  | fitted pose (t_x, t_y, θ, z_p) | 180.7, 233.3, -0.6°, 841.0 | 181.1, 232.9, -0.6°, 840.0 |

  Net effect: loosening χ² dominates the two cap-driven tightenings; more
  coincidences survive, pose stays essentially stable. Confirms the gates are
  wired correctly and behave as the plan predicted.

  **Follow-up: reproduced the plan's exact `~182/2201` baseline.** The run
  above used the *full* `0_testLab_20210723` acquisition (1449 files), not the
  specific 3-file-pair subset the MATLAB comparison (memory
  `matlab-vs-python-coincidence-yield`) used — so it didn't literally check
  the plan's stated verification target. Built a subset dir from the first 3
  chronological file pairs of `Base`/`Probe_0` (copies, not symlinks, under
  `/tmp/claude-1000/subset3/` — gone after reboot, rebuild via `ls | sort |
  head -3` on each of `Base`/`Probe_0` if needed again) and reran:

  | | default (chi2=4.0, no cap) | `--chi2-track 37 --max-cluster-width 4` |
  |---|---|---|
  | raw coincidences | 2201 (exact match to memory) | 2201 |
  | `no_anchor_plane` rejections | 1070 | 1019 |
  | `chi2_track_cut` rejections | 207 | 67 |
  | accepted | 197 (8.9%, memory recorded 182/8.3% — close, likely drifted by
    PR #22's rigidity/off-probe gates landing after that measurement) | 242 (11.0%) |

  Confirms the plan's letter: raw-coincidence count matches the MATLAB
  comparison exactly, default acceptance is in the same ballpark as the
  recorded 182/2201, and loosening moves acceptance up (197→242) — **but**
  `no_anchor_plane` is now the dominant rejection reason by far (1019 of
  2201), and neither `--chi2-track` nor `--max-cluster-width` touch it, so
  this plan alone cannot close the full 5x MATLAB gap. That gate
  (`--min-anchor-planes`) already exists and is a separate, orthogonal lever
  — memory `testmili-20220905-anchor-and-gate-interplay` warns it interacts
  non-obviously with the rigidity gate, so don't tune it in isolation.

  Full run logs (not
  committed, regenerate if needed):
  `/tmp/claude-1000/pipeline_baseline/summary.txt` and
  `/tmp/claude-1000/pipeline_chi2_37/summary.txt` — **these are in `/tmp`,
  gone after reboot; nothing here is preserved beyond this document.**
- `@macro.args` flag resolution confirmed directly against
  `monrad.monitor.timeseries._parse_args` / `monrad.monitor.multiprobe._parse_args`
  (both resolve `--chi2-track`/`--max-cluster-width` correctly through the
  `@file` mechanism).
- `ruff check` / `ruff format`: clean.

## Code review — 2 bugs fixed, 5 findings deferred

Ran `/code-review high` (8 parallel finder angles + verify pass) against the
full working-tree diff before committing. Findings and disposition:

**Fixed (in commit `39c7c7d`):**
1. `--chi2-track nan` slipped past validation in all three drivers — Python's
   `float('nan') <= 0` is `False`, so the `<= 0` check didn't reject it,
   silently disabling the whole `chi2_track_cut` gate. Fixed to
   `not args.chi2_track > 0` (NaN comparisons are always `False`, so `not (nan
   > 0)` is `True` → correctly rejected) in `scripts/run_pipeline.py`,
   `src/monrad/monitor/timeseries.py`, `src/monrad/monitor/multiprobe.py`.
2. Stale comment in `scripts/run_pipeline.py` (~line 679) still said
   "`<_CHI2_TRACK` cut" after the threshold became configurable — reworded to
   "chi2_track cut".

**Deferred, not acted on this session** (all low-severity; none are
regressions or block merge):
1. **CLI validation/argparse/logging is copy-pasted 3x** across
   `run_pipeline.py`/`timeseries.py`/`multiprobe.py` instead of joining the
   existing `validate_probe_footprint` (`src/monrad/monitor/io.py`) pattern.
   The plan already explicitly defers a shared `cli_args.py` refactor to a
   **separate follow-up PR** (see plan doc, "Deferred" section) — this finding
   just reconfirms that scope call, it isn't new information.
2. `tests/test_stage3.py::_write_block` is now defined identically 3x in one
   file (my new `TestMaxClusterWidth` class added a third copy of a helper
   that `TestFibersPerRibbon`/`TestPlaneCandidates` already had — pre-existing
   duplication, not introduced fresh, but not fixed either).
3. `tests/test_stage5.py`'s two new classes (`TestChi2TrackOverride`,
   `TestMaxClusterWidthOverride`) each define their own identical `_fitter()`
   helper instead of sharing one.
4. `src/monrad/reconstruction/candidates.py`'s new `max_cluster_width` filter
   is folded into the existing dense candidate-tuple list comprehension
   rather than pre-filtering — a minor readability nit, not a correctness
   issue (verified the cap does run before the `max_per_plane` slice, as the
   docstring promises).
5. **Latent design gap, not currently reachable:** `decode_position`'s
   `'unresolved'`-fallback path populates `Hit.candidates_x/y` via
   `_axis_candidates()`, which has no `max_cluster_width` param — so if a
   future caller ever combined `decode_position(n_cols=3,
   max_cluster_width=...)` with `disambiguate_telescope_hits` (as
   `src/monrad/alignment/accumulator.py:291` already does for its own,
   uncapped hits), the 2-plane projection could silently reinstate an
   over-cap candidate as `quality="cluster"`. **No live caller does this
   today** (stage 5's telescope decode uses `reconstruct_plane_candidates`,
   which *does* apply the cap correctly, not this path) — flagging so a future
   session doesn't get bitten if `max_cluster_width` is ever threaded into the
   stage-4 alignment path (`fit_daily_alignment`, which today intentionally
   does **not** receive it — see plan doc, Part 2).

## Current repo state

- Branch `feat/chi2-track-max-cluster-width-flags`, one commit (`39c7c7d`),
  **not pushed**, no PR opened.
- `git status` also shows two pre-existing untracked paths from before this
  session (`.claude/`, `pipeline_out/`) — left alone, not part of this change,
  do not assume they need cleanup.

## Next steps

1. Push + open the PR — branch is pushed (`origin/feat/chi2-track-max-cluster-width-flags`),
   no PR opened yet as of this update.
2. Optional: act on any of the 5 deferred findings above (all are genuinely
   optional — none block merge). The CLI-validation-triplication one (#1) is
   the most consequential but is explicitly slated for its own follow-up PR
   per the plan, not this one.
3. **Answered, not fully closed:** the plan's motivating question — "does
   loosening `--chi2-track`/`--max-cluster-width` close the ~5x MATLAB-vs-
   Python stage-3 yield gap?" — was re-run on the exact 3-file-pair/2201-
   coincidence subset (see the "Follow-up" note above): acceptance moves from
   197→242 (8.9%→11.0%) but stays far short of MATLAB's 44% because
   `no_anchor_plane` (not touched by either new flag) is now the dominant
   rejection reason on this subset. Closing the rest of the gap needs
   `--min-anchor-planes` tuning (existing flag) or a new gate — **out of
   scope for this plan**, which only promised to make the two named
   thresholds tunable, not to close the yield gap outright.

## Suggested skills

- `/code-review` (or `/code-review high`) — if more changes land on this
  branch before a PR, re-run before pushing; the review this session ran
  caught a real bug (NaN validation) that would have shipped otherwise.
- `/verify` — if any of the deferred findings get addressed, verify the
  change end-to-end on real data rather than trusting unit tests alone (this
  codebase's own lesson from the MATLAB comparison: unit tests can't surface
  a 5x yield gap).
- `gh pr create` (no dedicated skill — just the standard PR flow) once ready
  to push; there is no open PR for this branch yet.
