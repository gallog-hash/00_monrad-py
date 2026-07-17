# Handoff — MATLAB reference vs `monrad-py` pipeline comparison

**Date:** 2026-07-17
**Branch:** `main` (no source changes made — see "Repo state")
**Reference implementation:** `/home/gallog/MATLAB/2023-03_Probe_New_Test/`
(entry point `master_BuSmonitor.m` + `projectFun/`; there is **no**
`masterBusMonitor` directory despite the name)

## What this session did

Compared the `monrad-monitor` pipeline against the legacy MATLAB implementation,
first by reading both codebases, then by **running both side by side** on real
data (3 file pairs of `data/0_testLab_20210723`, Base + Probe_0) and comparing
the coincidences event-by-event.

## Headline result

**Stages 1–2 (timing + coincidence identification) are functionally equivalent.
The entire divergence lives in stage-3 geometric cuts.**

| | |
|---|---|
| Raw coincidences, MATLAB | 2201 |
| Raw coincidences, Python | 2201 |
| **Pairs identical in both** (verified by GEN event identity) | **2201 — 100.00%** |
| MATLAB-only / Python-only | 0 / 0 |
| Base event coupled to a *different* probe partner | 0 |

Both sides match strictly one-to-one; neither reuses an event. Timestamps
differ by ~16 ns RMS, zero bias (integer-ns vs double `datetime` rounding).
The 200 ns window is identical on both sides.

**But after geometric cuts:** MATLAB accepts **969** (44%), Python accepts
**182** (8.3%) — ~5x fewer — and the overlap is **not nested**: 49 of Python's
182 (27%) are events MATLAB rejects. Each pipeline accepts events the other
does not.

## Where the detail lives — do not re-derive

Full findings are in project memory at
`~/.claude/projects/-home-gallog-00-work-new-00-monrad-new-00-monrad-py/memory/`.
These are **not** in this repo, but they **do sync across machines** via the
private `claude-config` repo (`github.com/gallog-hash/claude-config`, commit
`2a79db6` added the `!projects/*/memory/**` allowlist). On a fresh machine:
`git -C ~/.claude pull` before starting, or you will re-derive all of this from
scratch. Session transcripts deliberately stay machine-local — only `memory/`
syncs.

- `matlab-vs-python-coincidence-yield.md` — the comparison: raw-coincidence
  identity proof, the GEN ↔ `evt_seq` mapping recipe, the stage-3 funnel
  breakdown, tel_z sensitivity table.
- `chi2-track-cut-in-mm.md` — `_CHI2_TRACK = 4.0` converted to mm: ≈4.71 mm max
  residual for width-1 hits; the general unequal-σ formula; why the middle
  plane's σ carries 4x weight; the demonstrated mechanism for the 49
  Python-only events.
- `matlab-reference-pipeline-how-to-run.md` — **read this before touching the
  MATLAB side.** Headless harness recipe, the stale `nargin==1` path, where to
  patch for a pre-cut dump, and the landmines below.

## Landmines (learned the hard way)

- **`distributeFileInSubDir` MOVES raw files** into `Header/GPS/TRK` subdirs
  under `localPathRoot`. Never point the MATLAB side at real data. Always copy
  to a sandbox first.
- **`master_call0` at `k == 2` randomly halves the coincidence list**
  (`idr = randi(...)`), for the 2nd probe only. Single-probe runs are
  unaffected, but multi-probe MATLAB runs are **not reproducible**.
- `master_call0` writes a `<YYYYMMDD>_log.txt` into the **cwd** — run it from a
  scratch dir, or it litters the repo root (one was cleaned up this session).
- MATLAB's `Tout` holds only events that **passed** the track fit;
  `coinPerFile` is *not* the raw coincidence count. Comparing it against
  Python's stage-2 output is apples-to-oranges (this caused an initial false
  "1240 Python-only" result that the GEN check retired).

## Repo state

**No source changes.** Nothing to review or commit from this session. All
artifacts were either session-local scratchpad (now gone) or memory files
outside the repo. Pre-existing untracked: `.claude/`, `pipeline_out/`.

## Open questions / suggested next steps

Ordered roughly by value:

1. **Is `_CHI2_TRACK = 4.0` too tight?** This is the main open decision. It
   works out to ~4.71 mm vs MATLAB's 14.29 mm `ALIGNDIST`, and drives most of
   the ~5x yield gap. `_CHI2_TRACK ≈ 37` would match MATLAB at width 1. If
   Python is over-cutting, most statistics are being discarded — consistent
   with the long-standing "~0.2 accepted coincidences/s" (see memory
   `monitor-window-rate`). **Do not just loosen it blindly**: memory
   `testmili-20220905-anchor-and-gate-interplay` records that gates interact
   non-obviously, and the cut being σ-adaptive is physically *correct*
   behaviour that MATLAB lacks.
2. **Re-run Python with a fitted alignment** (this session used
   `AlignmentCorrection.identity()` to match MATLAB, which has no alignment
   stage at all). Should reduce χ² rejections and raise yield — quantifies how
   much of the gap is just missing alignment. Cheap, do this first.
3. **The `no_anchor_plane` gate costs 48.6%** of clusters and MATLAB has no
   equivalent (it runs a combinatorial "golden line fit" over all clusters).
   But `min_anchor_planes=0` is known harmful (memory
   `testmili-20220905-anchor-and-gate-interplay`). Worth understanding why the
   two positions differ so much.
4. **`probe_FiberPerFiberCh = 4` — UNRESOLVED.** The MATLAB conf sets this for
   the 40x40 probe (telescope is 10/10), and `monrad-py` only models
   `n_fibers_per_ribbon`. I traced it to `probeS` in `start_script0_Win10.m`
   and found no consumer, so it *looks* vestigial — but this was not verified.
   If it is live, our probe decode is missing a parameter for exactly the
   40x40 probe.
5. **Python's rigid-rotation model cannot represent a reflection.** MATLAB does
   no rotation fit at all; it does a discrete 4-way "coupling check"
   (`segmented_probe_position_calculation.m:250-294`) whose cases 2 and 4 are
   x/y *swaps* (det = −1). Check whether any real probe ever selects
   `xy_dir` 2 or 4 — if so, Python would land on a garbage θ.
6. **MATLAB's 2° opening-angle cut has no Python analogue**
   (`alpha_lthr`, same file, line 139) — it suppresses near-parallel track
   pairs where its pairwise solve degenerates. Note this is the *opposite* end
   of the lever from the reverted `--max-track-slope` gate (memory
   `wide-track-cut-gate-shipped`).
7. **Track-fit estimator differs:** MATLAB uses unweighted 3D PCA (total least
   squares, perpendicular residuals) via `linearRegression.m`; Python uses
   weighted LS of x(z)/y(z). Python's is more correct here (z is the controlled
   coordinate), but they diverge for steep tracks — which is also why the mm
   comparison in `chi2-track-cut-in-mm.md` carries a caveat.

## Architectural notes worth keeping in mind

The two pipelines solve the same physics with **different estimators**. MATLAB
is pairwise + closed-form: for each pair of coincident tracks it solves a
Symbolic-Toolbox quadratic (`setSolZ.m`) for the z where the tracks are
separated by the probe-frame distance `d`, picks the root nearest a prior guess,
then bootstraps medians. `monrad-py` does a global weighted least-squares fit of
all four pose parameters at once.

Notable convergence: MATLAB's quadratic exists *because* a rigid transform
preserves pairwise distances — which is exactly the identity behind our
`filter_rigidity` / `--max-rigidity-resid-mm` gate. We arrived at MATLAB's core
equation independently and demoted it to a quality cut.

Also: MATLAB uses **centered** detector coordinates (probe edges `[-20, 20]`;
telescope `[-50, 50]`), `monrad-py` uses edge-origin `(ch+0.5)*10` → `[0, 1000]`.
Constant half-size offset when comparing `t_x`/`t_y`.

## Suggested skills

- **`/astral:uv`** — all Python invocation goes through `uv run` in this repo.
- **`/verify`** — if you change `_CHI2_TRACK`, `min_anchor_planes`, or any gate,
  drive the real pipeline and observe yield rather than trusting tests. The
  whole point of this comparison is that unit tests would not have surfaced a
  5x yield gap.
- **`/astral:ruff`** — lint/format before any commit (`uv run ruff check --fix .`).
- **`/code-review`** — before committing any cut change; these are subtle and
  interact.
- **`/llm-council`** — genuinely warranted for step 1 ("should we loosen
  `_CHI2_TRACK`, and to what?"). It is a real decision with stakes and
  competing physics arguments on both sides, not a lookup.

Note the repo convention in `CLAUDE.md`: `DESIGN.md` is the authoritative
algorithm reference, but **when code and `DESIGN.md` disagree, the code wins**.
Also memory `feedback_branch_before_commit`: branch before committing if on
`main`.
