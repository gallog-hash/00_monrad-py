# Handoff / plan: in-fit absolute-mm residual recovery (Follow-up 2)

Written 2026-07-06. Execution handoff — intended to be executed in a fresh
session. Continues [`2026-07-03-rms-gate-followups.md`](2026-07-03-rms-gate-followups.md)
Follow-up 2.

Repo: `00_monrad-py`. Branch: `feat/probe-monitoring`.

Decision already made: **Follow-up 1 (`--min-inlier-fraction` gate) is dropped.**
Only this (Follow-up 2) is being pursued.

## Goal

Add an **opt-in** absolute-mm residual rejection to the stage-5 robust step that
**complements** the Mahalanobis cut. Wide-angle "wild" telescope tracks have
inflated `σ_tel(z_p)` (large `var_b·z_p²`), so their Mahalanobis distance stays
small and they survive the `d>4` cut — landing the pose wrong. An absolute-mm cut
catches exactly those. When the probe is far and good coincidences are scarce,
this recovers the good core instead of dropping the whole window (which is what
the shipped `--max-resid-rms` gate does).

## Scope correction to the prior handoff

`fit_probe_pose` is **stage-5 only**. Stage 4 alignment uses
`_tel_line_fit`/`_fit_triple`, NOT `fit_probe_pose` (verified via grep — the only
callers are `PoseFitter`, `monitor/timeseries`, `monitor/resolution`, and tests).
So the prior handoff's "core fitter shared by stages 4 and 5" caution is looser
than stated: **this change does not touch stage 4.** The real regression surface
is stage-5 normal windows carrying the ~15–28% baseline wild fraction the current
fit already handles.

## Design decisions

1. **Opt-in, default off.** New param `max_abs_resid_mm: float | None = None` on
   `fit_probe_pose`. `None` ⇒ byte-for-byte current behaviour (zero regression).
   Mirrors how `--max-resid-rms` shipped. There is no universal mm — tune per
   setup, same philosophy as the window gate.
2. **Layer it as a THIRD robust stage, after the existing Mahalanobis cut +
   refit** (insert between `src/monrad/pose/optimize.py:357` and the "Final
   residuals" block at line 359). Cut against the **post-Mahalanobis-refit** pose
   (best available), not the raw LM pose. Rejected coincidences append to
   `outliers`; `inliers`/`n_inliers`/`cov` are updated so the downstream
   final-residuals and stratified-half blocks and `PoseResult.inliers/outliers`
   stay consistent.
3. **Residual metric:** combined magnitude `hypot(r_x, r_y)` in mm vs a single
   threshold (matches the investigation's `resid < 20 mm` core recovery →
   z_p 839.0/840.7 mm).
4. **Bounded iteration (≤3 passes, early-stop when the mask stabilises).** The
   reference pose in a contaminated window starts landed-wrong; one pass
   recovered z_p in the investigation, but a 2–3 pass loop is barely more code and
   materially more robust. Deterministic. (Single-pass is the minimal fallback if
   we want to match the existing one-pass-refit structure exactly.)
5. **Guards:** only apply when it removes something AND ≥3 survive; otherwise
   no-op (keep the Mahalanobis inliers). Never drop below the 3-coincidence floor.

## Threading (far-probe recovery is the point — wire it to the monitor)

- `fit_probe_pose(..., max_abs_resid_mm=None)` — core, `src/monrad/pose/optimize.py`.
- `PoseFitter.__init__` (`src/monrad/pose/fitter.py:38`) new param → passed at the
  `fit_probe_pose(...)` call, `fitter.py:285`.
- `scripts/run_pipeline.py` — new `--max-abs-resid` CLI flag → `PoseFitter(...)`
  (constructed ~line 570).
- `src/monrad/monitor/timeseries.py` — `monitor_probe(..., max_abs_resid_mm=...)`
  param (~line 111 alongside `max_resid_rms_mm`) + `--max-abs-resid` CLI, threaded
  into the `fit_probe_pose` call at line 183.
- `src/monrad/monitor/resolution.py` — leave at default `None` (resolution study,
  not the recovery path).

## Tests (delicate part — the new cut must NOT over-trim the baseline)

In `tests/test_stage5.py` (+ maybe `tests/test_corner_probe_edge_cases.py`, which
already builds `genuine + accidental`):

- **No-op:** `max_abs_resid_mm=None` → identical `PoseResult` to today (protects
  the default path).
- **Regression / no over-trim:** clean synthetic window with the baseline wild
  fraction; a reasonably-set threshold leaves z_p and n_inliers within noise of
  the ungated fit.
- **Recovery:** heavily contaminated window (wild wide-angle tracks that pass
  Mahalanobis); ungated fit lands z_p wrong, gated fit recovers z_p to truth with
  σ inflated only mildly (~1.3× per the investigation).
- **Edge:** threshold so tight that <3 survive → falls back to Mahalanobis
  inliers, no crash.
- Monitor-level test in `tests/test_monitor_timeseries.py` that the flag plumbs
  through (fast synthetic).

## Tuning aid (nice-to-have, low cost)

Abs-cut-removed coincidences already land in `PoseResult.outliers` — enough to
inspect. Optionally have the monitor print the per-window abs-residual
distribution (like it prints the RMS distribution) so the user can pick the mm
from data.

## Verify

- `uv run pytest tests/test_stage5.py tests/test_corner_probe_edge_cases.py tests/test_monitor_timeseries.py`
- Repro on `data/0_testLab_20210723` (`--z-tel 0 -1340 -670 --min-fit 2000
  --min-anchor-planes 0`): the 17:08–18:26 window's z_p should pull back toward
  the ~837 mm baseline instead of the wild ~845 mm, WITHOUT dropping the window.
  Compare against the shipped `--max-resid-rms 220` gate run.
- Ruff clean (pre-commit enforces).

## Gotchas carried from the prior handoff

- **z-tel order for testLab 20210723 is `0 -1340 -670`** — file columns are NOT in
  z order; filenames are NOT UTC; bucket by decoded `t_ns`.
- **`PoseResult.residuals_x/y` are inlier-only** (post-Mahalanobis).
- **The 17:08 anomaly is telescope-side, not probe motion.** The recovery's job is
  to make the pose fit ignore the wild tracks, not to track a probe move.
- **Absolute-time anchoring**: stream from run start; do not front-slice the file
  list to restrict the interval.
