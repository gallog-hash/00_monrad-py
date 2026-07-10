# Handoff: fixed-z_p experiment outcome + next steps

Written 2026-07-06. Continues the anomaly-mitigation thread from
[`2026-07-06-in-fit-abs-mm-recovery-plan.md`](2026-07-06-in-fit-abs-mm-recovery-plan.md)
(the abs-mm gate) and [`2026-07-03-rms-gate-followups.md`](2026-07-03-rms-gate-followups.md).

Repo: `00_monrad-py`. Branch: `feat/probe-monitoring`. **Working tree is clean**
(the experiment code was reverted — see below).

## What this session investigated

Question raised: is there a *better* fix for the testLab 20210723 17:08–18:26 UTC
`z_p` excursion than the shipped `--max-resid-rms` (drops the window) and
`--max-abs-resid` (barely bites; see prior handoff) gates? Two ideas examined:

1. **Four-point fit** (3 telescope + 1 probe, jointly). Analysed, not built.
   Verdict: cleaner *discriminant* for individual wild wide-angle tracks (a
   proper per-coincidence χ² that isn't hidden by the inflated `σ_tel(z_p)`),
   but does **not** fix the testLab anomaly, which is a *coherent* telescope-side
   systematic, not an outlier population. The extra point sits at the unknown
   `z_p`, so it adds no independent leverage on the slope×`z_p` degeneracy, and
   letting the probe inform the track entangles the `z_p` measurement. Worth
   prototyping only as a diagnostic, not as the estimator.

2. **Constrain `z_p`** ("item 1"): since the probe is physically fixed in z, hold
   `z_p` at the run baseline (flavor a) or Gaussian-prior it (flavor b) so
   telescope degradation can't masquerade as a `z_p` move. **This was
   implemented (flavor a) and tested on real data — see result below.**

## Result — constrain-`z_p` is REFUTED as a recovery

Full finding is in project memory:
[`fixed-zp-does-not-recover-anomaly.md`](../../../.claude/projects/-home-gallog-00-work-new-00-monrad-new-00-monrad-py/memory/fixed-zp-does-not-recover-anomaly.md)
(indexed in `MEMORY.md`). Key points, do not re-derive:

- testLab 20210723, `--z-tel 0 -1340 -670`, 30-min windows, default anchor gate.
  Baseline `z_p` (median of free fits) = 842.1 mm; 24 windows; 10668 coincidences.
- Pinning `z_p`=842 on the 17:08 window (free `z_p`=874.9, rms=300) pushed the
  bias into `t_x` (189→196, a +16 mm excursion vs ~180 baseline) and `θ`
  (0.7→1.5°), and **residual rms stayed flat (300→296)** — zero goodness-of-fit
  recovered. Same on 18:08 / 18:38.
- Cause: anomaly windows are data-starved (n_inliers ~85/142/154 vs ~330–410
  clean), so `z_p` is weakly identified and **degenerate with `t_x`**. Fixing
  `z_p` ≈ fixing `t_x`; the prior variant (b) would relocate the bias the same
  way. **The whole constrain-`z_p` family is ruled out as a recovery.**
- Silver lining: the bad windows are trivially *detectable* — rms ~2× and
  n_inliers ~4× separated from baseline.

## Current code / repo state

- **Reverted.** The opt-in `z_p_fixed` path added to
  `src/monrad/pose/optimize.py` this session was `git checkout`-reverted at the
  user's request. `fit_probe_pose` is back to its committed Mahalanobis +
  abs-mm behaviour (commit `64ec238`). `git status` shows no tracked changes.
- Untracked: `pipeline_out/` (experiment output) and `.claude/` — neither is
  code.
- The throwaway experiment script lived in the session scratchpad (not in the
  repo) and is gone; it is straightforward to rebuild from the memory note if
  needed (decode once via `monrad.monitor.io.stream_coincidences`, bucket into
  1800 s / MIN_FIT windows, free vs fixed fit, compare).

## Recommended next steps (in priority order)

1. **Add an `n_inliers`-drop auto-flag to the monitor** as a tuning-free
   complement to `--max-resid-rms`. The anomaly windows collapse to ~85–154
   inliers vs ~330–410 baseline — a sharp, near-setup-independent ratio that
   needs no mm threshold. Flag/drop windows whose inlier count falls below, say,
   a fraction of the running median. Touch points mirror the shipped gates:
   `src/monrad/monitor/timeseries.py` (`monitor_probe` + `_emit` + CLI) and
   `_window_resid_rms`'s neighbourhood.
2. **Window-widening recovery** for genuine z-monitoring across a degraded
   stretch: accumulate more good tracks per window (see project memory
   `monitor-window-rate`) instead of reprojecting bad ones.
3. **Time-resolved telescope alignment / health** — the root cause is
   telescope-side (project memory `testlab-20210723-anomalous-window-telescope-side`).
   Re-aligning the telescope for the bad window (if enough good tracks survive)
   is the principled fix vs dropping it. Larger effort.
4. Do **not** revisit constrain-`z_p` (settled) or invest in the four-point fit
   as an estimator (won't fix a coherent systematic). Four-point χ² is only
   worth it as a per-window telescope-health diagnostic.

## Gotchas (carried forward)

- z-tel order for testLab 20210723 is `0 -1340 -670` (file columns not in z
  order; filenames not UTC). Project memory `testlab-20210723-plane-z-order`.
- The full-run decode with `--min-anchor-planes 1` is the tractable config;
  `min_anchor_planes 0` is ~25 min/config and too slow for iteration.
- `PoseResult.residuals_x/y` are inlier-only; the honest window-quality signal
  is `_window_resid_rms` over inliers+outliers.

## Suggested skills for the next session

- **`/run`** or **`monrad-monitor`** directly — to reproduce the per-window
  timeseries when validating the n_inliers-drop flag on testLab.
- **`astral:ruff`** — lint/format any monitor changes (pre-commit enforces).
- **`/code-review`** (medium) on the diff before committing the new gate.
- **`/verify`** — drive the monitor end-to-end on testLab to confirm the flag
  drops exactly the 17:08 / 18:08 / 18:38 windows and nothing else.
