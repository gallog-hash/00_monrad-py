---
name: testlab-20210723-anomalous-window-telescope-side
description: "The 17:08–18:26 UTC monitor anomaly in testLab 20210723 is telescope-side track degradation, NOT probe motion"
metadata: 
  node_type: memory
  type: project
  originSessionId: d22efe38-edd0-4566-80e6-3f1e15bec57a
---

In the `monrad-monitor` run over testLab 20210723 (Base telescope, Probe_0,
`--z-tel 0 -1340 -670`, `--min-fit 2000 --min-anchor-planes 0`), one 77-min
window (**2021-07-23 17:08:53–18:26:39 UTC**) showed an apparent probe excursion
(t_x≈188 vs ~180, z_p≈845 vs ~837) with a halved aggregate inlier count (656 vs
~1300).

**Diagnosed as telescope-side, not the probe.** Sub-window (min_fit=300) +
per-span analysis of the `Coincidence` bundle showed:
- Probe-frame hits `u,v` are unchanged (mean AND spread) in the disturbed
  spans — the probe never moved. A real move would shift the `u,v` mean.
- Telescope-track lab intercepts blow up only in the disturbed ~10-min spans
  (17:08–17:20, 18:07–18:18, partly 17:32): intercept scatter doubles (~300 vs
  ~140 mm) and residual RMS jumps to ~300 mm (normal ~10 mm). Intermittent
  telescope-track degradation / mis-association, not beam (raw coincidence rate
  was normal ~0.43/s) and not fit degeneracy (`corr(t_x,z_p)` small everywhere).

**Why:** the aggregate window pooled garbage telescope stretches with normal
ones; the global pose fit landed at a spurious pose with a halved inlier count.

Mechanism pinned down: singles rates (TEL ~33 Hz, PRB ~29 Hz), stage-1 timing
quality (0% DEGRADED/UNTRUSTED), and per-plane candidate multiplicity are ALL
flat across the disturbed vs normal spans — no plane fault, no rate spike, no
timing hiccup. The corruption is an intermittent **excess of wide-angle track
coincidences** (|b| p99 ~0.68 vs ~0.44 normal) in two ~11-min bursts (17:08,
18:07), ~50 min apart — likely beam halo / shower bursts. Wild fraction (resid
>100 mm vs a fixed reference pose) ~doubles to ~50% (baseline ~15–28%).

**Recovery is viable:** refitting a disturbed span on only its good-core
coincidences (resid <20 mm, ~113 of 300) recovers a NORMAL z_p (839–841 mm) —
the probe never moved.

**How to apply:** don't re-investigate this window as probe motion.

**SHIPPED (2026-07-03): the residual-RMS window gate in `monrad-monitor`**
(`--max-resid-rms`, `resid_rms` col; `_window_resid_rms` in
`src/monrad/monitor/timeseries.py`). Building it corrected several signal
claims from the original investigation — the "~30× / 30–80 mm / inlier-fraction-
useless" numbers below were measured on a DIFFERENT quantity (per-coincidence
residual vs a FIXED reference pose, over ALL coincidences, at min_fit=300 SUB-
spans) and do NOT hold for the shipped gate at full 77-min / min_fit=2000
windows. What the live gate actually shows on this dataset:
- **Inlier-only residual RMS (what `PoseResult.residuals_x/y` expose) is FLAT
  ~14 mm across ALL 9 windows — NO separation.** Those residuals are post-
  Mahalanobis; the cut rejects the wild tracks (bad window n_inliers 656 vs
  ~1300, i.e. it cuts ~2× more), so the surviving core is clean everywhere.
- The gate therefore computes RMS over **all coincidences fed to the fit
  (inliers + Mahalanobis-cut outliers) vs the FITTED pose** (`_window_resid_rms`).
  Result: anomaly **281.6 mm vs a ~132–174 mm baseline** → only **~1.6× margin,
  NOT 30×**. Every window's RMS is ~150 mm (dominated by the ever-present
  ~15–28% wild-track baseline the fit correctly ignores), so the handoff's
  50 mm threshold would drop ALL windows. **Correct threshold for THIS setup
  ≈ 220 mm** (drops only 17:08, keeps the other 8 — verified). Tune per-setup
  from the whole-run RMS distribution the run prints; there is no universal mm.
- **Inlier FRACTION actually separates BETTER here** (656/2000=33% vs 60–72%,
  ~2×), contradicting the memory's earlier "not usable" — that claim held only
  at sub-span granularity (a wild-ONLY span keeps 300/300). Considered adding a
  min-inlier-fraction gate; deferred pending user decision (they were away).
- Follow-up to RECOVER instead of drop (still open, NEXT session): add an
  absolute-mm residual rejection to the stage-5 robust step (`fit_probe_pose`,
  the Mahalanobis d>4 cut + refit ~line 259) — worth it when the probe is far
  and good coincidences are scarce; touches the core fitter shared by stages
  4/5 → needs regression coverage.
Related: [[monitor-window-rate]], [[testlab-20210723-plane-z-order]].
