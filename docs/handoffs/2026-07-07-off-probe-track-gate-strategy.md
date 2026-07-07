# Strategy: two geometric gates to reject telescope tracks that don't point at the probe

Written 2026-07-07. Branch `feat/probe-monitoring`. **This is a strategy/plan
doc for the next (execution) session — no source was changed this session.**
Closes out the testLab 20210723 z_p burst thread with a concrete, buildable fix.

Ship order decided by the user: **rigidity gate FIRST, footprint gate SECOND.**
Both gates ship; rigidity is the primary always-on gate, footprint is the
anchored cross-check.

## Read these first (authoritative — do not re-derive)

- `memory/testlab-20210723-anomaly-root-cause.md` — the anomaly is two 5-min bins
  (17:15 & 18:10 UTC) of time-coincident but spatially-inconsistent wide-angle
  coincidences. Timing REFUTED; |b|-slope gate REFUTED and reverted.
- `memory/wide-track-cut-gate-shipped.md` — the `--max-track-slope` gate was the
  wrong lever (tightening |b| strips z_p leverage) and is NOT in the code. The
  gates below reject on **where a track lands**, not on its slope — see §"Why
  this preserves z_p leverage".
- `memory/testlab-20210723-anomaly-no-raw-telescope-signature.md`,
  `memory/testlab-20210723-plane-z-order.md` (`--z-tel 0 -1340 -670`),
  `memory/monitor-window-rate.md`.
- Prior handoff (the diagnosis this builds on):
  `docs/handoffs/2026-07-07-burst-track-category-analysis.md`.

## The mechanism being fixed (one paragraph)

In the two bad windows, an excess of telescope events whose three planes are hit
by *different* particles produces spurious wide-angle 2-plane-resolved "tracks"
(a line through 2 points always fits at low χ²). These get time-matched inside
the 200 ns window to unrelated probe hits. The result is geometrically-
inconsistent coincidences: the telescope track, extrapolated to the probe plane,
lands hundreds-to->1000 mm outside the probe (e.g. (−389,644), (1017,185),
(−74,1431)), while the decoded probe hit sits normally inside the active area.
These drag the pose fit — z_p especially — to nonsense. Both gates below detect
exactly this geometric inconsistency and reject the offending coincidences
*before* the pose fit.

## Key correction vs earlier drafts: probe size is DATA-DERIVED

CLAUDE.md's "probe 30 cm × 30 cm" is stale for this data. The probe active
extent is `L = n_probe_ch × 10 mm` (channel→coord is `(ch + 0.5) × 10 mm`).
For testLab 20210723 the probe is **40 channels → L = 400 mm** (400×400 mm²).
The footprint gate must read `L` from the decoded probe channel count
(`n_probe_ch`, already carried by `monitor_probe`), NOT hardcode 300 or 400.

---

## Gate 1 (SHIP FIRST) — Rigidity gate (pose-free pairwise-distance invariant)

### Principle

Probe hits `(u,v)` and telescope-track projections onto the probe plane `(X,Y)`
are the *same physical points* in two frames related by a rigid transform
`(X,Y) = t + R(θ)(u,v)`. A rigid transform **preserves distances**, so for any
two coincidences i, j:

```
D_probe(i,j) = hypot(u_i − u_j, v_i − v_j)          # z-independent, from decoded probe hits
D_track(i,j) = hypot(X_i − X_j, Y_i − Y_j)          # from track projections at z_ref
    where X_i = a_x,i + b_x,i · z_ref ,  Y_i = a_y,i + b_y,i · z_ref
```

For genuine coincidences `D_track ≈ D_probe`. A cross-particle track lands far
off (X,Y huge) while its probe hit sits normally → `D_track ≫ D_probe` for every
pair involving it → the invariant breaks exactly on the contamination.

### Why this is the primary gate

- **Pose-free.** Needs no `t_x, t_y, θ`. Only needs `z_ref` (stable, ~840 mm,
  roughly known). Solves the **cold-start** window (window 0 has no previous
  accepted pose for the footprint gate; rigidity needs no reference at all).
- **Self-contained** internal consistency check on the window's own coincidences.
- **Works in a fully-contaminated sub-window** (post-step: 0 good / 5–9 bad):
  the probe hits are still genuine, so *their* mutual distances are the ground
  truth and the scattered tracks fail to reproduce them → every pair flags.
- **It is the invariant z_p is fit from** — removing rigidity-violators before
  the fit is the cleanest route to recovering z_p.

### The two subtleties (design around these)

1. **Relative, not absolute** — a bad *pair* doesn't say which of the two is
   wrong. Use ≥3 to localize by voting: per coincidence i, score it by the
   **median over the other coincidences j of `|D_track(i,j) − D_probe(i,j)|`**.
   Genuine tracks score ~0; contaminants score large (hundreds of mm). Reject
   coincidences whose median pairwise distance-residual exceeds
   `--max-rigidity-resid-mm`. (Consecutive-neighbour pairs give an O(N) running
   flag; the full median-over-all is O(N²) per window but N≈30–500, fine for a
   pre-filter. Start with median-over-all for correctness; optimise later only
   if profiling says so.)
2. **Mild `z_ref` sensitivity.** `D_track` depends on `z_ref` through
   `|b_i − b_j|·Δz`. Good tracks (small slopes) move ~10–25 mm for a ±tens-of-mm
   `z_ref` error — negligible vs the 400 mm scale — while bad tracks mismatch by
   hundreds regardless. Robust for `z_ref` within tens of mm of true z_p. Use
   the previous accepted window's `z_p` as `z_ref`; for window 0 use the neutral
   `mean(tel_z)` seed the fit already uses, or a `--z-ref-seed`-free default of
   the mid-plane — the gate tolerates being off by tens of mm.

### Implementation sketch (Gate 1)

- New pure function in `src/monrad/pose/optimize.py` (unit-testable in isolation,
  keeps `fit_probe_pose` signature stable):
  ```
  def filter_rigidity(
      coincs: list[Coincidence], z_ref: float, max_resid_mm: float,
  ) -> tuple[list[Coincidence], list[Coincidence]]:
      """Return (kept, dropped). Drop any coincidence whose median pairwise
      |D_track − D_probe| (D_track evaluated at z_ref) exceeds max_resid_mm."""
  ```
  D_track uses `co.a_x + co.b_x*z_ref`, `co.a_y + co.b_y*z_ref`; D_probe uses
  `co.u, co.v`. Guard N<3 (no-op, return all). Never drop below the 3-coinc
  floor `fit_probe_pose` requires.
- Wire into `monrad/monitor/timeseries.py::monitor_probe._emit`: apply
  `filter_rigidity` to the window's `coincs` **before** `fit_probe_pose`, using
  `z_ref =` previous accepted `WindowResult.z_p` (fall back to `mean(z_corr)` for
  the first window). Expose CLI `--max-rigidity-resid-mm` (default `None` = off,
  matching the opt-in philosophy of `--max-resid-rms` / `--max-abs-resid`).
- The same helper should also be applied in `PoseFitter` for the streaming
  pipeline path if/when the user wants it there — but the monitor is the
  deliverable; do NOT change `PoseFitter`'s default behaviour.

---

## Gate 2 (SHIP SECOND) — Footprint gate (absolute, anchored to previous pose)

### Principle

The probe is a physical `L×L` plane at `z≈z_ref`, centred at `(t_x,t_y)`, rotated
by `θ`. Extrapolate the track to the probe plane and map into probe coordinates:

```
X = a_x + b_x·z_ref ;  Y = a_y + b_y·z_ref
[u_pred; v_pred] = R(−θ) · [X − t_x ; Y − t_y]
```

Reject when `(u_pred, v_pred)` falls more than `--max-off-probe-mm` outside the
`[0, L]²` footprint (on-probe projections have outside-distance 0). `L` is
data-derived (`n_probe_ch × 10`). In the burst, wild projections land at
`u_pred/v_pred ≈ −389 … 1431` — 90 to >1000 mm outside `[0,400]`; a genuine
coincidence lands inside by construction. Huge separation → not
threshold-sensitive.

### Reference pose = previous accepted window's pose

The probe moves slowly; the last accepted `WindowResult` pose `(t_x,t_y,θ,z_p)`
is the footprint anchor for the current window. Generous `--max-off-probe-mm`
absorbs real inter-window motion. **No `--ref-pose` CLI** (user decision).
Window 0 has no predecessor → footprint gate is skipped for window 0 and the
rigidity gate alone guards it.

### Why the footprint gate is still needed on top of rigidity

Rigidity is relative: a whole window that drifts *rigidly but wrongly* (all
tracks mutually consistent yet collectively off the probe) passes rigidity. The
absolute footprint anchor is the only thing that catches that. Rare, but cheap
insurance once a reference pose exists.

### Implementation sketch (Gate 2)

- New pure function in `optimize.py`:
  ```
  def filter_off_probe(
      coincs, ref_pose: PoseResult, probe_size_mm: float, max_off_probe_mm: float,
  ) -> tuple[list[Coincidence], list[Coincidence]]:
  ```
  Uses `ref_pose.t_x/t_y/theta/z_p`, inverse-rotates the track projection into
  probe frame, computes signed outside-distance from `[0, probe_size_mm]²`.
- `monitor_probe`: after `filter_rigidity`, apply `filter_off_probe` with the
  previous accepted pose and `probe_size_mm = n_probe_ch * 10`. Skip on window 0.
  CLI `--max-off-probe-mm` (default `None` = off).

---

## Why BOTH gates preserve z_p leverage (retires the |b| reversion concern)

z_p leverage comes from the spread of track slopes `b` among **genuine** tracks
(steep on-probe tracks pin z_p). The reverted `--max-track-slope` gate rejected
on `|b|` directly, stripping steep tracks good *and* bad → killed leverage. Both
new gates reject on **where a track lands / whether its geometry is consistent**,
never on slope. A steep track that still hits the probe and matches the ensemble
geometry is kept (keeps its leverage); only tracks that land off-probe or violate
rigidity — carrying a *false* `b` that was dragging z_p — are removed. Opposite
effect to the |b| gate.

## How the two new gates relate to the existing two

| gate | frame | recover/detect | reference | weakness |
|---|---|---|---|---|
| `--max-resid-rms` (shipped) | drops whole window | detect | none | throws good core away |
| `--max-abs-resid` (shipped) | residual vs *this window's own* fit | recover | self (circular) | Mahalanobis-blind to inflated σ_tel; needs 3 passes |
| **`--max-rigidity-resid-mm` (NEW, ship 1st)** | pairwise probe-vs-track distances | recover | none (pose-free) | relative → needs ≥3 |
| **`--max-off-probe-mm` (NEW, ship 2nd)** | track landing vs prev-window footprint | recover | previous accepted pose | needs a prior pose (skip window 0) |

The new gates' advantage over `--max-abs-resid`: they do **not** depend on the
contaminated window's own converged pose, so no multi-pass bootstrap fragility.

## Validation plan (the /verify deliverable)

Config: `data/0_testLab_20210723/{Base=telescope,Probe_0=probe}`,
`z_tel=[0,-1340,-670]`, `min_anchor_planes=1`. testLab probe L = 400 mm
(40 ch). File→UTC in the burst-analysis handoff §"How to reproduce".

1. **Rigidity alone recovers the bursts.** Run `monrad-monitor` over ~17:00–18:30
   with `--max-rigidity-resid-mm` set (no footprint gate); confirm 17:15 & 18:10
   z_p → ~840 mm and every clean bin is a **strict no-op** (0 coincidences
   dropped). Tune the threshold from the whole-run pairwise-residual distribution
   (print it like the existing RMS distribution), not a universal constant.
2. **Cold-start check.** Confirm the rigidity gate flags the burst on **window 0**
   (no previous pose) — the footprint gate cannot, rigidity must.
3. **Footprint anchor adds value.** With both gates on, confirm no regression on
   clean bins and that the footprint gate catches any rigid-but-off-probe window
   rigidity misses (construct/spot-check if one exists).
4. **Step structure.** Within a burst file, `dropped` count ≈ 0 for the first
   ~90 s then jumps after the step — the gates must track the good→bad transition.
5. **Beats `--max-abs-resid` on passes.** Same z_p recovery in one pass where the
   residual cut needed three.

## Suggested skills for the execution session

- `astral:ruff` / `astral:ty` — the changes touch `src/monrad/pose/optimize.py`
  and `src/monrad/monitor/timeseries.py`.
- `/verify` — drive `monrad-monitor` over 17:00–18:30 per the plan above.
- `/code-review` (medium) on the stage-5/monitor change before committing.
- New unit tests: `tests/test_stage5.py` (or a new `test_pose_gates.py`) for
  `filter_rigidity` and `filter_off_probe` in isolation — a hand-built good
  cluster + one injected wide-angle track, asserting the wild one is dropped and
  the clean set is a no-op.

## Anchors in the current code (verified this session)

- `Coincidence` fields: `a_x,b_x,a_y,b_y, cov_ab_x, cov_ab_y, u, v,
  sigma_prb_x, sigma_prb_y, tel_quality, t_ns` — `src/monrad/pose/types.py:20`.
- `fit_probe_pose(..., max_abs_resid_mm=None)` — `src/monrad/pose/optimize.py:246`;
  its in-fit absolute-mm robust stage is the model for a pre-filter's structure.
- `monitor_probe._emit` builds the window pose — `src/monrad/monitor/timeseries.py:187`;
  window loops at :247 (count-based) / :256 (time-window). This is where both
  pre-filters slot in, before the `fit_probe_pose` call at :192.
- CLI args block — `src/monrad/monitor/timeseries.py:373` onward; add
  `--max-rigidity-resid-mm` and `--max-off-probe-mm` alongside `--max-resid-rms`
  (:417) and `--max-abs-resid` (:430).
- `n_probe_ch` param (→ probe size) — `monitor_probe(...)` signature at
  `src/monrad/monitor/timeseries.py:107`.
