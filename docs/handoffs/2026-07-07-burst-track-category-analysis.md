# Handoff: the testLab 20210723 z_p bursts are a wide-angle telescope track-pairing effect, NOT timing and NOT combinatorial fabrication

Written 2026-07-07. Branch `feat/probe-monitoring`. **Working tree clean** (all
work this session was read-only diagnostics in the ephemeral scratchpad; no
source changed). Continues the testLab 20210723 z_p anomaly thread.

## Read these first (do not re-derive — memory is authoritative)

- `memory/testlab-20210723-anomaly-root-cause.md` — **corrected 2026-07-07.** The
  stage-1 timing hypothesis is **REFUTED**: both PPS chains flawless, the ±120 ns
  relative-Δt swing is real but benign and uncorrelated with z_p. The anomaly is
  confined to **exactly two 5-min bins: 17:15 & 18:10 UTC**.
- `memory/testlab-20210723-anomaly-no-raw-telescope-signature.md`,
  `memory/wide-track-cut-gate-shipped.md` (|b| gate tried & reverted — do NOT
  re-implement), `memory/testlab-20210723-plane-z-order.md` (`--z-tel 0 -1340 -670`).
- Prior handoff (timing chase, now closed out): `docs/handoffs/2026-07-06-stage1-timing-root-cause.md`.

## What this session established (the user's question + the mechanism)

**User's question:** split telescope tracks into (a) *golden/recovered* — ≤1
telescope axis is multi-hit (5–6 of the 6 axes = 3 planes × {X,Y} resolve to a
single golden/cluster candidate; the lone ambiguous axis is pinned by the line
through the resolved ones) vs (b) *combinatorial search* — ≥2 axes multi-hit
(the χ² finder searches genuine candidate combinations). Does the split explain
the two bursts?

**Answer: NO — the category split does not discriminate.** Decisive facts:

1. **Category ratio is constant across ALL bins** (~15 % DET / ~85 % COMB), clean
   and bad alike. Bad bins are not enriched in combinatorial tracks. (COMB
   dominates everywhere because this detector's axes are usually
   mirror-fold-ambiguous — baseline, not pathology.)
2. **Both categories break equally in the bad bins.** `z_implied` fraction-near
   (spatial consistency) collapses from ~0.86–1.00 (good) to **0.13–0.33** for
   *both* DET and COMB (17:15: DET 0.25, COMB 0.13; 18:10: DET 0.33, COMB 0.24).
3. **Track χ² stays low (~1–2) in both categories** in the bad bins — the tracks
   are not low-χ² fabrications from ambiguous hits.

**The actual mechanism (characterised this session):**

- **Sharp STEP, not a gradual burst.** In both bad bins, coincidences are all
  spatially-good for the first ~90 s, then switch **abruptly to all-bad and stay
  bad** for the rest of the file (30 s sub-bins after the step: `0 good / 5–9 bad`).
  The accepted-coincidence *rate* is maintained (~0.2/s).
- **Bad-window telescope tracks are WIDE-ANGLE and extrapolate far outside the
  probe.** `track@z_p` lands at e.g. (−389,644), (1017,185), (−74,1431) — hundreds
  to >1000 mm outside the probe's [0,300]² area. The probe plane sits at z≈+834,
  *opposite* the telescope stack (z=0,−670,−1340), so the ~1500 mm lever arm
  amplifies any wide slope and throws the extrapolation off the probe entirely.
- **These wide tracks are 2-plane-resolved + 1-plane-recovered lines connecting
  hits that belong to DIFFERENT particles.** A line through 2 points always fits
  (low χ², "determined"), which is why even golden/recovered tracks are affected.
  The two resolved planes sit at very different x/y in bad tracks (e.g. col0 x=115
  @ z=0 and col1 x=915 @ z=−1340) vs close together in good tracks.
- **Probe side and timing are clean.** Decoded probe (u,v) fill the full active
  area normally in the bad window but are uncorrelated with the track prediction
  (corr≈0, mirror≈0). GEN alignment (GPS-event ↔ position-block) is **perfect
  within every file** (0 element-wise mismatches; the `diff=1/2` warnings are just
  trailing blocks straddling the next file — benign). Δt is unchanged across the
  step (~−90 ns, the ambient benign offset).
- Occupancy (fired bits ~19) and multi-axis multiplicity histograms are flat
  across good/bad — consistent with the "no raw telescope signature" memory.

**One-line root cause:** in these two 5-min windows, an excess of telescope
events whose planes are hit by *different* particles produces spurious wide-angle
2-plane "tracks" that get time-matched (within the 200 ns window) to unrelated
probe hits → geometrically-inconsistent coincidences → z_p dragged to nonsense.
The right fix remains a **per-coincidence geometric-consistency gate** (z_implied
plausibility / track-to-probe-hit residual at a physical z_p), which removes
these regardless of track category or |b|.

## The open question for the next session

**Why does the cross-particle wide-track excess switch on at ~90 s into these two
specific files and persist?** Candidates not yet distinguished:
1. A real influx of correlated multi-particle events (beam halo / shower burst)
   that raises cross-plane different-particle pairing — but raw occupancy is flat,
   which argues against a big cascade.
2. A telescope readout/gate effect for those windows (e.g. a widened acquisition
   gate letting a second particle into the 16-row/80 ns block on a subset of
   planes) — check whether bad-window telescope events more often have exactly one
   plane resolving to a hit that is mutually *inconsistent* with the other two at
   high would-be-χ² before recovery.
3. Verify it's genuinely two isolated windows (scan the whole ~17:00–18:25 span at
   30 s resolution for the same step signature; there may be more sub-windows).

Suggested concrete next steps: (a) build the geometric-consistency gate and
confirm it recovers 17:15/18:10 z_p to ~840 mm while leaving clean bins
untouched (this is the deliverable the thread has been converging on — ship only
if the user asks); (b) to nail the physics, correlate the step time against any
run/beam log the user can provide, and check the *raw* telescope singles rate at
30 s resolution around 17:16:30 and 18:11:30 for a step.

## How to reproduce (scratchpad is ephemeral — scripts must be regenerated)

All scripts read real data directly; run with `uv run python <script>`. Config:
`data/0_testLab_20210723/{Base=telescope,Probe_0=probe}`, `z_tel=[0,-1340,-670]`,
`min_anchor_planes=1`, reference pose from the clean 18:00 UTC 5-min bin.
File→UTC: probe/telescope have *independent* file rotation (Base 17:15 file
`20210723_191534`, Probe_0 17:15 file `20210723_191558`). `utc0` from the header
is naive-UTC (tel 09:36:19, prb 09:36:21).

Key regenerable diagnostics (described so a fresh agent can rebuild them; the
2026-07-06 handoff has `dt_test.py`/`dump_coincs.py`/`zimplied.py`):

- **`pps_walk.py`** — walk every PPS interval per detector; report irregular
  intervals + f_local stability. (Result: 0 irregular, both clean.)
- **`nearest_dt.py`** — unbiased nearest-neighbour Δt (±1 µs, no 200 ns cut) per
  5-min bin. (Result: single tight peak, zero background, rigidly shifts ±120 ns.)
- **`combined.py`** — per-bin z_p (`fit_probe_pose`) + z_implied. (Result:
  anomaly only in 17:15 & 18:10; uncorrelated with the Δt offset.)
- **`burst_tracks.py`** — categorise each accepted coincidence's telescope track
  as DET (≤1 multi axis) vs COMB (≥2) using per-axis candidate counts from
  `_axis_candidates_with_tot`; capture χ² via `PoseFitter(on_decode=…)`; cross
  against z_implied. **This is the script that answers the user's question.**
- **`temporal.py`** — 30 s sub-bin good/bad structure (found the step), plus
  occupancy (fired-bit) and multi-axis histograms.
- **`probe_residual.py`** — decoded vs track-predicted probe (u,v) across the
  step; correlation test.
- **`raw_inspect.py` / `raw_inspect2.py`** — list probe events near bad
  coincidences; search all-bin spatial matches + event-index offsets.
- **`plane_inspect.py`** — raw 3-plane hits of determined bad tracks (showed the
  wide 2-plane lines).
- **`gen_align.py`** — GPS-event-GEN vs position-block-GEN alignment (ruled out
  a PosRef slip).
- **`final_summary.py`** — consolidated per-window × per-category table (|b|,
  in-probe fraction, z_implied). Confirms DET/COMB break equally.

Category helper (the crux of DET vs COMB): for a telescope `PosRef`, read the
16×3 block with `_read_block`, get per-bit counts with `_bit_counts`, OR to
`x_or/y_or`, and count `len(_axis_candidates_with_tot(x_or, x_counts, False)) > 1`
per axis across the 3 planes → `n_multi`; DET if `n_multi < 2`, else COMB.

## Suggested skills for the next session

- **`/verify`** — after building the geometric-consistency gate, re-run
  `monrad-monitor` over 17:00–18:30 and confirm 17:15/18:10 z_p → ~840 mm with
  clean bins untouched.
- **`astral:ruff`** / **`astral:ty`** — if the gate touches `src/monrad/pose/` or
  `src/monrad/monitor/`.
- **`/code-review`** (medium) on any stage-5/monitor change before committing.
