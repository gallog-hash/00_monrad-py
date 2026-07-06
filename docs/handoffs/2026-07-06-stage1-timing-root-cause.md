# Handoff: chase the stage-1 relative-timing root cause of the testLab 20210723 z_p anomaly

Written 2026-07-06. Branch `feat/probe-monitoring`. **Working tree is clean** — the
`--max-track-slope` / `wide_track_cut` gate built earlier in this thread was
implemented, validated on real data, proven to be the wrong lever, and **reverted**
(it is not in the code; do not re-implement it — see
`memory/wide-track-cut-gate-shipped.md`).

Continues the testLab 20210723 anomaly thread. **Do not re-derive the diagnosis** —
it is fully captured in project memory
[`testlab-20210723-anomaly-root-cause.md`](../../../.claude/projects/-home-gallog-00-work-new-00-monrad-new-00-monrad-py/memory/testlab-20210723-anomaly-root-cause.md)
(indexed in `MEMORY.md`). Read that first. This doc is only the next step: find *why*
the relative timing is unstable.

## The established result (one paragraph — proven, do not re-litigate)

The two 5-min z_p excursions (17:16 & 18:11 UTC, worst of a broader ~17:00–18:25
instability) are **cross-detector coincidence MISPAIRING**: a genuine telescope
track paired with a genuine-but-spatially-unrelated probe hit. Four independent
tests established this: (1) time-resolved stage-4 alignment is clean in the burst
bins → not an alignment shift, not telescope-track degradation; (2) the z_implied
test (where each angled telescope track actually crosses its probe hit) scatters to
random/negative z in bursts (median −487/−257 mm) vs tight at the true plane in
clean windows (~840 mm, 85–93% within 150 mm) → the paired tel track and prb hit are
NOT the same particle; (3) probe and telescope hits are individually genuine (full
active-area spread, clean geometry) — only the pairing is wrong; (4) **per-window
Δt = t_tel − t_prb shows the relative clock is unstable**: the genuine near-zero-Δt
fraction (|Δt|<50 ns) collapses 0.97 (clean) → 0.08/0.29 (burst) and the Δt median
swings ±120 ns across 5-min bins, while staying tight (std ~20–33 ns, not the flat
±200 ns of uniform random accidentals). z_p is carried almost entirely by angled
tracks, which is why the |b| gate (which strips those) made it worse.

## The task

**Isolate which detector's clock drifts, and why.** The relative tel↔prb timing
wanders by ~±120 ns on a 5-min cadence during ~17:00–18:25 UTC. Each detector's
*own* timing looks self-consistent (singles rates flat), but the reconstructed
**absolute** time of one (or both) drifts relative to the other enough that genuine
same-particle pairs fall to the edge of / outside the 200 ns coincidence window and
the surviving coincidences are dominated by mispairings. Find the mechanism in
stage 1 (GPS/PPS anchoring, oscillator `f_local`, or file-boundary re-anchoring).

## Concrete plan

Stage 1 lives in `src/monrad/timing/reconstruct.py` (`reconstruct_stream`,
`load_header_params`, `find_file_pairs`) and the decoders in
`src/monrad/decoders/` (`gps.py` `GPSDecoder`, `header.py` `decode_ubx_tm2`). Key
invariant (CLAUDE.md): `t_ns` uses `f_local = (C_{k+1} − C_k)/Δsec` — the *measured*
PPS interval — to absorb oscillator drift. A misbehaving PPS chain corrupts exactly
this.

1. **Per-detector PPS / f_local stability over the anomaly hour.** For telescope and
   probe *separately*, walk the PPS counters `C_k` from the `*_GPS.bin` stream and
   plot/tabulate the measured `f_local` per PPS interval and the inter-PPS Δsec.
   Look for jumps, missed PPS (Δsec ≠ 1 s), or duplicated PPS in ~17:00–18:25.
   Whichever detector shows PPS irregularity there is the culprit.
2. **Diff clean vs burst raw GPS.** `monrad-decode-gps` on the burst files vs
   neighbours for both detectors (burst files below). Compare PPS interval
   regularity, satellite/fix-quality flags, and the UBX-TM2 timestamps. A GPS
   holdover / poor-lock episode on one unit is the leading candidate.
3. **Check file-boundary re-anchoring.** The Δt swing has a ~5-min cadence = the
   5-min file rotation. Verify how `reconstruct_stream` carries the PPS chain and
   `utc0`/`f0` across file boundaries: if the burst-region files re-anchor absolute
   time slightly differently (e.g. a per-file `utc0` rounding or a PPS-continuity
   break at the stitch), that produces exactly a per-file relative offset. NOTE the
   benign ±1–2 GPS-vs-position-block stitching (present run-wide, nets to zero across
   consecutive files) was already ruled out as the cause — but the two burst files
   showed diff **2** (vs diff 1 elsewhere), so re-check whether the mid-block stitch
   in *those* files mis-assigns event times.
4. **Confirm by correction.** Once the offending detector + mechanism is found, apply
   the corrected per-window relative offset (or fix the anchoring) and re-run the
   monitor over 17:00–18:30; the burst z_p should recover to ~840 mm and the
   per-window Δt median should return to ~0.

**Downstream mitigation (separate from the root cause, do not conflate):** the right
per-coincidence filter for this failure mode is a **geometric-consistency gate**
(z_implied plausibility / telescope-line-to-probe-hit 4-point residual at a physical
z_p), NOT |b| and NOT only the blunt `--max-resid-rms` window drop. Ship that only if
asked; the goal of *this* task is the timing fix.

## Validation

- Reproduce the per-window Δt table (script below) and confirm the ±120 ns swing and
  the genuine-fraction collapse in the burst bins. That is the signal to explain.
- After a candidate fix: burst z_p → ~840 mm, Δt median → ~0, genuine fraction → ~0.9.

## Data, config, file→UTC mapping

- Data: `data/0_testLab_20210723/{Base,Probe_0}` (Base = telescope, Probe_0 = probe),
  297 files each, 5-min rotation.
- `--z-tel 0 -1340 -670` (testLab plane z-order; see `MEMORY.md`), `--min-anchor-planes 1`.
- **File→UTC:** local filename = UTC + 1h59m40s (CEST). Burst files:
  telescope `Base/20210723_191534*` (17:16 UTC) & `Base/20210723_201033*` (18:11 UTC),
  with the matching `Probe_0/` pairs.
- Full monitor run (either baseline): `uv run monrad-monitor --telescope
  data/0_testLab_20210723/Base --probe data/0_testLab_20210723/Probe_0 --z-tel 0 -1340
  -670 --min-anchor-planes 1 --window-s 300 --out <dir> --no-plots`. Each full run does
  the whole-acquisition alignment pass + coincidence stream (~1–2 min); run heavy
  scripts in the background. Use absolute paths (an earlier `cd` into the data dir
  bit this session).

## Artifacts — diagnostic scripts (session scratchpad is gone; reproduce from here)

All read the real data directly. `dump_coincs.py` writes `$SP/coincs.npz` (per-coinc
dump: t_ns, |b|, a_x,b_x,a_y,b_y, u,v, sigma_prb_x/y, cov_ab_x[3], cov_ab_y[3]) which
`zimplied.py` consumes (set `SP` env var). `dt_test.py` is the key timing tool.

### `dt_test.py` — per-window Δt = t_tel − t_prb (THE timing signal)

```python
import numpy as np, math, os
from datetime import datetime, timezone
from pathlib import Path
from monrad.monitor.io import load_detector
from monrad.timing import reconstruct_stream
from monrad.coincidence import coincidence_stream

ROOT=Path("/home/gallog/00_work_new/00_monrad_new/00_monrad-py")
tel=load_detector(ROOT/"data/0_testLab_20210723/Base")
prb=load_detector(ROOT/"data/0_testLab_20210723/Probe_0")
def ns(h,m,s): return int(datetime(2021,7,23,h,m,s,tzinfo=timezone.utc).timestamp()*1e9)
RANGES=[(ns(17,0,0),ns(17,35,0)),(ns(17,55,0),ns(18,30,0))]
def inr(t): return any(a<=t<b for a,b in RANGES)
BIN=300*10**9
recs={}
ts=reconstruct_stream(tel.gps_paths,tel.pos_paths,tel.utc0,tel.f0)
ps=reconstruct_stream(prb.gps_paths,prb.pos_paths,prb.utc0,prb.f0)
for cl in coincidence_stream([ts,ps], detector_ids=[0,1]):
    tel_t=[ev.t_ns for d,ev,r in cl if d==0]
    prb_t=[ev.t_ns for d,ev,r in cl if d==1]
    if len(tel_t)!=1 or len(prb_t)!=1: continue
    tt=tel_t[0]
    if not inr(tt): continue
    recs.setdefault((tt//BIN)*BIN,[]).append(tt-prb_t[0])
def utc(b): return datetime.fromtimestamp(b/1e9,tz=timezone.utc).strftime("%H:%M")
print("bin(UTC)   n    dt_median  dt_std   frac|dt|<50")
for b in sorted(recs):
    d=np.array(recs[b]); n=len(d); near=np.sum(np.abs(d)<50)
    print("%s  %4d  %+8.0f  %8.0f  %.2f"%(utc(b),n,np.median(d),np.std(d),near/n))
```

### `dump_coincs.py` — per-coincidence dump → `$SP/coincs.npz`

```python
import numpy as np, math, os
from pathlib import Path
from monrad.monitor.io import load_detector, fit_alignment, stream_coincidences
ROOT = Path("/home/gallog/00_work_new/00_monrad_new/00_monrad-py")
tel = load_detector(ROOT/"data/0_testLab_20210723/Base")
prb = load_detector(ROOT/"data/0_testLab_20210723/Probe_0")
z_tel = np.array([0.0, -1340.0, -670.0])
align, _ = fit_alignment(tel, z_tel)
rows = []
for co in stream_coincidences(tel, prb, z_tel=z_tel, alignment=align, min_anchor_planes=1):
    rows.append((co.t_ns, math.hypot(co.b_x, co.b_y), co.a_x, co.b_x, co.a_y, co.b_y,
                 co.u, co.v, co.sigma_prb_x, co.sigma_prb_y, *co.cov_ab_x, *co.cov_ab_y))
arr = np.array(rows)
np.savez(Path(os.environ["SP"])/"coincs.npz", rows=arr)
print("dumped", len(arr), "coincidences")
```

### `zimplied.py` — spatial-correlation test (needs coincs.npz; reference/validation)

```python
import numpy as np, math, os
from datetime import datetime, timezone
from pathlib import Path
from monrad.pose import Coincidence, fit_probe_pose
from monrad.alignment import AlignmentCorrection
SP=Path(os.environ["SP"]); arr=np.load(SP/"coincs.npz")["rows"]
z_tel=np.array([0.0,-1340.0,-670.0]); ident=AlignmentCorrection.identity()
def ns(h,m,s): return int(datetime(2021,7,23,h,m,s,tzinfo=timezone.utc).timestamp()*1e9)
def mk(r): return Coincidence(a_x=r[2],b_x=r[3],a_y=r[4],b_y=r[5],cov_ab_x=(r[10],r[11],r[12]),
    cov_ab_y=(r[13],r[14],r[15]),u=r[6],v=r[7],sigma_prb_x=r[8],sigma_prb_y=r[9],t_ns=int(r[0]))
t=arr[:,0]
def win(a,b): return arr[(t>=a)&(t<b)]
pr=fit_probe_pose([mk(r) for r in win(ns(18,16,14),ns(18,21,0))], z_tel, ident)  # clean ref
tx,ty,th,zp=pr.t_x,pr.t_y,pr.theta,pr.z_p; c,s=math.cos(th),math.sin(th)
def z_implied(rows, bfloor=0.15):
    out=[]
    for r in rows:
        ax,bx,ay,by,u,v=r[2],r[3],r[4],r[5],r[6],r[7]; b2=bx*bx+by*by
        if math.sqrt(b2)<bfloor: continue
        px=tx+u*c-v*s; py=ty+u*s+v*c
        out.append((bx*(px-ax)+by*(py-ay))/b2)
    return np.array(out)
for lbl,a,b in [("18:11 BURST",ns(18,11,6),ns(18,16,14)),("18:16 clean",ns(18,16,14),ns(18,21,0)),
                ("18:00 clean",ns(18,0,43),ns(18,6,1)),("17:14 BURST",ns(17,14,31),ns(17,19,38))]:
    zi=z_implied(win(a,b)); fn=np.mean(np.abs(zi-zp)<150) if len(zi) else 0
    print("%-12s n=%3d z_implied med=%7.0f frac_near=%.2f"%(lbl,len(zi),np.median(zi),fn))
```

There is also a `tres_align.py` (time-resolved stage-4 alignment) that already proved
the telescope geometry is clean in the bursts — reconstruct it from
`fit_telescope_alignment(hits_per_5min_bin, z)` if you need to re-confirm; it is not
needed to chase timing.

## Suggested skills for the next session

- **`/verify`** — after a candidate timing fix, re-run `monrad-monitor` and confirm
  the burst z_p recovers and per-window Δt re-centres.
- **`astral:ruff`** / **`astral:ty`** — if the fix touches `src/monrad/timing/` or
  `src/monrad/decoders/`.
- **`/code-review`** (medium) on any stage-1 change before committing.
