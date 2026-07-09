"""
Per-coincidence Δt = t_tel - t_prb test — is the wide-angle burst a timing
misalignment (mis-pairing) or a combinatorial-finder artifact?

Replicates stream_coincidences' fitter wiring but iterates coincidence_stream
directly so we can read BOTH event times in each accepted cluster.  Records,
per accepted coincidence: tel time, |b|=hypot(b_x,b_y), Δt (ns), probe u,v.
Then compares Δt for wide (|b|>0.5) vs narrow coincidences, and per-window.
"""

import os
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
OUT = os.environ.get("MONRAD_DIAG_OUT", ".")
from monrad.monitor.io import load_detector, fit_alignment  # noqa: E402
from monrad.timing import reconstruct_stream  # noqa: E402
from monrad.coincidence import coincidence_stream  # noqa: E402
from monrad.pose import PoseFitter  # noqa: E402
from _config import Z_TEL  # noqa: E402

tel = load_detector(Path("data/0_testLab_20210723/Base"))
prb = load_detector(Path("data/0_testLab_20210723/Probe_0"))
align, _ = fit_alignment(tel, Z_TEL)
print("alignment done", file=sys.stderr)

fitter = PoseFitter(
    tel_z=Z_TEL,
    alignment=align,
    tel_id=0,
    prb_id=1,
    tel_pos_paths=tel.pos_paths,
    prb_pos_paths=prb.pos_paths,
    min_anchor_planes=1,
)
tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
prb_stream = reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0)

t_ns, b, dt, uu, vv = [], [], [], [], []
n = 0
for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
    co = fitter.decode_cluster(cluster)
    if co is None:
        continue
    tel_t = [ev.t_ns for did, ev, _ in cluster if did == 0][0]
    prb_t = [ev.t_ns for did, ev, _ in cluster if did == 1][0]
    t_ns.append(co.t_ns)
    b.append(math.hypot(co.b_x, co.b_y))
    dt.append(tel_t - prb_t)
    uu.append(co.u)
    vv.append(co.v)
    n += 1
    if n % 2000 == 0:
        print(f"  {n}...", file=sys.stderr)

t_ns = np.array(t_ns, dtype=np.int64)
b = np.array(b)
dt = np.array(dt, dtype=np.int64)
uu = np.array(uu)
vv = np.array(vv)
np.savez(
    os.path.join(OUT, "coinc_dt.npz"),
    t_ns=t_ns,
    b=b,
    dt=dt,
    uu=uu,
    vv=vv,
)
print(f"total {n}", file=sys.stderr)

wide = b > 0.5
print("\n=== Δt (ns): narrow vs wide, WHOLE RUN ===")
print(
    f"narrow (|b|<=0.5) n={(~wide).sum()}: dt mean={dt[~wide].mean():.1f} "
    f"std={dt[~wide].std():.1f} med={np.median(dt[~wide]):.1f} "
    f"absmed={np.median(np.abs(dt[~wide])):.1f}"
)
print(
    f"wide   (|b|>0.5)  n={wide.sum()}: dt mean={dt[wide].mean():.1f} "
    f"std={dt[wide].std():.1f} med={np.median(dt[wide]):.1f} "
    f"absmed={np.median(np.abs(dt[wide])):.1f}"
)

# per 5-min window: Δt stats + wide fraction, around the bursts
t0 = t_ns.min()
BS = int(300 * 1e9)
bk = ((t_ns - t0) // BS).astype(int)
lo = datetime(2021, 7, 23, 16, 50, tzinfo=timezone.utc).timestamp() * 1e9
hi = datetime(2021, 7, 23, 18, 45, tzinfo=timezone.utc).timestamp() * 1e9
print("\n=== per 5-min window: Δt distribution + |Δt| of wide tracks ===")
print(
    f"{'utc':>8} {'n':>4} {'wide':>4} {'dt_mean':>7} {'dt_std':>7} "
    f"{'|dt|p95':>7} {'wide_dt_absmed':>14} {'narrow_dt_absmed':>16}"
)
for k in range(bk.max() + 1):
    m = bk == k
    if m.sum() == 0:
        continue
    tc = t0 + k * BS
    if tc < lo or tc > hi:
        continue
    utc = datetime.fromtimestamp(tc / 1e9, tz=timezone.utc).strftime("%H:%M:%S")
    dm = dt[m]
    wm = m & wide
    nm = m & ~wide
    wdt = np.median(np.abs(dt[wm])) if wm.sum() else float("nan")
    ndt = np.median(np.abs(dt[nm])) if nm.sum() else float("nan")
    print(
        f"{utc:>8} {m.sum():>4} {wm.sum():>4} {dm.mean():>7.1f} {dm.std():>7.1f} "
        f"{np.percentile(np.abs(dm), 95):>7.1f} {wdt:>14.1f} {ndt:>16.1f}"
    )
