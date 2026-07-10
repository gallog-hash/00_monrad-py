"""
Raw Probe_0 inspector, bucketed by absolute UTC (apples-to-apples with the
coincidence |b| bursts at 17:16:21 and 18:11:21 UTC).

Streams the probe from file 0 (correct utc0 anchoring), reads each event's
single-plane 16-row block directly, and per event records:
  valid    : OR mask passes BinDecoder._is_valid
  pcx/pcy  : popcount of the 20-bit X/Y OR mask (hit multiplicity)
  quality  : golden / cluster / unresolved / invalid
  u,v      : decoded probe centroid (mm) when resolved
Buckets into 5-min UTC windows and reports rate, occupancy, multiplicity,
quality mix, and u,v mean/scatter, so we can see what changes on the probe side
in the two burst windows.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
OUT = os.environ.get("MONRAD_DIAG_OUT", ".")
from monrad.monitor.io import load_detector  # noqa: E402
from monrad.timing import reconstruct_stream  # noqa: E402
from monrad.reconstruction.hit import _read_block, _decode_axis  # noqa: E402
from monrad.decoders.position import (  # noqa: E402
    POS_COORD_MASK,
    POS_X_SHIFT,
    BinDecoder,
)

STRIP_MM = 10.0
prb = load_detector(Path("data/0_testLab_20210723/Probe_0"))
print(f"prb utc0={prb.utc0}  nfiles={len(prb.pos_paths)}", file=sys.stderr)

rows = []  # (t_ns, valid, pcx, pcy, qcode, u, v)
# qcode: 0 invalid, 1 golden, 2 cluster, 3 unresolved
GOOD = {"golden": 1, "cluster": 2}
n = 0
for ev, ref in reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0):
    words = _read_block(prb.pos_paths, ref, 1)  # 16 words
    ored = 0
    for w in words:
        ored |= w
    y_or = ored & POS_COORD_MASK
    x_or = (ored >> POS_X_SHIFT) & POS_COORD_MASK
    valid, _ = BinDecoder._is_valid(x_or, y_or)
    pcx = int(x_or).bit_count()
    pcy = int(y_or).bit_count()
    u = v = np.nan
    qcode = 0
    if valid:
        cx, _, qx = _decode_axis(x_or)
        cy, _, qy = _decode_axis(y_or)
        if qx == "unresolved" or qy == "unresolved":
            qcode = 3
        else:
            qcode = 1 if (qx == "golden" and qy == "golden") else 2
            u = (cx + 0.5) * STRIP_MM
            v = (cy + 0.5) * STRIP_MM
    rows.append((ev.t_ns, int(valid), pcx, pcy, qcode, u, v))
    n += 1
    if n % 100000 == 0:
        print(f"  {n} events...", file=sys.stderr)

t_ns = np.array([r[0] for r in rows], dtype=np.int64)
valid = np.array([r[1] for r in rows], dtype=np.int8)
pcx = np.array([r[2] for r in rows], dtype=np.int16)
pcy = np.array([r[3] for r in rows], dtype=np.int16)
qcode = np.array([r[4] for r in rows], dtype=np.int8)
u = np.array([r[5] for r in rows])
v = np.array([r[6] for r in rows])
np.savez(
    os.path.join(OUT, "probe_raw.npz"),
    t_ns=t_ns,
    valid=valid,
    pcx=pcx,
    pcy=pcy,
    qcode=qcode,
    u=u,
    v=v,
)
print(f"total probe events: {n}", file=sys.stderr)

# 5-min UTC buckets, print 16:50-18:45 plus whole-run baseline
t0 = t_ns.min()
BS = int(300 * 1e9)
bk = ((t_ns - t0) // BS).astype(int)
lo = datetime(2021, 7, 23, 16, 50, tzinfo=timezone.utc).timestamp() * 1e9
hi = datetime(2021, 7, 23, 18, 45, tzinfo=timezone.utc).timestamp() * 1e9

print("\n=== WHOLE RUN probe baseline ===")
print(
    f"n={n} rate~{n / ((t_ns.max() - t0) / 1e9):.1f}Hz  occ={valid.mean():.3f}  "
    f"pcx_mean={pcx[valid == 1].mean():.2f}  gold_frac={(qcode == 1).mean():.3f} "
    f"clus={(qcode == 2).mean():.3f} unres={(qcode == 3).mean():.3f}"
)

print("\n=== per 5-min UTC window (16:50-18:45) ===")
print(
    f"{'utc':>8} {'n':>5} {'rate':>5} {'occ':>5} {'pcx':>5} {'pcy':>5} "
    f"{'gold':>5} {'clus':>5} {'unres':>5} {'u_mean':>6} {'u_std':>6} "
    f"{'v_mean':>6} {'v_std':>6}"
)
for k in range(bk.max() + 1):
    m = bk == k
    if m.sum() == 0:
        continue
    tc = t0 + k * BS
    if tc < lo or tc > hi:
        continue
    utc = datetime.fromtimestamp(tc / 1e9, tz=timezone.utc).strftime("%H:%M:%S")
    vm = m & (valid == 1)
    res = m & (qcode >= 1) & (qcode <= 2)
    print(
        f"{utc:>8} {m.sum():>5} {m.sum() / 300:>5.1f} {valid[m].mean():>5.3f} "
        f"{pcx[vm].mean():>5.2f} {pcy[vm].mean():>5.2f} "
        f"{(qcode[m] == 1).mean():>5.3f} {(qcode[m] == 2).mean():>5.3f} "
        f"{(qcode[m] == 3).mean():>5.3f} "
        f"{np.nanmean(u[res]):>6.1f} {np.nanstd(u[res]):>6.1f} "
        f"{np.nanmean(v[res]):>6.1f} {np.nanstd(v[res]):>6.1f}"
    )
