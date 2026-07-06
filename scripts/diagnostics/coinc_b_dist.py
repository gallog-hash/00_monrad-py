"""
Coincidence-level telescope-track slope |b| distribution across the run.

Streams the full pipeline (Base tel + Probe_0, stage-4 alignment, stage-5
coincidence decode) and, per surviving Coincidence, records the telescope track
slope |b| = hypot(b_x, b_y) and lab intercepts (a_x, a_y) with its UTC time.
Buckets into fixed UTC windows and reports the |b| distribution so we can see
whether the wide-angle excess appears in the 17:08 / 18:07 UTC bursts — the
quantity prior work measured (p99 ~0.68 vs ~0.44 baseline).
"""

import os
import sys
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
OUT = os.environ.get("MONRAD_DIAG_OUT", ".")
from monrad.monitor.io import load_detector, fit_alignment, stream_coincidences  # noqa: E402

Z_TEL = np.array([0.0, -1340.0, -670.0])
BUCKET_S = 600  # 10-min windows

tel = load_detector(Path("data/0_testLab_20210723/Base"))
prb = load_detector(Path("data/0_testLab_20210723/Probe_0"))
print(f"tel utc0={tel.utc0}  prb utc0={prb.utc0}", file=sys.stderr)

align, qual = fit_alignment(tel, Z_TEL)
print(f"alignment done; tel quality={dict(qual)}", file=sys.stderr)

t_ns = []
b = []
ax = []
ay = []
n = 0
for co in stream_coincidences(
    tel, prb, z_tel=Z_TEL, alignment=align, min_anchor_planes=1
):
    t_ns.append(co.t_ns)
    b.append(math.hypot(co.b_x, co.b_y))
    ax.append(co.a_x)
    ay.append(co.a_y)
    n += 1
    if n % 2000 == 0:
        print(f"  {n} coincidences...", file=sys.stderr)

t_ns = np.array(t_ns, dtype=np.int64)
b = np.array(b)
ax = np.array(ax)
ay = np.array(ay)
np.savez(
    os.path.join(OUT, "coinc_b.npz"),
    t_ns=t_ns,
    b=b,
    ax=ax,
    ay=ay,
)
print(f"\nTotal coincidences: {n}", file=sys.stderr)

# whole-run baseline
print("\n=== WHOLE RUN |b| ===")
print(
    f"n={n}  median={np.median(b):.3f}  p90={np.percentile(b, 90):.3f}  "
    f"p99={np.percentile(b, 99):.3f}  wide(>0.5)frac={np.mean(b > 0.5):.3f}"
)

# bucket into fixed UTC windows
t0 = t_ns.min()
bucket = (t_ns - t0) // int(BUCKET_S * 1e9)
print(f"\n=== per {BUCKET_S // 60}-min UTC window ===")
print(
    f"{'utc_start':>19} {'n':>5} {'b_med':>6} {'b_p90':>6} {'b_p99':>6} "
    f"{'wide_f':>6} {'ax_std':>7} {'ay_std':>7}"
)
for bk in range(bucket.max() + 1):
    m = bucket == bk
    if m.sum() == 0:
        continue
    bb = b[m]
    utc = datetime.fromtimestamp((t0 + bk * int(BUCKET_S * 1e9)) / 1e9, tz=timezone.utc)
    print(
        f"{utc.strftime('%H:%M:%S'):>19} {m.sum():>5} {np.median(bb):>6.3f} "
        f"{np.percentile(bb, 90):>6.3f} {np.percentile(bb, 99):>6.3f} "
        f"{np.mean(bb > 0.5):>6.3f} {ax[m].std():>7.1f} {ay[m].std():>7.1f}"
    )
