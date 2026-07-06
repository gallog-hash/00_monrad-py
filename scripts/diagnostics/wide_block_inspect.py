"""
Is a wide-angle burst coincidence sitting on a DOUBLE-OCCUPANCY telescope block?

For the 38 wide (|b|>0.5) coincidences and a matched sample of narrow ones, map
the coincidence's telescope time back to its raw telescope 16-row block and
measure the per-plane candidate multiplicity (reconstruct_plane_candidates) and
OR popcount.  A ghost wide track built by the stage-5 combinatorial finder needs
extra candidates -> a genuine second hit (double occupancy) shows up as more
candidates / higher popcount than a normal narrow coincidence.
"""

import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from monrad.monitor.io import load_detector  # noqa: E402
from monrad.timing import reconstruct_stream  # noqa: E402
from monrad.reconstruction import reconstruct_plane_candidates  # noqa: E402
from monrad.reconstruction.hit import _read_block  # noqa: E402
from monrad.decoders.position import POS_COORD_MASK, POS_X_SHIFT  # noqa: E402

S = os.environ.get("MONRAD_DIAG_OUT", ".").rstrip("/") + "/"
d = np.load(S + "coinc_dt.npz")
t_ns, b = d["t_ns"], d["b"]
wide_t = set(int(x) for x in t_ns[b > 0.5])
rng = np.random.default_rng(0)
narrow_all = t_ns[b <= 0.5]
narrow_t = set(int(x) for x in rng.choice(narrow_all, 400, replace=False))
targets = wide_t | narrow_t
print(f"wide={len(wide_t)} narrow_sample={len(narrow_t)}", file=sys.stderr)

tel = load_detector(Path("data/0_testLab_20210723/Base"))
refs = {}
for ev, ref in reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0):
    if ev.t_ns in targets:
        refs[ev.t_ns] = ref
print(f"matched {len(refs)}/{len(targets)}", file=sys.stderr)


def block_stats(ref):
    cands = reconstruct_plane_candidates(ref, tel.pos_paths, n_cols=3)
    ncand = [len(c) for c in cands]  # candidates per plane
    words = _read_block(tel.pos_paths, ref, 3)
    ored = [0, 0, 0]
    for r in range(16):
        for c in range(3):
            ored[c] |= words[r * 3 + c]
    pc = [
        int((o & POS_COORD_MASK)).bit_count()
        + int(((o >> POS_X_SHIFT) & POS_COORD_MASK)).bit_count()
        for o in ored
    ]
    return ncand, pc


def summarize(times, label):
    tot_c, max_c, tot_pc, nmulti = [], [], [], []
    for t in times:
        if t not in refs:
            continue
        ncand, pc = block_stats(refs[t])
        tot_c.append(sum(ncand))
        max_c.append(max(ncand))
        tot_pc.append(sum(pc))
        nmulti.append(sum(1 for c in ncand if c >= 2))  # planes with >=2 cands
    tot_c, max_c, tot_pc, nmulti = map(np.array, (tot_c, max_c, tot_pc, nmulti))
    print(f"\n{label}  (n={len(tot_c)})")
    print(
        f"  sum candidates/3planes : mean={tot_c.mean():.2f} med={np.median(tot_c):.0f} p90={np.percentile(tot_c, 90):.0f} max={tot_c.max()}"
    )
    print(
        f"  max plane candidates   : mean={max_c.mean():.2f} med={np.median(max_c):.0f} max={max_c.max()}"
    )
    print(
        f"  planes with >=2 cands  : mean={nmulti.mean():.2f}  frac>=1multi={np.mean(nmulti >= 1):.2f}"
    )
    print(
        f"  sum OR popcount/3planes: mean={tot_pc.mean():.1f} med={np.median(tot_pc):.0f} p90={np.percentile(tot_pc, 90):.0f}"
    )


summarize(sorted(wide_t), "WIDE (|b|>0.5) coincidences")
summarize(sorted(narrow_t), "NARROW sample")
