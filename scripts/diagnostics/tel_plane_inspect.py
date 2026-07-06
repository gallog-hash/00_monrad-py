"""
Per-plane raw-occupancy / quality / OR-spread inspector.

For each Base *.bin file, per plane column (0,1,2) measure the raw hit pattern
that a shower/halo burst would perturb:
  occ      : fraction of 16-row blocks in which the plane's OR mask is 'valid'
  pc_x/pc_y: mean popcount of the 20-bit X/Y OR mask over valid blocks
             (>2 means several fibers/ribbons fired -> ambiguous/cluster hit,
              the raw material for mirror-fold wide-angle mis-picks)
  gold     : fraction of valid-plane decodes that are 'golden' (both axes w=1)
  unres    : fraction 'unresolved' (axis failed -> multi-candidate mirror fold)
  m2       : fraction of blocks with >=2 valid planes (track-capable material)
"""

import sys
import struct
import numpy as np

sys.path.insert(0, "src")
from monrad.reconstruction.hit import _decode_axis  # noqa: E402
from monrad.decoders.position import (  # noqa: E402
    POS_COORD_MASK,
    POS_X_SHIFT,
    BinDecoder,
)


def load(path):
    with open(path, "rb") as f:
        raw = f.read()
    n_rows, n_cols = struct.unpack_from("<II", raw, 0)
    w = np.frombuffer(raw, dtype="<u8", count=n_rows * n_cols, offset=8)
    return w.reshape(n_rows // 16, 16, n_cols)


def popcount20(v):
    return int(v).bit_count()


def analyze(path):
    words = load(path)
    nblk = words.shape[0]
    ored = np.bitwise_or.reduce(words, axis=1)  # (nblk,3)
    y_or = (ored & POS_COORD_MASK).astype(np.uint64)
    x_or = ((ored >> POS_X_SHIFT) & POS_COORD_MASK).astype(np.uint64)

    cache = {}

    def dec(v):
        v = int(v)
        r = cache.get(v)
        if r is None:
            _, _, q = _decode_axis(v)
            r = q
            cache[v] = r
        return r

    out = []
    nvalid_any = np.zeros(nblk, dtype=int)
    for col in range(3):
        occ = 0
        pcx = []
        pcy = []
        qcount = {"golden": 0, "cluster": 0, "unresolved": 0}
        for i in range(nblk):
            xv, yv = int(x_or[i, col]), int(y_or[i, col])
            valid, _ = BinDecoder._is_valid(xv, yv)
            if not valid:
                continue
            occ += 1
            nvalid_any[i] += 1
            pcx.append(popcount20(xv))
            pcy.append(popcount20(yv))
            qx, qy = dec(xv), dec(yv)
            if qx == "golden" and qy == "golden":
                qcount["golden"] += 1
            elif qx == "unresolved" or qy == "unresolved":
                qcount["unresolved"] += 1
            else:
                qcount["cluster"] += 1
        nv = max(occ, 1)
        out.append(
            {
                "col": col,
                "occ": occ / nblk,
                "pcx": np.mean(pcx) if pcx else 0,
                "pcy": np.mean(pcy) if pcy else 0,
                "pcx99": np.percentile(pcx, 99) if pcx else 0,
                "gold": qcount["golden"] / nv,
                "unres": qcount["unresolved"] / nv,
            }
        )
    m2 = float((nvalid_any >= 2).mean())
    return path.split("/")[-1][9:15], nblk, out, m2


if __name__ == "__main__":
    print(
        f"{'file':>7} {'nblk':>6} {'m2':>6}   per-plane: occ  pcx  pcy pcx99 gold unres"
    )
    for p in sys.argv[1:]:
        name, nblk, out, m2 = analyze(p)
        line = f"{name:>7} {nblk:>6} {m2:>6.4f}   "
        cells = []
        for o in out:
            cells.append(
                f"c{o['col']}:{o['occ']:.3f}/{o['pcx']:.2f}/{o['pcy']:.2f}/"
                f"{o['pcx99']:.0f}/{o['gold']:.2f}/{o['unres']:.2f}"
            )
        print(line + "  ".join(cells))
