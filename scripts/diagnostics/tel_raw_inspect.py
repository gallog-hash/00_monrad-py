"""
Raw telescope-file inspector for the testLab 20210723 wide-angle anomaly.

Reads a Base *.bin file directly (no timing, no coincidence), decodes all 3
planes for every 16-row block, fits a straight line through the 3 z-planes and
characterises the per-file TRACK-ANGLE (slope) distribution.  This measures the
full telescope self-track population, independent of the probe / pose fit, so we
can see whether the raw telescope data themselves carry the wide-angle excess
that the prior work saw only at the coincidence level.

z-tel order for this dataset: file columns [0,1,2] -> z = [0, -1340, -670] mm.
"""

import sys
import struct
import math
import numpy as np

sys.path.insert(0, "src")
from monrad.reconstruction.hit import _decode_axis  # noqa: E402
from monrad.decoders.position import (  # noqa: E402
    POS_COORD_MASK,
    POS_X_SHIFT,
    BinDecoder,
)

Z_TEL = np.array([0.0, -1340.0, -670.0])  # per file column
STRIP_MM = 10.0

# Precompute the LS line-fit projection for 3 fixed z points.
# For points (z_i, y_i) fit y = a + b z.  Slope b and per-point residuals.
_Z = Z_TEL
_Zc = _Z - _Z.mean()
_Szz = (_Zc**2).sum()


def fit3(vals):
    """Return (slope, resid_rms) for 3 values at the fixed z positions."""
    b = (_Zc * (vals - vals.mean())).sum() / _Szz
    a = vals.mean() - b * _Z.mean()
    pred = a + b * _Z
    resid = vals - pred
    return b, resid


def load_blocks(path):
    with open(path, "rb") as f:
        raw = f.read()
    n_rows, n_cols = struct.unpack_from("<II", raw, 0)
    assert n_cols == 3, n_cols
    words = np.frombuffer(raw, dtype="<u8", count=n_rows * n_cols, offset=8)
    words = words.reshape(n_rows // 16, 16, n_cols)  # (nblk, 16, 3)
    return words


def or_block(words):
    """OR the 16 rows -> (nblk, 3) u64, then split x_or/y_or (nblk,3)."""
    ored = np.bitwise_or.reduce(words, axis=1)  # (nblk, 3)
    y_or = ored & POS_COORD_MASK
    x_or = (ored >> POS_X_SHIFT) & POS_COORD_MASK
    return x_or, y_or


def analyze(path, wide_thresh=0.5):
    words = load_blocks(path)
    nblk = words.shape[0]
    x_or, y_or = or_block(words)

    # Decode caches keyed by 20-bit OR value (huge reuse: only ~hundreds of
    # distinct single-hit patterns).
    cache = {}

    def dec(v):
        v = int(v)
        r = cache.get(v)
        if r is None:
            c, s, q = _decode_axis(v)
            r = (c, q)
            cache[v] = r
        return r

    slopes = []  # |b| for full 3-plane tracks (golden/cluster both axes)
    resid_rms = []  # 3-point line residual rms (mm), max over x/y
    good_planes = np.zeros(nblk, dtype=int)
    n_valid_plane = 0

    GOOD = ("golden", "cluster")
    for i in range(nblk):
        xs = np.empty(3)
        ys = np.empty(3)
        ng = 0
        for col in range(3):
            xv, yv = int(x_or[i, col]), int(y_or[i, col])
            valid, _ = BinDecoder._is_valid(xv, yv)
            if not valid:
                continue
            n_valid_plane += 1
            cx, qx = dec(xv)
            cy, qy = dec(yv)
            if qx in GOOD and qy in GOOD:
                ng += 1
                xs[col] = (cx + 0.5) * STRIP_MM
                ys[col] = (cy + 0.5) * STRIP_MM
        good_planes[i] = ng
        if ng == 3:
            bx, rx = fit3(xs)
            by, ry = fit3(ys)
            slopes.append(math.hypot(bx, by))
            resid_rms.append(max(math.sqrt((rx**2).mean()), math.sqrt((ry**2).mean())))

    slopes = np.array(slopes)
    resid_rms = np.array(resid_rms)
    n3 = len(slopes)

    def pct(a, p):
        return np.percentile(a, p) if len(a) else float("nan")

    return {
        "file": path.split("/")[-1],
        "nblk": nblk,
        "n_track3": n3,
        "frac_track3": n3 / nblk,
        "slope_med": pct(slopes, 50),
        "slope_p90": pct(slopes, 90),
        "slope_p99": pct(slopes, 99),
        "wide_frac": float((slopes > wide_thresh).mean()) if n3 else float("nan"),
        "wide_count": int((slopes > wide_thresh).sum()),
        "resid_med": pct(resid_rms, 50),
        "resid_p90": pct(resid_rms, 90),
        "mean_goodplanes": float(good_planes.mean()),
    }


if __name__ == "__main__":
    files = sys.argv[1:]
    hdr = (
        "file",
        "nblk",
        "n_tr3",
        "f_tr3",
        "sl_med",
        "sl_p90",
        "sl_p99",
        "wide_f",
        "wide_n",
        "rms_med",
        "rms_p90",
        "gp",
    )
    print(
        f"{'file':>13} {'nblk':>6} {'n_tr3':>6} {'f_tr3':>6} "
        f"{'sl_med':>7} {'sl_p90':>7} {'sl_p99':>7} {'wide_f':>7} {'wide_n':>7} "
        f"{'rms_med':>8} {'rms_p90':>8} {'gp':>5}"
    )
    for p in files:
        r = analyze(p)
        print(
            f"{r['file'][9:15]:>13} {r['nblk']:>6} {r['n_track3']:>6} "
            f"{r['frac_track3']:>6.3f} {r['slope_med']:>7.3f} {r['slope_p90']:>7.3f} "
            f"{r['slope_p99']:>7.3f} {r['wide_frac']:>7.4f} {r['wide_count']:>7} "
            f"{r['resid_med']:>8.2f} {r['resid_p90']:>8.2f} {r['mean_goodplanes']:>5.2f}"
        )
