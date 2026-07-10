"""
Temporal-clustering (burstiness) inspector for telescope events.

A beam-halo / shower burst deposits many events in a short time even if the
5-minute-averaged rate is flat.  From each *_GPS.bin file take the CLK counter
of every event record (FLAG=0), convert to seconds (100 MHz clock), and measure:

  rate    : events / duration  (Hz)
  cv      : coefficient of variation of inter-event gaps
            (Poisson memoryless -> cv~1; bursty/clustered -> cv>1)
  fano    : Fano factor var/mean of per-100ms-bin counts
            (Poisson -> ~1; clustered -> >1)
  f_short : fraction of gaps < 1 ms (pile-up / shower short gaps)
"""

import sys
import numpy as np

sys.path.insert(0, "src")
from monrad.decoders.gps import GPSDecoder, GPS_CLK_MASK, GPS_FLAG_SHIFT  # noqa: E402

CLK_HZ = 1e8  # 100 MHz


def load_event_clks(path):
    _, d = GPSDecoder(path).read()
    flag = (d >> GPS_FLAG_SHIFT) & 1
    clk = (d & GPS_CLK_MASK).astype(np.int64)
    return clk[flag == 0]


def analyze(path):
    clk = load_event_clks(path)
    t = clk / CLK_HZ  # seconds
    # guard against wrap / non-monotonic: use sorted unique-ish
    t = t - t[0]
    dur = t[-1] - t[0] if len(t) > 1 else 1.0
    gaps = np.diff(t)
    n_negative = int((gaps < 0).sum())
    if n_negative:
        print(
            f"{path}: dropping {n_negative} negative CLK gap(s) "
            "(non-monotonic sequence)",
            file=sys.stderr,
        )
    gaps = gaps[gaps >= 0]
    cv = gaps.std() / gaps.mean() if gaps.mean() > 0 else float("nan")
    # per-100ms bin counts
    nb = max(int(dur / 0.1), 1)
    counts, _ = np.histogram(t, bins=nb)
    fano = counts.var() / counts.mean() if counts.mean() > 0 else float("nan")
    f_short = float((gaps < 1e-3).mean())
    return {
        "file": path.split("/")[-1][9:15],
        "n": len(t),
        "dur": dur,
        "rate": len(t) / dur,
        "cv": cv,
        "fano": fano,
        "f_short": f_short,
        "gap_med_ms": np.median(gaps) * 1e3,
    }


if __name__ == "__main__":
    print(
        f"{'file':>7} {'n':>6} {'dur':>6} {'rate':>6} {'cv':>6} "
        f"{'fano':>6} {'f_short':>8} {'gapmed_ms':>9}"
    )
    for p in sys.argv[1:]:
        r = analyze(p)
        print(
            f"{r['file']:>7} {r['n']:>6} {r['dur']:>6.1f} {r['rate']:>6.2f} "
            f"{r['cv']:>6.2f} {r['fano']:>6.2f} {r['f_short']:>8.4f} "
            f"{r['gap_med_ms']:>9.2f}"
        )
