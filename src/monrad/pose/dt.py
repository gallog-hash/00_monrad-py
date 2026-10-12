"""
Stage 5 (part) — summarising probe-minus-telescope event time (Δt).

Δt is pure instrumentation: no stage of the pipeline gates on it.  See
:attr:`~monrad.pose.types.Coincidence.dt_ns` for the quantity itself and
DESIGN.md §5.3 for why its *per-event* sign is not physically meaningful
while its *distribution* is the measurement DESIGN.md §5 defers the
coincidence-window decision to.

Public API
----------
dt_counter(coincidences)        -> Counter keyed by Δt in ns
dt_stats(hist)                  -> DtStats
dt_histogram(hist, bin_ns)      -> list[str]   (text histogram rows)

Δt is carried as a ``Counter`` keyed by the integer ns value rather than a
list of samples: it is quantised to the 10 ns clock tick and bounded by the
coincidence window, so the key set stays small however long the run is, and
both the monitor's per-window summary and the whole-run pipeline report can
use the same accumulator.
"""

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import NamedTuple

__all__ = ["DtStats", "dt_counter", "dt_stats", "dt_histogram"]


class DtStats(NamedTuple):
    """Summary of a Δt distribution, all times in integer-ns units.

    ``std`` is the *sample* standard deviation (n−1 denominator): the width
    is an estimate of the detectors' combined timing resolution from a
    finite sample, not the spread of a known population.  It is 0.0 for a
    single entry and nan for an empty one, as are ``mean``/``median``.
    """

    n: int
    mean: float
    std: float
    median: float
    min_ns: int
    max_ns: int

    @classmethod
    def empty(cls) -> "DtStats":
        return cls(0, math.nan, math.nan, math.nan, 0, 0)


def dt_counter(coincidences: Iterable) -> Counter:
    """Counter of ``dt_ns`` over an iterable of :class:`Coincidence`."""
    return Counter(co.dt_ns for co in coincidences)


def dt_stats(hist: Mapping[int, int]) -> DtStats:
    """Summarise a Δt histogram (Counter keyed by Δt in ns)."""
    n = sum(hist.values())
    if n == 0:
        return DtStats.empty()
    keys = sorted(hist)
    mean = sum(k * c for k, c in hist.items()) / n
    if n == 1:
        return DtStats(1, mean, 0.0, float(keys[0]), keys[0], keys[0])
    var = sum(c * (k - mean) ** 2 for k, c in hist.items()) / (n - 1)
    return DtStats(n, mean, math.sqrt(var), _median(hist, keys, n), keys[0], keys[-1])


def _median(hist: Mapping[int, int], keys: list[int], n: int) -> float:
    """Median by walking the cumulative count.

    Averages the two central order statistics for an even ``n``, matching
    what a median over the equivalent flat sample list would give.
    """
    lo_target, hi_target = (n - 1) // 2, n // 2
    lo = hi = keys[-1]
    seen = 0
    for k in keys:
        seen += hist[k]
        if seen > lo_target:
            lo = k
            break
    seen = 0
    for k in keys:
        seen += hist[k]
        if seen > hi_target:
            hi = k
            break
    return (lo + hi) / 2.0


def dt_histogram(
    hist: Mapping[int, int], bin_ns: int = 20, width: int = 40
) -> list[str]:
    """Text histogram rows for a Δt Counter, in ``bin_ns``-wide bins.

    Bins are floor-aligned to multiples of ``bin_ns`` so that a bin edge
    falls on 0 — the point where probe and telescope are timestamped alike,
    which is what the distribution's offset should be read against.  Empty
    interior bins are kept, so a bimodal or tailed distribution is visible
    rather than collapsed.
    """
    if not hist:
        return []
    if bin_ns < 1:
        raise ValueError(f"bin_ns must be >= 1, got {bin_ns}")
    binned: Counter = Counter()
    for k, c in hist.items():
        binned[math.floor(k / bin_ns) * bin_ns] += c
    peak = max(binned.values())
    lo, hi = min(binned), max(binned)
    rows = []
    for edge in range(lo, hi + bin_ns, bin_ns):
        c = binned.get(edge, 0)
        bar = "#" * int(round(width * c / peak)) if c else ""
        rows.append(f"[{edge:+6d},{edge + bin_ns:+6d}) {c:>7}  {bar}")
    return rows
