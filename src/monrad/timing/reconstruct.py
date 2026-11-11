"""
Stage 1 — per-detector time reconstruction.

Public API
----------
load_header_params(header_path)  -> (utc0, f0)
find_file_pairs(detector_dir)    -> (gps_paths, pos_paths)
reconstruct_stream(gps_paths, pos_paths, utc0, f0)
                                 -> Iterator[(TimedEvent, PosRef)]
"""

import bisect
import logging
from collections.abc import Iterator
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..decoders.gps import GPSDecoder, unpack_gps_word
from ..decoders.header import decode_ubx_tm2, find_header_files, parse_header_group
from ..decoders.position import POS_GEN_MASK, POS_GEN_SHIFT

log = logging.getLogger(__name__)

_UNIX_EPOCH = datetime(1970, 1, 1)
PPS_TAU = 1e-4  # residual threshold for PPS acceptance
F0_DEFAULT = 100_000_000  # Hz — used when header has no freq field

# A PPS record more than this fraction of a second away from the
# whole-second grid of the last accepted PPS is spurious (real data: two
# extra pulses splitting one second into 0.49 + 0.12 + 0.39 s).  It is far
# above oscillator drift, which _pps_is_spurious removes by predicting with
# the measured local frequency rather than f0.
PPS_SPURIOUS_FRAC = 0.05
# Safety valve: after this many consecutive rejected PPS, accept the next
# one anyway, so a wrong frequency reference cannot starve the chain.
_MAX_CONSECUTIVE_SPURIOUS = 10


# ── public types ─────────────────────────────────────────────────


class Quality(IntEnum):
    """Event timestamp quality, ordered worst-to-best via min()."""

    GOOD = 0  # bracketed by two accepted PPS anchors
    DEGRADED = 1  # extrapolated beyond PPS coverage
    UNTRUSTED = 2  # inside a failed PPS interval


class TimedEvent(NamedTuple):
    t_ns: int
    evt_seq: int
    quality: Quality


class PosRef(NamedTuple):
    """Location of one event's 16-row block in the position files."""

    file_idx: int
    row_offset: int  # first row of the 16-row block in file_idx
    split_rows: int = 0  # >0: this many rows are in file_idx,
    # the rest (16-split_rows) start at row 0
    # of file_idx+1


# ── internal helpers ─────────────────────────────────────────────


def _utc_to_ns(utc: datetime) -> int:
    """Integer nanoseconds since Unix epoch, microsecond precision."""
    d = utc - _UNIX_EPOCH
    return (d.days * 86_400 + d.seconds) * 1_000_000_000 + d.microseconds * 1_000


class _Interval:
    """One PPS-to-PPS interval."""

    __slots__ = ("c0", "c1", "n0", "n1", "dc", "dn", "trusted")

    def __init__(
        self,
        c0: int,
        c1: int,
        n0: int,
        n1: int,
        dc: int,
        dn: int,
        trusted: bool,
    ) -> None:
        self.c0 = c0
        self.c1 = c1
        self.n0 = n0  # seconds elapsed at c0
        self.n1 = n1  # seconds elapsed at c1
        self.dc = dc  # c1 - c0  (ticks)
        self.dn = dn  # n1 - n0  (seconds)
        self.trusted = trusted


def _iter_gps_records(
    path: Path,
) -> Iterator[tuple[int, int, bool]]:
    """Yield (tick, gen, is_pps) from one *_GPS.bin in order."""
    _, data = GPSDecoder(str(path)).read()
    for raw in data:
        tick, gen, flag = unpack_gps_word(int(raw))
        yield tick, gen, flag


def _parse_gps_file(
    path: Path,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Return (evt_ticks, evt_gens, pps_ticks, pps_gens) from one
    *_GPS.bin file, in acquisition order.
    """
    evt_ticks: list[int] = []
    evt_gens: list[int] = []
    pps_ticks: list[int] = []
    pps_gens: list[int] = []
    for tick, gen, is_pps in _iter_gps_records(path):
        if is_pps:
            pps_ticks.append(tick)
            pps_gens.append(gen)
        else:
            evt_ticks.append(tick)
            evt_gens.append(gen)
    return evt_ticks, evt_gens, pps_ticks, pps_gens


def _read_row_gens(path: Path) -> np.ndarray:
    """GEN of every row of one *.bin (column 0; all columns share it)."""
    hdr = np.fromfile(path, dtype="<u4", count=2)
    if len(hdr) < 2:
        return np.zeros(0, dtype=np.int16)
    n_rows, n_cols = int(hdr[0]), int(hdr[1])
    if n_rows == 0 or n_cols == 0:
        return np.zeros(0, dtype=np.int16)
    words = np.fromfile(path, dtype="<u8", offset=8, count=n_rows * n_cols)
    col0 = words[: (len(words) // n_cols) * n_cols : n_cols]
    return ((col0 >> np.uint64(POS_GEN_SHIFT)) & np.uint64(POS_GEN_MASK)).astype(
        np.int16
    )


# A GPS event and its *.bin block are paired by GEN.  On a mismatch the
# cursor scans this many blocks ahead for the event's GEN.  It is far below
# the 2048-event GEN period, so a match inside the window is unambiguous.
_RESYNC_BLOCKS = 64


class _BlockCursor:
    """
    Walk the concatenated ``*.bin`` block stream of one detector.

    The DAQ rotates ``*_GPS.bin`` and ``*.bin`` files independently, a few
    events apart: ``*.bin`` file k+1 often begins with the block of the last
    event in ``*_GPS.bin`` file k (or ends a block early).  Pairing the k-th
    GPS event of a file with the k-th block of the same-index ``*.bin``
    therefore mispairs every event of that file.  Concatenated end to end,
    though, the event-GEN and block-GEN sequences coincide, so this cursor
    advances through the block stream independently of GPS file boundaries
    and checks every pairing by GEN.  A block that straddles two files comes
    out naturally as a ``split_rows`` PosRef.

    Only the per-row GEN arrays of the file under the cursor (and of files
    reached by a straddling block or a resync look-ahead) are held in memory.
    """

    def __init__(self, pos_paths: list[Path]) -> None:
        self._paths = [Path(p) for p in pos_paths]
        self._gens: dict[int, np.ndarray] = {}
        self._file = 0
        self._row = 0
        self.n_resyncs = 0
        self.n_skipped_blocks = 0
        self.n_unpaired = 0

    def _row_gens(self, i: int) -> np.ndarray:
        g = self._gens.get(i)
        if g is None:
            g = _read_row_gens(self._paths[i])
            self._gens[i] = g
        return g

    def _block_at(self, f: int, r: int) -> tuple[int, PosRef, tuple[int, int]] | None:
        """(block GEN, PosRef, cursor after the block), or None at stream end."""
        n = len(self._paths)
        while f < n and r >= len(self._row_gens(f)):
            r -= len(self._row_gens(f))
            f += 1
        if f >= n:
            return None
        gens = self._row_gens(f)
        gen = int(gens[r])
        remaining = len(gens) - r
        if remaining >= 16:
            return gen, PosRef(f, r), (f, r + 16)
        if f + 1 < n and len(self._row_gens(f + 1)) >= 16 - remaining:
            return gen, PosRef(f, r, split_rows=remaining), (f + 1, 16 - remaining)
        # Truncated final block: decode_position pads the short read with zeros.
        return gen, PosRef(f, r), (f, r + 16)

    def take(self, gen: int, label: str) -> PosRef | None:
        """
        PosRef of the block carrying ``gen``, advancing the cursor past it.

        Returns None (cursor unchanged) when no block within
        ``_RESYNC_BLOCKS`` carries ``gen``: the event has no position data.
        """
        pos = (self._file, self._row)
        for skip in range(_RESYNC_BLOCKS + 1):
            blk = self._block_at(*pos)
            if blk is None:
                break
            blk_gen, ref, nxt = blk
            if blk_gen == gen:
                if skip:
                    self.n_resyncs += 1
                    self.n_skipped_blocks += skip
                    log.warning(
                        "%s: GEN %d found %d block(s) ahead in %s;"
                        " skipped the unmatched block(s)",
                        label,
                        gen,
                        skip,
                        self._paths[ref.file_idx].name,
                    )
                self._file, self._row = nxt
                for k in [k for k in self._gens if k < self._file]:
                    del self._gens[k]
                return ref
            pos = nxt
        self.n_unpaired += 1
        log.warning(
            "%s: no *.bin block with GEN %d within %d blocks of %s row %d;"
            " event dropped (no position data)",
            label,
            gen,
            _RESYNC_BLOCKS,
            self._paths[min(self._file, len(self._paths) - 1)].name,
            self._row,
        )
        return None


def _build_next_interval(
    c0: int,
    c1: int,
    n0: int,
    f0: int,
    tau: float,
) -> _Interval:
    """Build one PPS-to-PPS _Interval from a consecutive tick pair."""
    dc = c1 - c0
    if dc <= 0:
        log.warning("PPS tick not monotonic (dc=%d); untrusted", dc)
        return _Interval(c0, c1, n0, n0 + 1, max(dc, 1), 1, False)
    n = round(dc / f0)
    if n == 0:
        log.warning("Two PPS records within one f0 period; untrusted")
        return _Interval(c0, c1, n0, n0 + 1, dc, 1, False)
    res = abs(dc - n * f0) / (n * f0)
    trusted = res <= tau
    n1 = n0 + n
    if not trusted:
        log.warning(
            "PPS residual %.2e > tau=%.0e at N~%d — interval untrusted",
            res,
            tau,
            n1,
        )
    elif n > 1:
        log.warning(
            "%d dropped PPS pulses between N=%d and N=%d",
            n - 1,
            n0,
            n1,
        )
    return _Interval(c0, c1, n0, n1, dc, n, trusted)


def _pps_is_spurious(dc: int, f_ref: float) -> bool:
    """
    True when a PPS ``dc`` ticks after the last accepted PPS is off the
    whole-second grid: fewer than half a second later, or more than
    PPS_SPURIOUS_FRAC of a second away from an integer number of seconds.

    Without this check, each sub-second fragment would be counted as a full
    second and the detector's clock would jump ahead by whole seconds for the
    rest of the run.  ``dc <= 0`` (a counter reset) is not spurious: it is
    handled, as untrusted, by _build_next_interval.
    """
    if dc <= 0:
        return False
    n = round(dc / f_ref)
    return n == 0 or abs(dc - n * f_ref) > PPS_SPURIOUS_FRAC * f_ref


def _linear(utc0_ns: int, iv: _Interval, tick: int) -> int:
    """
    Integer-arithmetic timestamp for tick inside (or extrapolated
    from) interval iv.

    t_ns = UTC0_ns + N0·10⁹ + (tick−C0)·10⁹·dn // dc
    """
    return (
        utc0_ns
        + iv.n0 * 1_000_000_000
        + (tick - iv.c0) * 1_000_000_000 * iv.dn // iv.dc
    )


def _timestamp(
    tick: int,
    ivs: list[_Interval],
    c0s: list[int],  # [iv.c0 for iv in ivs], pre-built
    c0: int,  # tick of the first PPS
    utc0_ns: int,
    f0: int,
    back_iv: _Interval | None,
    fwd_iv: _Interval | None,
) -> tuple[int, Quality]:
    """
    Compute (t_ns, quality) for a single event clock tick.
    All arithmetic stays in integers.
    """
    # ── locate containing interval ──────────────────────────────
    if ivs:
        idx = bisect.bisect_right(c0s, tick) - 1
        if 0 <= idx < len(ivs) and tick < ivs[idx].c1:
            iv = ivs[idx]
            q = Quality.GOOD if iv.trusted else Quality.UNTRUSTED
            return _linear(utc0_ns, iv, tick), q

    # ── before first interval (or no intervals at all) ──────────
    if not ivs or tick < c0:
        if back_iv is not None:
            t = utc0_ns + (tick - c0) * 1_000_000_000 * back_iv.dn // back_iv.dc
        else:
            t = utc0_ns + (tick - c0) * 1_000_000_000 // f0
        return t, Quality.DEGRADED

    # ── after last interval ──────────────────────────────────────
    ref = fwd_iv if fwd_iv is not None else (ivs[-1] if ivs else None)
    if ref is not None:
        t = _linear(utc0_ns, ref, tick)
    else:
        t = utc0_ns + (tick - c0) * 1_000_000_000 // f0
    return t, Quality.DEGRADED


# ── public utilities ─────────────────────────────────────────────


def load_header_params(
    header_path: Path,
) -> tuple[datetime, int]:
    """
    Extract (utc0, f0) from a *_header.txt file.

    utc0 is the UTC time of the TIMEPULSE rising edge, which the
    pipeline treats as the time of the first PPS record.

    f0 is the nominal clock frequency in Hz.  Falls back to
    F0_DEFAULT (100 MHz) when the header has no explicit field.

    ``header_path`` names any one file of an acquisition run; every
    ``<stem>_header<NNN>.txt`` sibling is merged, because the DAQ may split
    the GPS string across two of them or park it in a higher-numbered file
    on its own.
    """
    header_files = find_header_files(header_path)
    modules = parse_header_group(header_files)

    # Real hardware headers carry no clock frequency field; F0_DEFAULT is
    # always the live fallback.  The key-search below exists solely to read
    # the 'clock_freq' entry written by monrad.synth.generate() in synthetic
    # test headers — it is unreachable on production data.
    f0 = F0_DEFAULT
    for params in modules.values():
        for key, val in params.items():
            if "clock" in key.lower() and "freq" in key.lower():
                f0 = int(val)
                break

    # UTC0 from the GPS UBX-TIM-TM2 frame
    gps_bytes: bytes | None = None
    gps_mod = modules.get("GPS", {})
    for key, val in gps_mod.items():
        if key.startswith("GPS_String") and isinstance(val, bytes):
            gps_bytes = val
            break
    if gps_bytes is None:
        inspected = ", ".join(p.name for p in header_files)
        raise ValueError(
            f"No GPS_String found in {header_files[0].parent} ({inspected})"
        )
    tm2 = decode_ubx_tm2(gps_bytes)
    utc0_raw: datetime = tm2["timeR"]
    # The header's TIMEPULSE is a 100 Hz calibration pulse, not a 1 Hz PPS.
    # PPS edges always fire at integer seconds.  Snap to the next whole
    # second so that utc0 correctly anchors the first PPS in the GPS stream.
    if utc0_raw.microsecond:
        utc0: datetime = utc0_raw.replace(microsecond=0) + timedelta(seconds=1)
    else:
        utc0 = utc0_raw
    return utc0, f0


def find_file_pairs(
    detector_dir: Path,
) -> tuple[list[Path], list[Path]]:
    """
    Find all (*_GPS.bin, *.bin) file pairs in a detector directory,
    sorted by filename (i.e. by acquisition timestamp).

    Raises FileNotFoundError if any GPS file has no matching
    position file.
    """
    gps_paths = sorted(Path(detector_dir).glob("*_GPS.bin"))
    pos_paths: list[Path] = []
    for gps in gps_paths:
        stem = gps.name[: -len("_GPS.bin")]
        pos = gps.parent / f"{stem}.bin"
        if not pos.exists():
            raise FileNotFoundError(f"No position file for {gps.name}")
        pos_paths.append(pos)
    return gps_paths, pos_paths


# ── streaming entry point ────────────────────────────────────────


def reconstruct_stream(
    gps_paths: list[Path],
    pos_paths: list[Path],
    utc0: datetime,
    f0: int = F0_DEFAULT,
    tau: float = PPS_TAU,
) -> Iterator[tuple[TimedEvent, PosRef]]:
    """
    Streaming Stage 1.  Yields (TimedEvent, PosRef) pairs in time
    order, emitting each PPS interval's events as soon as the
    closing PPS is observed.

    Memory bound: at most ~2 s of events in pending buffers at
    startup (pre-PPS_1 + PPS_1→PPS_2), ~1 s thereafter.

    Yields
    ------
    TimedEvent  (t_ns, evt_seq, quality)
    PosRef      (file_idx, row_offset, split_rows)
    """
    if len(gps_paths) != len(pos_paths):
        raise ValueError("gps_paths and pos_paths must have the same length")

    utc0_ns = _utc_to_ns(utc0)

    # (tick, PosRef) — events buffered since last PPS
    _pending: list[tuple[int, PosRef]] = []
    # (tick, PosRef) — events before the first PPS record
    _pre_pps1: list[tuple[int, PosRef]] = []

    pps_count = 0
    prev_pps_tick: int | None = None
    n0 = 0  # cumulative seconds at prev_pps_tick
    last_iv: _Interval | None = None  # last built, for fwd extrap
    evt_seq = 0

    # Events are paired with *.bin blocks by GEN over the whole detector
    # stream, not per file: see _BlockCursor for why.
    blocks = _BlockCursor(pos_paths)

    n_spurious = 0  # off-grid PPS records ignored (see _pps_is_spurious)
    consecutive_spurious = 0

    for gps_path in gps_paths:
        gps_path = Path(gps_path)

        for tick, gen, is_pps in _iter_gps_records(gps_path):
            if is_pps:
                if prev_pps_tick is not None:
                    # Predict with the measured local frequency when we have
                    # a trusted interval, so oscillator drift over a long
                    # dropped-PPS gap cannot look like an off-grid pulse.
                    f_ref = (
                        last_iv.dc / last_iv.dn
                        if last_iv is not None and last_iv.trusted
                        else f0
                    )
                    if (
                        _pps_is_spurious(tick - prev_pps_tick, f_ref)
                        and consecutive_spurious < _MAX_CONSECUTIVE_SPURIOUS
                    ):
                        consecutive_spurious += 1
                        n_spurious += 1
                        log.warning(
                            "%s: spurious PPS %.6f s after the last accepted"
                            " one; ignored",
                            gps_path.name,
                            (tick - prev_pps_tick) / f_ref,
                        )
                        continue
                consecutive_spurious = 0
                pps_count += 1

                if pps_count == 1:
                    # Note PPS_1 tick; cannot close any interval yet.
                    prev_pps_tick = tick
                    n0 = 0

                elif pps_count == 2:
                    # PPS_2 closes the startup window.  Build the
                    # PPS_1→PPS_2 interval, use it to back-extrapolate
                    # pre-PPS_1 events (DEGRADED) and to timestamp
                    # PPS_1→PPS_2 events normally.
                    back_iv = _build_next_interval(prev_pps_tick, tick, n0, f0, tau)
                    last_iv = back_iv

                    for ev_tick, ref in _pre_pps1:
                        t_ns = _linear(utc0_ns, back_iv, ev_tick)
                        yield (
                            TimedEvent(t_ns, evt_seq, Quality.DEGRADED),
                            ref,
                        )
                        evt_seq += 1
                    _pre_pps1.clear()

                    q = Quality.GOOD if back_iv.trusted else Quality.UNTRUSTED
                    for ev_tick, ref in _pending:
                        t_ns = _linear(utc0_ns, back_iv, ev_tick)
                        yield TimedEvent(t_ns, evt_seq, q), ref
                        evt_seq += 1
                    _pending.clear()

                    n0 = back_iv.n1
                    prev_pps_tick = tick

                else:
                    # PPS_3+: standard one-interval-at-a-time flow.
                    iv = _build_next_interval(prev_pps_tick, tick, n0, f0, tau)
                    last_iv = iv
                    q = Quality.GOOD if iv.trusted else Quality.UNTRUSTED
                    for ev_tick, ref in _pending:
                        t_ns = _linear(utc0_ns, iv, ev_tick)
                        yield TimedEvent(t_ns, evt_seq, q), ref
                        evt_seq += 1
                    _pending.clear()

                    n0 = iv.n1
                    prev_pps_tick = tick

            else:
                # Event record: pair it with its *.bin block by GEN and
                # buffer it.  An event with no block is dropped.
                ref = blocks.take(gen, gps_path.name)
                if ref is None:
                    continue
                if pps_count == 0:
                    _pre_pps1.append((tick, ref))
                else:
                    _pending.append((tick, ref))

    if n_spurious:
        log.warning("%d spurious PPS record(s) ignored", n_spurious)

    if blocks.n_resyncs or blocks.n_unpaired:
        log.warning(
            "GPS/*.bin pairing: %d resync(s) skipped %d unmatched block(s);"
            " %d event(s) dropped with no block",
            blocks.n_resyncs,
            blocks.n_skipped_blocks,
            blocks.n_unpaired,
        )

    if not pps_count:
        raise ValueError("No PPS records found — cannot anchor timestamps")

    # Flush remaining events (no closing PPS) with forward extrapolation.
    if _pre_pps1 or _pending:
        if last_iv is not None:
            fwd_iv = last_iv
        elif prev_pps_tick is not None:
            fwd_iv = _Interval(
                prev_pps_tick,
                prev_pps_tick + f0,
                n0,
                n0 + 1,
                f0,
                1,
                False,
            )
        else:
            fwd_iv = _Interval(0, f0, 0, 1, f0, 1, False)

        # pre-PPS_1 events have smaller ticks → yield first.
        for buf in (_pre_pps1, _pending):
            for ev_tick, ref in buf:
                t_ns = _linear(utc0_ns, fwd_iv, ev_tick)
                yield (
                    TimedEvent(t_ns, evt_seq, Quality.DEGRADED),
                    ref,
                )
                evt_seq += 1
