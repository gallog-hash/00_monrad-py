"""
Stage 1 — per-detector time reconstruction.

Public API
----------
load_header_params(header_path)  -> (utc0, f0)
find_file_pairs(detector_dir)    -> (gps_paths, pos_paths)
reconstruct_stream(gps_paths, pos_paths, utc0, f0)
                                 -> Iterator[(TimedEvent, PosRef)]
reconstruct(gps_paths, pos_paths, utc0, f0)
                                 -> (events, pos_map)  [deprecated]
"""

import bisect
import logging
import struct
import warnings
from collections.abc import Iterator
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple

from .decoders.gps import GPSDecoder
from .decoders.header import parse_header, decode_ubx_tm2

log = logging.getLogger(__name__)

_UNIX_EPOCH = datetime(1970, 1, 1)
PPS_TAU = 1e-4          # residual threshold for PPS acceptance
F0_DEFAULT = 100_000_000  # Hz — used when header has no freq field


# ── public types ─────────────────────────────────────────────────

class Quality(IntEnum):
    """Event timestamp quality, ordered worst-to-best via min()."""
    GOOD      = 0   # bracketed by two accepted PPS anchors
    DEGRADED  = 1   # extrapolated beyond PPS coverage
    UNTRUSTED = 2   # inside a failed PPS interval


class TimedEvent(NamedTuple):
    t_ns:    int
    evt_seq: int
    quality: Quality


class PosRef(NamedTuple):
    """Location of one event's 16-row block in the position files."""
    file_idx:   int
    row_offset: int   # first row of the 16-row block in file_idx
    split_rows: int = 0  # >0: this many rows are in file_idx,
                         # the rest (16-split_rows) start at row 0
                         # of file_idx+1


# ── internal helpers ─────────────────────────────────────────────

def _utc_to_ns(utc: datetime) -> int:
    """Integer nanoseconds since Unix epoch, microsecond precision."""
    d = utc - _UNIX_EPOCH
    return (
        (d.days * 86_400 + d.seconds) * 1_000_000_000
        + d.microseconds * 1_000
    )


class _Interval:
    """One PPS-to-PPS interval."""
    __slots__ = ('c0', 'c1', 'n0', 'n1', 'dc', 'dn', 'trusted')

    def __init__(
        self,
        c0: int, c1: int,
        n0: int, n1: int,
        dc: int, dn: int,
        trusted: bool,
    ) -> None:
        self.c0 = c0
        self.c1 = c1
        self.n0 = n0   # seconds elapsed at c0
        self.n1 = n1   # seconds elapsed at c1
        self.dc = dc   # c1 - c0  (ticks)
        self.dn = dn   # n1 - n0  (seconds)
        self.trusted = trusted


def _iter_gps_records(
    path: Path,
) -> Iterator[tuple[int, int, bool]]:
    """Yield (tick, gen, is_pps) from one *_GPS.bin in order."""
    _, data = GPSDecoder(str(path)).read()
    for raw in data:
        v    = int(raw)
        tick = v & 0xFFFFFFFFFFFFF
        gen  = (v >> 52) & 0x7FF
        flag = bool((v >> 63) & 1)
        yield tick, gen, flag


def _parse_gps_file(
    path: Path,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Return (evt_ticks, evt_gens, pps_ticks, pps_gens) from one
    *_GPS.bin file, in acquisition order.
    """
    evt_ticks: list[int] = []
    evt_gens:  list[int] = []
    pps_ticks: list[int] = []
    pps_gens:  list[int] = []
    for tick, gen, is_pps in _iter_gps_records(path):
        if is_pps:
            pps_ticks.append(tick)
            pps_gens.append(gen)
        else:
            evt_ticks.append(tick)
            evt_gens.append(gen)
    return evt_ticks, evt_gens, pps_ticks, pps_gens


def _pos_file_meta(
    path: Path,
) -> tuple[int, int, int, int]:
    """
    Read (n_rows, n_cols, first_gen, last_gen) from *.bin
    without loading the full array.
    """
    with open(path, 'rb') as fh:
        n_rows = struct.unpack_from('<I', fh.read(4))[0]
        n_cols = struct.unpack_from('<I', fh.read(4))[0]
        if n_rows == 0 or n_cols == 0:
            return n_rows, n_cols, -1, -1
        first_word = struct.unpack('<Q', fh.read(8))[0]
        first_gen = (first_word >> 52) & 0x7FF
        fh.seek(8 + (n_rows - 1) * n_cols * 8)
        last_word = struct.unpack('<Q', fh.read(8))[0]
        last_gen = (last_word >> 52) & 0x7FF
    return n_rows, n_cols, first_gen, last_gen


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
        log.warning('PPS tick not monotonic (dc=%d); untrusted', dc)
        return _Interval(c0, c1, n0, n0 + 1, max(dc, 1), 1, False)
    n = round(dc / f0)
    if n == 0:
        log.warning(
            'Two PPS records within one f0 period; untrusted'
        )
        return _Interval(c0, c1, n0, n0 + 1, dc, 1, False)
    res = abs(dc - n * f0) / (n * f0)
    trusted = res <= tau
    n1 = n0 + n
    if not trusted:
        log.warning(
            'PPS residual %.2e > tau=%.0e at N~%d — '
            'interval untrusted',
            res, tau, n1,
        )
    elif n > 1:
        log.warning(
            '%d dropped PPS pulses between N=%d and N=%d',
            n - 1, n0, n1,
        )
    return _Interval(c0, c1, n0, n1, dc, n, trusted)



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
    tick:     int,
    ivs:      list[_Interval],
    c0s:      list[int],       # [iv.c0 for iv in ivs], pre-built
    c0:       int,             # tick of the first PPS
    utc0_ns:  int,
    f0:       int,
    back_iv:  _Interval | None,
    fwd_iv:   _Interval | None,
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
            t = (
                utc0_ns
                + (tick - c0) * 1_000_000_000 * back_iv.dn
                // back_iv.dc
            )
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
    """
    modules = parse_header(str(header_path))

    # Real hardware headers carry no clock frequency field; F0_DEFAULT is
    # always the live fallback.  The key-search below exists solely to read
    # the 'clock_freq' entry written by monrad.synth.generate() in synthetic
    # test headers — it is unreachable on production data.
    f0 = F0_DEFAULT
    for params in modules.values():
        for key, val in params.items():
            if 'clock' in key.lower() and 'freq' in key.lower():
                f0 = int(val)
                break

    # UTC0 from the GPS UBX-TIM-TM2 frame
    gps_bytes: bytes | None = None
    gps_mod = modules.get('GPS', {})
    for key, val in gps_mod.items():
        if key.startswith('GPS_String') and isinstance(val, bytes):
            gps_bytes = val
            break
    if gps_bytes is None:
        raise ValueError(
            f'No GPS_String found in {header_path}'
        )
    tm2 = decode_ubx_tm2(gps_bytes)
    utc0_raw: datetime = tm2['timeR']
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
    gps_paths = sorted(Path(detector_dir).glob('*_GPS.bin'))
    pos_paths: list[Path] = []
    for gps in gps_paths:
        stem = gps.name[: -len('_GPS.bin')]
        pos = gps.parent / f'{stem}.bin'
        if not pos.exists():
            raise FileNotFoundError(
                f'No position file for {gps.name}'
            )
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
        raise ValueError(
            'gps_paths and pos_paths must have the same length'
        )

    utc0_ns = _utc_to_ns(utc0)

    # (tick, PosRef) — events buffered since last PPS
    _pending:  list[tuple[int, PosRef]] = []
    # (tick, PosRef) — events before the first PPS record
    _pre_pps1: list[tuple[int, PosRef]] = []

    pps_count = 0
    prev_pps_tick: int | None = None
    n0 = 0                      # cumulative seconds at prev_pps_tick
    last_iv: _Interval | None = None   # last built, for fwd extrap
    evt_seq = 0

    # split-block detection
    prev_pos_last_gen: int | None = None
    prev_pos_nr:       int | None = None

    for file_idx, (gps_path, pos_path) in enumerate(
        zip(gps_paths, pos_paths)
    ):
        gps_path = Path(gps_path)
        pos_path = Path(pos_path)

        nr, nc, first_gen, last_gen = _pos_file_meta(pos_path)

        if nr % 16 != 0:
            log.warning(
                '%s: row count %d not multiple of 16'
                ' (possible split block at file end)',
                pos_path.name, nr,
            )

        # GEN continuity check and split-block detection
        if prev_pos_last_gen is not None and nr > 0:
            expected = (prev_pos_last_gen + 1) % 2048
            if first_gen == prev_pos_last_gen:
                # Last block of previous file straddled the boundary
                tail = (prev_pos_nr or 0) % 16
                if tail and _pending:
                    old_tick, old_ref = _pending[-1]
                    _pending[-1] = (
                        old_tick,
                        PosRef(
                            old_ref.file_idx,
                            old_ref.row_offset,
                            split_rows=tail,
                        ),
                    )
                log.warning(
                    '%s: split block detected — %d rows in prev file',
                    pos_path.name, tail if tail else 0,
                )
            elif first_gen != expected:
                log.warning(
                    '%s: GEN discontinuity at boundary'
                    ' — expected %d, got %d',
                    pos_path.name, expected, first_gen,
                )

        n_pos_events = nr // 16
        local_event_idx = 0

        for tick, gen, is_pps in _iter_gps_records(gps_path):
            if is_pps:
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
                    back_iv = _build_next_interval(
                        prev_pps_tick, tick, n0, f0, tau
                    )
                    last_iv = back_iv

                    for ev_tick, ref in _pre_pps1:
                        t_ns = _linear(utc0_ns, back_iv, ev_tick)
                        yield (
                            TimedEvent(t_ns, evt_seq, Quality.DEGRADED),
                            ref,
                        )
                        evt_seq += 1
                    _pre_pps1.clear()

                    q = (
                        Quality.GOOD
                        if back_iv.trusted
                        else Quality.UNTRUSTED
                    )
                    for ev_tick, ref in _pending:
                        t_ns = _linear(utc0_ns, back_iv, ev_tick)
                        yield TimedEvent(t_ns, evt_seq, q), ref
                        evt_seq += 1
                    _pending.clear()

                    n0 = back_iv.n1
                    prev_pps_tick = tick

                else:
                    # PPS_3+: standard one-interval-at-a-time flow.
                    iv = _build_next_interval(
                        prev_pps_tick, tick, n0, f0, tau
                    )
                    last_iv = iv
                    q = (
                        Quality.GOOD if iv.trusted else Quality.UNTRUSTED
                    )
                    for ev_tick, ref in _pending:
                        t_ns = _linear(utc0_ns, iv, ev_tick)
                        yield TimedEvent(t_ns, evt_seq, q), ref
                        evt_seq += 1
                    _pending.clear()

                    n0 = iv.n1
                    prev_pps_tick = tick

            else:
                # Event record: assign PosRef and buffer it.
                ref = PosRef(file_idx, local_event_idx * 16)
                if pps_count == 0:
                    _pre_pps1.append((tick, ref))
                else:
                    _pending.append((tick, ref))
                local_event_idx += 1

        if local_event_idx != n_pos_events:
            log.warning(
                '%s: %d GPS events but %d position blocks',
                pos_path.name, local_event_idx, n_pos_events,
            )

        prev_pos_last_gen = last_gen if nr > 0 else prev_pos_last_gen
        prev_pos_nr = nr

    if not pps_count:
        raise ValueError(
            'No PPS records found — cannot anchor timestamps'
        )

    # Flush remaining events (no closing PPS) with forward extrapolation.
    if _pre_pps1 or _pending:
        if last_iv is not None:
            fwd_iv = last_iv
        elif prev_pps_tick is not None:
            fwd_iv = _Interval(
                prev_pps_tick, prev_pps_tick + f0,
                n0, n0 + 1, f0, 1, False,
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


# ── deprecated batch wrapper ─────────────────────────────────────

def reconstruct(
    gps_paths: list[Path],
    pos_paths: list[Path],
    utc0: datetime,
    f0: int = F0_DEFAULT,
    tau: float = PPS_TAU,
) -> tuple[list[TimedEvent], dict[int, PosRef]]:
    """
    Deprecated. Use reconstruct_stream() for production pipelines.

    Thin wrapper around reconstruct_stream() that materialises all
    events and returns (events, pos_map) for backward compatibility.
    """
    warnings.warn(
        'reconstruct() is deprecated; use reconstruct_stream()',
        DeprecationWarning,
        stacklevel=2,
    )
    events:    list[TimedEvent] = []
    pos_index: list[PosRef]    = []
    for ev, ref in reconstruct_stream(
        gps_paths, pos_paths, utc0, f0, tau
    ):
        events.append(ev)
        pos_index.append(ref)
    pos_map = {
        ev.evt_seq: ref
        for ev, ref in zip(events, pos_index)
    }
    return events, pos_map
