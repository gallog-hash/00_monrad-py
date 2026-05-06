"""
Stage 1 — per-detector time reconstruction.

Converts one detector's set of binary files into a list of
TimedEvent(t_ns, evt_seq, quality), plus a PosRef table that
maps each event to its raw position rows on disk.

Public API
----------
load_header_params(header_path)  -> (utc0, f0)
find_file_pairs(detector_dir)    -> (gps_paths, pos_paths)
reconstruct(gps_paths, pos_paths, utc0, f0) -> (events, pos_map)
"""

import bisect
import logging
import struct
from datetime import datetime
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


def _parse_gps_file(
    path: Path,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Return (evt_ticks, evt_gens, pps_ticks, pps_gens) from one
    *_GPS.bin file, in acquisition order.
    """
    _, data = GPSDecoder(str(path)).read()
    evt_ticks: list[int] = []
    evt_gens:  list[int] = []
    pps_ticks: list[int] = []
    pps_gens:  list[int] = []
    for raw in data:
        v    = int(raw)
        tick = v & 0xFFFFFFFFFFFFF
        gen  = (v >> 52) & 0x7FF
        flag = (v >> 63) & 1
        if flag:
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


def _build_pps_chain(
    all_pps: list[tuple[int, int]],
    f0: int,
    tau: float,
) -> list[_Interval]:
    """
    Build PPS interval chain from a list of (tick, gen) pairs.
    Returns intervals sorted by c0.
    """
    if len(all_pps) < 2:
        return []

    intervals: list[_Interval] = []
    c_last, _ = all_pps[0]
    n_last = 0

    for c_next, _ in all_pps[1:]:
        dc = c_next - c_last
        if dc <= 0:
            log.warning('PPS tick not monotonic (dc=%d); skipping', dc)
            continue
        n = round(dc / f0)
        if n == 0:
            log.warning(
                'Two PPS records within one f0 period; skipping'
            )
            continue
        res = abs(dc - n * f0) / (n * f0)
        trusted = res <= tau
        n_next = n_last + n
        iv = _Interval(c_last, c_next, n_last, n_next, dc, n, trusted)
        intervals.append(iv)
        if not trusted:
            log.warning(
                'PPS residual %.2e > tau=%.0e at N~%d — '
                'interval untrusted',
                res, tau, n_next,
            )
        elif n > 1:
            log.warning(
                '%d dropped PPS pulses between N=%d and N=%d',
                n - 1, n_last, n_next,
            )
        c_last = c_next
        n_last = n_next

    return intervals


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

    # Clock frequency — present in synthetic headers, optional elsewhere
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
    utc0: datetime = tm2['timeR']
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


# ── main entry point ─────────────────────────────────────────────

def reconstruct(
    gps_paths: list[Path],
    pos_paths: list[Path],
    utc0: datetime,
    f0: int = F0_DEFAULT,
    tau: float = PPS_TAU,
) -> tuple[list[TimedEvent], dict[int, PosRef]]:
    """
    Stage 1: reconstruct UTC timestamps for all events in one
    detector's run.

    Parameters
    ----------
    gps_paths : *_GPS.bin files in acquisition order
    pos_paths : *.bin files in the same order
    utc0      : UTC time of the first PPS (from load_header_params)
    f0        : nominal clock frequency in Hz
    tau       : PPS residual acceptance threshold (default 1e-4)

    Returns
    -------
    events  : list of TimedEvent in time order (monotonic t_ns)
    pos_map : {evt_seq: PosRef} for O(1) stage-3 random access
    """
    if len(gps_paths) != len(pos_paths):
        raise ValueError(
            'gps_paths and pos_paths must have the same length'
        )

    utc0_ns = _utc_to_ns(utc0)

    # ── pass 1: collect all records across files ─────────────────
    # Entries: (tick, file_idx, local_event_idx)
    all_evt: list[tuple[int, int, int]] = []
    all_pps: list[tuple[int, int]] = []    # (tick, gen)
    pos_map: dict[int, PosRef] = {}

    evt_seq     = 0
    prev_gen:   int | None = None   # GEN of last pos-file row
    prev_file:  int | None = None

    for file_idx, (gps_path, pos_path) in enumerate(
        zip(gps_paths, pos_paths)
    ):
        gps_path = Path(gps_path)
        pos_path = Path(pos_path)

        evt_ticks, evt_gens, pps_ticks, pps_gens = (
            _parse_gps_file(gps_path)
        )
        n_gps_events = len(evt_ticks)

        nr, nc, first_gen, last_gen = _pos_file_meta(pos_path)

        # Validate row count
        if nr % 16 != 0:
            log.warning(
                '%s: row count %d not multiple of 16'
                ' (possible split block at file end)',
                pos_path.name, nr,
            )
        n_pos_events = nr // 16
        if n_pos_events != n_gps_events:
            raise ValueError(
                f'{pos_path.name}: {n_pos_events} position blocks'
                f' != {n_gps_events} GPS event records'
            )

        # Check GEN continuity at file boundary
        if prev_gen is not None and nr > 0:
            expected = (prev_gen + 1) % 2048
            if first_gen == prev_gen:
                # Same GEN means the previous file's last block
                # was split; the first rows here are its tail.
                tail = nr % 16 if (prev_file is not None) else 0
                if tail:
                    log.warning(
                        '%s: split block detected — last %d rows'
                        ' of file %d and first %d rows here'
                        ' form one block',
                        pos_path.name, tail,
                        prev_file, 16 - tail,
                    )
                    # Patch the PosRef written for the event that
                    # owns the split block's first half.
                    split_seq = evt_seq - 1
                    old = pos_map[split_seq]
                    pos_map[split_seq] = PosRef(
                        old.file_idx, old.row_offset,
                        split_rows=tail,
                    )
            elif first_gen != expected:
                log.warning(
                    '%s: GEN discontinuity at boundary'
                    ' — expected %d, got %d',
                    pos_path.name, expected, first_gen,
                )

        # Record events and build pos_map
        for local_idx in range(n_gps_events):
            all_evt.append((evt_ticks[local_idx], file_idx, local_idx))
            pos_map[evt_seq] = PosRef(file_idx, local_idx * 16)
            evt_seq += 1

        all_pps.extend(zip(pps_ticks, pps_gens))

        prev_gen  = last_gen if nr > 0 else prev_gen
        prev_file = file_idx

    if not all_pps:
        raise ValueError(
            'No PPS records found — cannot anchor timestamps'
        )

    # ── pass 2: build PPS anchor chain ───────────────────────────
    c0 = all_pps[0][0]   # tick of first PPS = UTC₀ anchor
    intervals = _build_pps_chain(all_pps, f0, tau)
    c0s = [iv.c0 for iv in intervals]

    trusted   = [iv for iv in intervals if iv.trusted]
    back_iv   = trusted[0]  if trusted else None
    fwd_iv    = trusted[-1] if trusted else None

    if not trusted:
        log.warning(
            'No trusted PPS intervals — all events will be DEGRADED'
        )
    log.debug(
        'PPS chain: %d intervals, %d trusted, c0=%d',
        len(intervals), len(trusted), c0,
    )

    # ── pass 3: timestamp each event ─────────────────────────────
    events: list[TimedEvent] = []
    for seq, (tick, _file_idx, _local) in enumerate(all_evt):
        t_ns, quality = _timestamp(
            tick, intervals, c0s, c0,
            utc0_ns, f0, back_iv, fwd_iv,
        )
        events.append(TimedEvent(t_ns, seq, quality))

    # Sanity check: timestamps must be non-decreasing
    for i in range(1, len(events)):
        if events[i].t_ns < events[i - 1].t_ns:
            log.warning(
                'Non-monotonic timestamp at evt_seq=%d:'
                ' t_ns=%d < prev=%d',
                i, events[i].t_ns, events[i - 1].t_ns,
            )

    return events, pos_map
