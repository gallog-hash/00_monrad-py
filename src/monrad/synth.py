"""
Synthetic BuS_Tracker dataset generator.

Produces telescope and probe binary files that exercise the
full pipeline (stages 1-5) without touching real data files.
Call generate() to write the files and get the track data back.
"""

import struct
from datetime import datetime
from pathlib import Path

import numpy as np

GPS_EPOCH = datetime(1980, 1, 6)
F0 = 100_000_000        # Hz — nominal 100 MHz clock
STRIP_MM = 10.0         # strip pitch (mm)
N_TEL = 99              # telescope channels per axis
N_PROBE_DEFAULT = 30    # probe channels per axis
# Telescope plane z-coordinates (mm); lever arm = 800 mm
Z_TEL = np.array([0.0, 400.0, 800.0])


# ── binary helpers ──────────────────────────────────────────────

def _make_ubx_tm2(utc: datetime, acc_ns: int = 30) -> bytes:
    """Return a 36-byte UBX-TIM-TM2 frame for the given UTC time."""
    delta = utc - GPS_EPOCH
    total_ms = int(delta.total_seconds() * 1000)
    week, tow_ms = divmod(total_ms, 7 * 24 * 3600 * 1000)
    payload = struct.pack(
        '<BBHHHIIIII',
        0,           # ch = 0 (TIMEPULSE)
        0x0F,        # flags
        1,           # rising-edge count
        week, week,
        tow_ms, 0,
        tow_ms + 500, 0,
        acc_ns,
    )
    assert len(payload) == 28
    hdr = bytes([0xB5, 0x62, 0x0D, 0x03]) + struct.pack('<H', 28)
    ck_a = ck_b = 0
    for b in hdr[2:] + payload:
        ck_a = (ck_a + b) & 0xFF
        ck_b = (ck_b + ck_a) & 0xFF
    return hdr + payload + bytes([ck_a, ck_b])


def _escape(data: bytes) -> str:
    """Encode bytes for header.txt GPS_String field."""
    out = []
    for b in data:
        if b == 0x5C:
            out.append('\\\\')
        elif 0x20 <= b <= 0x7E:
            out.append(chr(b))
        else:
            out.append(f'\\{b:02X}')
    return ''.join(out)


def _write_header(path: Path, utc: datetime, f0: int) -> None:
    ubx = _make_ubx_tm2(utc)
    with open(path, 'w', encoding='latin-1') as fh:
        fh.write('[System]\n')
        fh.write(f'Clock frequency (Hz) = {f0}\n')
        fh.write('\n[GPS]\n')
        fh.write(f'GPS_String_00 = "{_escape(ubx)}"\n')


def _write_gps_bin(path: Path, records: list[int]) -> None:
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<I', len(records)))
        for r in records:
            fh.write(struct.pack('<Q', r))


def _write_pos_bin(
    path: Path,
    blocks: list[list[int]],
    n_cols: int,
) -> None:
    """
    Write *.bin.  Each block is one event: a list of n_cols u64
    words (one per plane).  The block is replicated 16× as rows.
    """
    n_rows = len(blocks) * 16
    with open(path, 'wb') as fh:
        fh.write(struct.pack('<I', n_rows))
        fh.write(struct.pack('<I', n_cols))
        for words in blocks:
            for _ in range(16):
                for w in words:
                    fh.write(struct.pack('<Q', w))


# ── encoding ────────────────────────────────────────────────────

def _ch_to_u64(c_x: int, c_y: int, gen: int) -> int:
    """Encode a golden hit (single channel per axis) as a u64 word."""
    # ch = 10 * ribbon_bit + fiber_bit
    y_rib = 1 << (c_y // 10)
    y_fib = 1 << (c_y % 10)
    x_rib = 1 << (c_x // 10)
    x_fib = 1 << (c_x % 10)
    return (
        y_rib
        | (y_fib << 10)
        | (x_rib << 32)
        | (x_fib << 42)
        | (gen << 52)
    )


def _gps_rec(tick: int, gen: int, pps: bool) -> int:
    return tick | (gen << 52) | ((1 << 63) if pps else 0)


def _build_gps_stream(
    event_ticks: list[int],
    pps_ticks: list[int],
) -> list[int]:
    """
    Merge event and PPS records in tick order.
    PPS records use the GEN value current at the pulse time
    (i.e. after all events that precede the pulse).
    """
    records: list[int] = []
    gen = 0
    pi = 0
    for tick in event_ticks:
        while pi < len(pps_ticks) and pps_ticks[pi] < tick:
            records.append(_gps_rec(pps_ticks[pi], gen, True))
            pi += 1
        records.append(_gps_rec(tick, gen, False))
        gen = (gen + 1) % 2048
    while pi < len(pps_ticks):
        records.append(_gps_rec(pps_ticks[pi], gen, True))
        pi += 1
    return records


# ── track generation ─────────────────────────────────────────────

def _quantize(coord_mm: float, n_ch: int) -> int | None:
    ch = int(coord_mm / STRIP_MM)
    return ch if 0 <= ch < n_ch else None


def _sample_tracks(
    n: int,
    rng: np.random.Generator,
) -> list[tuple[float, float, float, float]]:
    """
    Sample n muon tracks that cross all 3 telescope planes.

    Returns list of (a_x, b_x, a_y, b_y) in mm / (mm/mm),
    where x(z) = a_x + b_x*z.

    Angular distribution: I ∝ cos²θ (standard muon approximation).
    Entry positions uniform over the active area at z=0.
    """
    active = N_TEL * STRIP_MM   # 990 mm
    z_bot = Z_TEL[-1]           # 800 mm
    tracks: list[tuple] = []
    while len(tracks) < n:
        batch = max(n * 5, 2000)
        x0 = rng.uniform(0, active, batch)
        y0 = rng.uniform(0, active, batch)
        # Inverse CDF for cos²θ: θ = arccos((1-u)^(1/3))
        u = rng.uniform(0, 1, batch)
        zen = np.arccos(np.cbrt(1.0 - u))
        phi = rng.uniform(0, 2 * np.pi, batch)
        bx = np.tan(zen) * np.cos(phi)
        by = np.tan(zen) * np.sin(phi)
        x_b = x0 + bx * z_bot
        y_b = y0 + by * z_bot
        ok = (
            (x_b >= 0) & (x_b < active)
            & (y_b >= 0) & (y_b < active)
        )
        for i in np.where(ok)[0]:
            if len(tracks) >= n:
                break
            tracks.append(
                (float(x0[i]), float(bx[i]),
                 float(y0[i]), float(by[i]))
            )
    return tracks[:n]


# ── public API ───────────────────────────────────────────────────

def generate(
    out_dir: str | Path,
    t_x: float = 50.0,
    t_y: float = -30.0,
    theta: float = 0.29671,   # radians (≈ 17°)
    z_p: float = 300.0,
    n_probe_ch: int = N_PROBE_DEFAULT,
    n_tracks: int = 1000,
    seed: int = 42,
    start_utc: datetime = datetime(2023, 4, 18, 19, 21, 0),
    f0: int = F0,
) -> dict:
    """
    Write synthetic telescope + probe files to out_dir.

    Parameters are in mm and radians.  Default pose:
      t_x=50 mm, t_y=-30 mm, theta=17°, z_p=300 mm.

    Returns a dict with keys:
      tracks          list of (a_x, b_x, a_y, b_y)
      probe_hits      dict {track_index: (c_u, c_v)}
      n_coincidences  int
      tel_dir, probe_dir  Path
      pose            (t_x, t_y, theta, z_p)
    """
    out_dir = Path(out_dir)
    tel_dir = out_dir / 'telescope'
    prb_dir = out_dir / 'probe'
    tel_dir.mkdir(parents=True, exist_ok=True)
    prb_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    tracks = _sample_tracks(n_tracks, rng)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # ── probe hits ───────────────────────────────────────────────
    probe_hits: dict[int, tuple[int, int]] = {}
    for i, (ax, bx, ay, by) in enumerate(tracks):
        xp = ax + bx * z_p
        yp = ay + by * z_p
        u = (xp - t_x) * cos_t + (yp - t_y) * sin_t
        v = -(xp - t_x) * sin_t + (yp - t_y) * cos_t
        cu = _quantize(u, n_probe_ch)
        cv = _quantize(v, n_probe_ch)
        if cu is not None and cv is not None:
            probe_hits[i] = (cu, cv)

    # ── timing ───────────────────────────────────────────────────
    dt = f0 // 10           # 10 Hz telescope rate
    t_off = dt // 2         # offset events away from PPS ticks
    tel_ticks = [t_off + i * dt for i in range(n_tracks)]
    duration = tel_ticks[-1] + dt
    pps_ticks = list(range(0, duration + f0, f0))

    # ── telescope GPS stream ─────────────────────────────────────
    tel_gps = _build_gps_stream(tel_ticks, pps_ticks)

    # ── telescope position blocks ────────────────────────────────
    gen = 0
    tel_blocks: list[list[int]] = []
    for ax, bx, ay, by in tracks:
        words = []
        for z in Z_TEL:
            cx = _quantize(ax + bx * z, N_TEL)
            cy = _quantize(ay + by * z, N_TEL)
            assert cx is not None and cy is not None, (
                f"Track out of range at z={z}: "
                f"x={ax+bx*z:.1f}, y={ay+by*z:.1f}"
            )
            words.append(_ch_to_u64(cx, cy, gen))
        tel_blocks.append(words)
        gen = (gen + 1) % 2048

    # ── probe GPS stream ─────────────────────────────────────────
    coinc_idx = sorted(probe_hits)
    prb_ticks = [tel_ticks[i] for i in coinc_idx]
    prb_gps = _build_gps_stream(prb_ticks, pps_ticks)

    # ── probe position blocks ────────────────────────────────────
    gen = 0
    prb_blocks: list[list[int]] = []
    for i in coinc_idx:
        cu, cv = probe_hits[i]
        prb_blocks.append([_ch_to_u64(cu, cv, gen)])
        gen = (gen + 1) % 2048

    # ── write files ──────────────────────────────────────────────
    ts = start_utc.strftime('%Y%m%d_%H%M%S')

    _write_header(tel_dir / f'{ts}_header.txt', start_utc, f0)
    _write_gps_bin(tel_dir / f'{ts}_GPS.bin', tel_gps)
    _write_pos_bin(tel_dir / f'{ts}.bin', tel_blocks, n_cols=3)

    _write_header(prb_dir / f'{ts}_header.txt', start_utc, f0)
    _write_gps_bin(prb_dir / f'{ts}_GPS.bin', prb_gps)
    _write_pos_bin(prb_dir / f'{ts}.bin', prb_blocks, n_cols=1)

    return {
        'tracks': tracks,
        'probe_hits': probe_hits,
        'n_coincidences': len(probe_hits),
        'tel_dir': tel_dir,
        'probe_dir': prb_dir,
        'pose': (t_x, t_y, theta, z_p),
    }
