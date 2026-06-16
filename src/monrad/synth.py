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
F0 = 100_000_000  # Hz — nominal 100 MHz clock
STRIP_MM = 10.0  # strip pitch (mm)
N_TEL = 99  # telescope channels per axis
N_PROBE_DEFAULT = 30  # probe channels per axis
# Telescope plane z-coordinates (mm); lever arm = 800 mm
Z_TEL = np.array([0.0, 400.0, 800.0])


# ── binary helpers ──────────────────────────────────────────────


def _make_ubx_tm2(utc: datetime, acc_ns: int = 30) -> bytes:
    """Return a 36-byte UBX-TIM-TM2 frame for the given UTC time."""
    delta = utc - GPS_EPOCH
    total_ms = int(delta.total_seconds() * 1000)
    week, tow_ms = divmod(total_ms, 7 * 24 * 3600 * 1000)
    payload = struct.pack(
        "<BBHHHIIIII",
        0,  # ch = 0 (TIMEPULSE)
        0x0F,  # flags
        1,  # rising-edge count
        week,
        week,
        tow_ms,
        0,
        tow_ms + 500,
        0,
        acc_ns,
    )
    assert len(payload) == 28
    hdr = bytes([0xB5, 0x62, 0x0D, 0x03]) + struct.pack("<H", 28)
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
            out.append("\\\\")
        elif 0x20 <= b <= 0x7E:
            out.append(chr(b))
        else:
            out.append(f"\\{b:02X}")
    return "".join(out)


def _write_header(path: Path, utc: datetime, f0: int) -> None:
    ubx = _make_ubx_tm2(utc)
    with open(path, "w", encoding="latin-1") as fh:
        fh.write("[System]\n")
        fh.write(f"Clock frequency (Hz) = {f0}\n")
        fh.write("\n[GPS]\n")
        fh.write(f'GPS_String_00 = "{_escape(ubx)}"\n')


def _write_gps_bin(path: Path, records: list[int]) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", len(records)))
        for r in records:
            fh.write(struct.pack("<Q", r))


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
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", n_rows))
        fh.write(struct.pack("<I", n_cols))
        for words in blocks:
            for _ in range(16):
                for w in words:
                    fh.write(struct.pack("<Q", w))


# ── encoding ────────────────────────────────────────────────────


def _cluster_fiber_mask(f: int, width: int) -> int:
    """Return a fiber mask of `width` contiguous bits whose centroid sits
    as close as possible to bit `f`, kept within a single ribbon [0, 9].

    width=1 reproduces the golden single-bit mask.  For width W>1 the
    combined-coordinate candidates 10*r + (f_start … f_start+W-1) form a
    contiguous range, so the axis decodes as a `cluster` with
    σ = STRIP_MM*W/√12 (decoded centroid channel = f_start + (W-1)/2).
    """
    if not 1 <= width <= 10:
        raise ValueError(f"cluster width must be in 1..10, got {width}")
    f_start = f - (width - 1) // 2
    f_start = max(0, min(10 - width, f_start))
    return ((1 << width) - 1) << f_start


def _ch_to_u64(
    c_x: int,
    c_y: int,
    gen: int,
    fold: bool = False,
    width_x: int = 1,
    width_y: int = 1,
) -> int:
    """Encode a hit as a u64 word.

    fold=False (default): golden hit — single fiber bit + single ribbon
    bit per axis, exactly as the real pipeline expects from clean data.

    fold=True: folded-fiber encoding — each axis fires both bit k and
    bit (9-k) in both the fiber and ribbon halves, mimicking the real
    telescope MAROC wiring.  Mirror-pair patterns are reported as
    diagnostics by or_visual() but are not resolved in reconstruction.

    width_x, width_y: per-axis cluster widths (channels).  width=1 is a
    golden hit; width>1 fires that many contiguous fiber bits so the axis
    decodes as a `cluster` of that width (σ = STRIP_MM*width/√12), letting
    a single plane carry a different σ on each axis.  Ignored when fold is
    True (the two encodings are mutually exclusive).
    """
    # ch = 10 * ribbon_bit + fiber_bit
    r_y, f_y = c_y // 10, c_y % 10
    r_x, f_x = c_x // 10, c_x % 10
    if fold:
        y_rib = (1 << r_y) | (1 << (9 - r_y))
        y_fib = (1 << f_y) | (1 << (9 - f_y))
        x_rib = (1 << r_x) | (1 << (9 - r_x))
        x_fib = (1 << f_x) | (1 << (9 - f_x))
    else:
        y_rib = 1 << r_y
        y_fib = _cluster_fiber_mask(f_y, width_y)
        x_rib = 1 << r_x
        x_fib = _cluster_fiber_mask(f_x, width_x)
    return y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)


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
    active = N_TEL * STRIP_MM  # 990 mm
    z_bot = Z_TEL[-1]  # 800 mm
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
        ok = (x_b >= 0) & (x_b < active) & (y_b >= 0) & (y_b < active)
        for i in np.where(ok)[0]:
            if len(tracks) >= n:
                break
            tracks.append((float(x0[i]), float(bx[i]), float(y0[i]), float(by[i])))
    return tracks[:n]


# ── public API ───────────────────────────────────────────────────


def _quantize_clamp(coord_mm: float, n_ch: int) -> int:
    """Quantize to strip channel, clamping to [0, n_ch-1]."""
    return max(0, min(n_ch - 1, int(coord_mm / STRIP_MM)))


def generate(
    out_dir: str | Path,
    t_x: float = 50.0,
    t_y: float = -30.0,
    theta: float = 0.29671,  # radians (≈ 17°)
    z_p: float = 300.0,
    n_probe_ch: int = N_PROBE_DEFAULT,
    n_tracks: int = 1000,
    seed: int = 42,
    start_utc: datetime = datetime(2023, 4, 18, 19, 21, 0),
    f0: int = F0,
    plane_offsets: dict[int, tuple[float, float]] | None = None,
    fold: bool = False,
    z_tel_offsets: dict[int, float] | None = None,
    z_tel_tilts: dict[int, tuple[float, float]] | None = None,
    tel_cluster_widths: dict[int, tuple[int, int]] | None = None,
    probe_cluster_width: tuple[int, int] | None = None,
) -> dict:
    """
    Write synthetic telescope + probe files to out_dir.

    Parameters are in mm and radians.  Default pose:
      t_x=50 mm, t_y=-30 mm, theta=17°, z_p=300 mm.

    plane_offsets  : optional dict {plane_idx: (dx_mm, dy_mm)} applied to
                     telescope hit coordinates before quantization.
    fold           : if True, encode telescope hits with both fold-pair
                     bits set (k and 9-k) in each mask half, mimicking
                     folded-fiber MAROC wiring.  fold-pair decoding should
                     recover the original channels.
    z_tel_offsets  : optional dict {plane_idx: dz_mm} — the telescope
                     *.bin file is written with hits computed at
                     Z_TEL[k] + dz, but the header still records the
                     nominal Z_TEL.  Used to test stage-4 Z-correction.
    z_tel_tilts    : optional dict {plane_idx: (tilt_x_rad, tilt_y_rad)} —
                     give that plane a small out-of-plane tilt so it is no
                     longer parallel to the others.  tilt_y (about the
                     y-axis) displaces the x hit by tilt_y·b_x·x, tilt_x
                     (about the x-axis) displaces the y hit by tilt_x·b_y·y,
                     reproducing the slope×lever-arm residual of DESIGN.md
                     §7.3.  Used to test stage-4 tilt detection.
    tel_cluster_widths : optional dict {plane_idx: (width_x, width_y)} —
                     encode that telescope plane's hit as a `cluster` of
                     the given per-axis width (channels) instead of a
                     golden hit, so the plane carries σ = STRIP_MM*width/√12
                     that can differ per axis and per plane.  Planes absent
                     from the dict stay golden (width 1).  Ignored when fold
                     is True.
    probe_cluster_width : optional (width_x, width_y) — same idea for the
                     probe plane's hit.

    Returns a dict with keys:
      tracks          list of (a_x, b_x, a_y, b_y)
      probe_hits      dict {track_index: (c_u, c_v)}
      n_coincidences  int
      tel_dir, probe_dir  Path
      pose            (t_x, t_y, theta, z_p)
      plane_offsets   dict as passed in (or {})
      z_tel_offsets   dict as passed in (or {})
    """
    if plane_offsets is None:
        plane_offsets = {}
    if z_tel_offsets is None:
        z_tel_offsets = {}
    if z_tel_tilts is None:
        z_tel_tilts = {}
    if tel_cluster_widths is None:
        tel_cluster_widths = {}
    out_dir = Path(out_dir)
    tel_dir = out_dir / "telescope"
    prb_dir = out_dir / "probe"
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
    dt = f0 // 10  # 10 Hz telescope rate
    t_off = dt // 2  # offset events away from PPS ticks
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
        for k, z_nom in enumerate(Z_TEL):
            # Hits are placed at the true z (z_nom + dz) but the pipeline
            # is told the nominal z — this lets stage 4 detect the offset.
            z = z_nom + z_tel_offsets.get(k, 0.0)
            off_x, off_y = plane_offsets.get(k, (0.0, 0.0))
            tilt_x, tilt_y = z_tel_tilts.get(k, (0.0, 0.0))
            # Nominal intersection of the track with this plane.
            x_base = ax + bx * z
            y_base = ay + by * z
            # A tilt about the y-axis tips the plane in x-z, so its x reading
            # is displaced by tilt_y·(track x-slope)·(lever arm) — and likewise
            # tilt_x for y.  This is the slope×position residual of §7.3.
            x_coord = x_base + off_x + tilt_y * bx * x_base
            y_coord = y_base + off_y + tilt_x * by * y_base
            if off_x == 0.0 and off_y == 0.0 and tilt_x == 0.0 and tilt_y == 0.0:
                cx = _quantize(x_coord, N_TEL)
                cy = _quantize(y_coord, N_TEL)
                assert cx is not None and cy is not None, (
                    f"Track out of range at z={z}: x={x_coord:.1f}, y={y_coord:.1f}"
                )
            else:
                cx = _quantize_clamp(x_coord, N_TEL)
                cy = _quantize_clamp(y_coord, N_TEL)
            wx, wy = tel_cluster_widths.get(k, (1, 1))
            words.append(_ch_to_u64(cx, cy, gen, fold=fold, width_x=wx, width_y=wy))
        tel_blocks.append(words)
        gen = (gen + 1) % 2048

    # ── probe GPS stream ─────────────────────────────────────────
    coinc_idx = sorted(probe_hits)
    prb_ticks = [tel_ticks[i] for i in coinc_idx]
    prb_gps = _build_gps_stream(prb_ticks, pps_ticks)

    # ── probe position blocks ────────────────────────────────────
    gen = 0
    prb_wx, prb_wy = probe_cluster_width or (1, 1)
    prb_blocks: list[list[int]] = []
    for i in coinc_idx:
        cu, cv = probe_hits[i]
        prb_blocks.append([_ch_to_u64(cu, cv, gen, width_x=prb_wx, width_y=prb_wy)])
        gen = (gen + 1) % 2048

    # ── write files ──────────────────────────────────────────────
    ts = start_utc.strftime("%Y%m%d_%H%M%S")

    _write_header(tel_dir / f"{ts}_header.txt", start_utc, f0)
    _write_gps_bin(tel_dir / f"{ts}_GPS.bin", tel_gps)
    _write_pos_bin(tel_dir / f"{ts}.bin", tel_blocks, n_cols=3)

    _write_header(prb_dir / f"{ts}_header.txt", start_utc, f0)
    _write_gps_bin(prb_dir / f"{ts}_GPS.bin", prb_gps)
    _write_pos_bin(prb_dir / f"{ts}.bin", prb_blocks, n_cols=1)

    return {
        "tracks": tracks,
        "probe_hits": probe_hits,
        "n_coincidences": len(probe_hits),
        "tel_dir": tel_dir,
        "probe_dir": prb_dir,
        "pose": (t_x, t_y, theta, z_p),
        "plane_offsets": plane_offsets,
        "z_tel_offsets": z_tel_offsets,
        "z_tel_tilts": z_tel_tilts,
        "tel_cluster_widths": tel_cluster_widths,
        "probe_cluster_width": probe_cluster_width,
    }
