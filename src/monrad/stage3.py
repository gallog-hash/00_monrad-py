"""
Stage 3 — position decoding.

Public API
----------
decode_position(pos_ref, pos_paths, n_cols)
    -> list[Hit | None]   one entry per column (detector plane)

Hit
    NamedTuple: (x_mm, y_mm, sigma_x, sigma_y, quality)
    quality: 'golden' | 'cluster' | 'unresolved' | 'invalid'
"""

import math
import struct
from pathlib import Path
from typing import Literal, NamedTuple

from .stage1 import PosRef
from .decoders.position import BinDecoder

_STRIP_MM = 10.0      # mm per channel strip — DESIGN.md §5.4
_N = 10               # fiber × ribbon encoding multiplier


class Hit(NamedTuple):
    x_mm:    float
    y_mm:    float
    sigma_x: float
    sigma_y: float
    quality: Literal['golden', 'cluster', 'unresolved', 'invalid']


def _read_block(
    pos_paths: list[Path],
    pos_ref:   PosRef,
    n_cols:    int,
) -> list[int]:
    """
    Read the 16 × n_cols u64 words for one event, handling split blocks.

    Returns a flat list of 16 * n_cols ints in row-major order:
      [col0_row0, col1_row0, …, col(n-1)_row0, col0_row1, …]
    """
    _HDR  = 8   # 4-byte n_rows + 4-byte n_cols
    _WORD = 8   # bytes per u64

    def _rows(path: Path, row_start: int, n_rows: int) -> list[int]:
        offset = _HDR + row_start * n_cols * _WORD
        with open(path, 'rb') as fh:
            fh.seek(offset)
            raw = fh.read(n_rows * n_cols * _WORD)
        return [
            struct.unpack_from('<Q', raw, i)[0]
            for i in range(0, len(raw), _WORD)
        ]

    if pos_ref.split_rows == 0:
        return _rows(pos_paths[pos_ref.file_idx], pos_ref.row_offset, 16)

    head = _rows(
        pos_paths[pos_ref.file_idx],
        pos_ref.row_offset,
        pos_ref.split_rows,
    )
    tail = _rows(
        pos_paths[pos_ref.file_idx + 1],
        0,
        16 - pos_ref.split_rows,
    )
    return head + tail


def _decode_axis(
    field_or: int,
) -> tuple[float, float, Literal['golden', 'cluster', 'unresolved']]:
    """
    Decode one 20-bit fiber×ribbon field (already extracted from u64).

    bits  0– 9: ribbon mask
    bits 10–19: fiber mask

    Returns (centroid_ch, sigma_mm, quality).
    Implements DESIGN.md §5.3 steps 1–5.
    """
    fiber_half  = (field_or >> _N) & 0x3FF
    ribbon_half =  field_or        & 0x3FF

    fcs = BinDecoder._find_clusters(fiber_half)
    rcs = BinDecoder._find_clusters(ribbon_half)

    res = BinDecoder._reconstruct_coord(fcs, rcs, _N)
    if res is None:
        return 0.0, 0.0, 'unresolved'

    centroid, candidates = res
    width = len(candidates)
    sigma = (_STRIP_MM * width) / math.sqrt(12)
    quality = 'golden' if width == 1 else 'cluster'
    return centroid, sigma, quality


def decode_position(
    pos_ref:   PosRef,
    pos_paths: list[Path],
    n_cols:    int,
) -> list[Hit | None]:
    """
    Decode one event's position from its PosRef.

    Implements DESIGN.md §5.1–§5.4 using the fiber×ribbon logic from
    decoders/position.py (BinDecoder._find_clusters, _reconstruct_coord,
    _is_valid).  The PosRef is received directly from the stage-1 stream
    (DESIGN_UPDATE.md §4) — no pos_map lookup.

    Parameters
    ----------
    pos_ref   : location of the event's 16-row block on disk
    pos_paths : *.bin file paths for this detector, in acquisition
                order (indexed by pos_ref.file_idx)
    n_cols    : number of position-sensitive planes in the detector
                (1 for a probe, 3 for the telescope)

    Returns
    -------
    list of length n_cols.  Each element is a Hit (always present,
    with quality 'invalid' or 'unresolved' when reconstruction fails)
    so callers can use a plain membership test on quality strings
    rather than checking for None.
    """
    words = _read_block(pos_paths, pos_ref, n_cols)

    hits: list[Hit | None] = []
    for col in range(n_cols):
        # Bitwise-OR all 16 rows for this column — DESIGN.md §5.1
        x_or = 0
        y_or = 0
        for row in range(16):
            w = words[row * n_cols + col]
            y_or |= w & 0xFFFFF           # bits  0–19
            x_or |= (w >> 32) & 0xFFFFF   # bits 32–51

        # Validity prefilter — DESIGN.md §5.2
        valid, _ = BinDecoder._is_valid(x_or, y_or)
        if not valid:
            hits.append(Hit(0.0, 0.0, 0.0, 0.0, 'invalid'))
            continue

        cx, sx, qx = _decode_axis(x_or)
        cy, sy, qy = _decode_axis(y_or)

        if qx == 'unresolved' or qy == 'unresolved':
            hits.append(Hit(0.0, 0.0, 0.0, 0.0, 'unresolved'))
            continue

        # Channel → physical coordinate — DESIGN.md §5.4
        x_mm = (cx + 0.5) * _STRIP_MM
        y_mm = (cy + 0.5) * _STRIP_MM
        quality = 'golden' if (qx == 'golden' and qy == 'golden') else 'cluster'
        hits.append(Hit(x_mm, y_mm, sx, sy, quality))

    return hits
