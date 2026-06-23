"""Synthetic test-data generator.

``generate()`` writes a self-consistent ``*.bin`` / ``*_GPS.bin`` / header
triplet for a telescope and probe with known ground-truth pose, used by the
test-suite and the resolution study (DESIGN.md §11).
"""

from .generate import (
    F0 as F0,
    GPS_EPOCH as GPS_EPOCH,
    N_PROBE_DEFAULT as N_PROBE_DEFAULT,
    N_TEL as N_TEL,
    STRIP_MM as STRIP_MM,
    Z_TEL as Z_TEL,
    generate as generate,
    _build_gps_stream as _build_gps_stream,
    _ch_to_u64 as _ch_to_u64,
    _quantize as _quantize,
    _write_gps_bin as _write_gps_bin,
    _write_header as _write_header,
    _write_pos_bin as _write_pos_bin,
)
