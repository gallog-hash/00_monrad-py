"""
Tests for stage 3 — position decoding.

Decodes every telescope event from the synthetic dataset and verifies
that hits are 'golden', coordinates match ground truth, and sigmas are
correct.
"""

import math
import pytest
from datetime import datetime

from monrad.stage1 import (
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.stage3 import decode_position
from monrad.synth import generate, F0, Z_TEL, STRIP_MM

_START_UTC  = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS   = 1000
_N_COLS     = 3       # telescope planes
_SIGMA_GOLD = STRIP_MM / math.sqrt(12)


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp('synth_stage3')
    result = generate(
        out_dir=out,
        t_x=50.0, t_y=-30.0,
        theta=0.29671, z_p=300.0,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
    )
    return result, out


@pytest.fixture(scope='module')
def tel_decoded(synth):
    """
    Run stage-1 on the telescope data, then decode position for every
    event.  Returns (decoded, tracks):
      decoded : list of list[Hit] — one inner list (3 Hits) per event
      tracks  : list of (a_x, b_x, a_y, b_y) ground-truth slopes
    """
    result, out = synth
    tel_dir = out / 'telescope'

    utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
    gps_paths, pos_paths = find_file_pairs(tel_dir)

    decoded = []
    for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
        hits = decode_position(ref, pos_paths, n_cols=_N_COLS)
        decoded.append(hits)

    return decoded, result['tracks']


# ── tests ─────────────────────────────────────────────────────────────

class TestDecodePosition:

    def test_event_count(self, tel_decoded):
        decoded, _ = tel_decoded
        assert len(decoded) == _N_TRACKS

    def test_hits_per_event(self, tel_decoded):
        decoded, _ = tel_decoded
        for i, hits in enumerate(decoded):
            assert len(hits) == _N_COLS, (
                f'event {i}: expected {_N_COLS} hits, got {len(hits)}'
            )

    def test_all_golden(self, tel_decoded):
        decoded, _ = tel_decoded
        for i, hits in enumerate(decoded):
            for k, hit in enumerate(hits):
                assert hit.quality == 'golden', (
                    f'event {i} plane {k}: quality={hit.quality!r}, '
                    f'expected "golden"'
                )

    def test_sigma(self, tel_decoded):
        decoded, _ = tel_decoded
        for i, hits in enumerate(decoded):
            for k, hit in enumerate(hits):
                assert hit.sigma_x == pytest.approx(_SIGMA_GOLD), (
                    f'event {i} plane {k}: sigma_x={hit.sigma_x}'
                )
                assert hit.sigma_y == pytest.approx(_SIGMA_GOLD), (
                    f'event {i} plane {k}: sigma_y={hit.sigma_y}'
                )

    def test_coordinates(self, tel_decoded):
        """
        For event i, plane k the ground-truth channel is:
          cx = int((a_x + b_x * Z_TEL[k]) / STRIP_MM)
        and the decoded coordinate must equal (cx + 0.5) * STRIP_MM.
        """
        decoded, tracks = tel_decoded
        for i, (hits, (ax, bx, ay, by)) in enumerate(
            zip(decoded, tracks)
        ):
            for k, (hit, z) in enumerate(zip(hits, Z_TEL)):
                cx = int((ax + bx * z) / STRIP_MM)
                cy = int((ay + by * z) / STRIP_MM)
                exp_x = (cx + 0.5) * STRIP_MM
                exp_y = (cy + 0.5) * STRIP_MM
                assert hit.x_mm == pytest.approx(exp_x), (
                    f'event {i} plane {k}: x_mm={hit.x_mm}, '
                    f'expected {exp_x} (cx={cx})'
                )
                assert hit.y_mm == pytest.approx(exp_y), (
                    f'event {i} plane {k}: y_mm={hit.y_mm}, '
                    f'expected {exp_y} (cy={cy})'
                )
