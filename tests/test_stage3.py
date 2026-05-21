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
from monrad.stage3 import decode_position, disambiguate_telescope_hits, Hit
from monrad.decoders.position import BinDecoder
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


# ── unfold_mask unit tests ─────────────────────────────────────────────

class TestUnfoldMask:

    def test_single_fold_pair(self):
        # bits 0 and 9 set → unfolded to bit 0
        mask = (1 << 0) | (1 << 9)
        assert BinDecoder._unfold_mask(mask) == (1 << 0)

    def test_middle_fold_pair(self):
        # bits 4 and 5 set → unfolded to bit 4
        mask = (1 << 4) | (1 << 5)
        assert BinDecoder._unfold_mask(mask) == (1 << 4)

    def test_multiple_pairs(self):
        # bits 0,9 and bits 2,7 → unfolded to bits 0 and 2
        mask = (1 << 0) | (1 << 9) | (1 << 2) | (1 << 7)
        assert BinDecoder._unfold_mask(mask) == ((1 << 0) | (1 << 2))

    def test_unpaired_bit_returns_none(self):
        # bit 0 set but not bit 9
        assert BinDecoder._unfold_mask(1 << 0) is None

    def test_empty_mask(self):
        assert BinDecoder._unfold_mask(0) == 0

    def test_all_pairs(self):
        # all five pairs (0,9),(1,8),(2,7),(3,6),(4,5)
        mask = 0b1111111111
        expected = 0b0000011111   # bits 0-4
        assert BinDecoder._unfold_mask(mask) == expected


# ── TOT threshold tests ────────────────────────────────────────────────

class TestTOTThreshold:

    def test_thresh1_same_as_default(self, synth):
        """tot_thresh=1 must be identical to the default behaviour."""
        result, out = synth
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits_default = decode_position(ref, pos_paths, n_cols=_N_COLS)
            hits_thresh1  = decode_position(ref, pos_paths, n_cols=_N_COLS,
                                            tot_thresh=1)
            assert hits_default == hits_thresh1

    def test_thresh16_passes_all_synthetic(self, synth):
        """All 16 rows of a synthetic golden hit carry the same bit, so
        tot_thresh=16 must not degrade the quality."""
        result, out = synth
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        n_bad = 0
        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits = decode_position(ref, pos_paths, n_cols=_N_COLS,
                                   tot_thresh=16)
            for h in hits:
                if h.quality not in ('golden', 'cluster'):
                    n_bad += 1
        assert n_bad == 0, (
            f'{n_bad} golden hits lost with tot_thresh=16'
        )

    def test_tot_weights_golden_unchanged(self, synth):
        """tot_weights=True must not change golden-hit coordinates (width=1,
        weighting only affects width > 1 clusters)."""
        result, out = synth
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            h_plain  = decode_position(ref, pos_paths, n_cols=_N_COLS)
            h_weight = decode_position(ref, pos_paths, n_cols=_N_COLS,
                                       tot_weights=True)
            assert h_plain == h_weight

    def test_tot_weighted_centroid_shifts_cluster(self, tmp_path):
        """A two-bit cluster where one bit fires more rows should pull
        the centroid toward the higher-TOT bit."""
        import struct
        from monrad.synth import _ch_to_u64
        from monrad.stage1 import PosRef

        # Build a minimal *.bin with one event where X has a 2-bit cluster.
        # Channel c_x=23 → ribbon=2, fiber=3.  Cluster = {23, 24}.
        # ribbon bits 2 and 3 both set (adjacent → valid cluster in ribbon).
        # fiber bit 3 only → fiber single bit.
        # ch candidates: 10*2+3=23 and 10*3+3=33  → non-contiguous, fails.
        #
        # Simpler: fiber bits 3 and 4 set (adjacent), ribbon bit 2 set.
        # ch candidates: 10*2+3=23 and 10*2+4=24 (contiguous) → cluster.
        # For the 2nd scenario, encode manually.
        n_cols = 1
        # y: golden (ribbon=1, fiber=1 → ch=11)
        y_rib = 1 << 1
        y_fib = 1 << 1
        # x: cluster over fiber bits 3 and 4 with ribbon bit 2
        # ribbon bit 2 fires in all 16 rows (TOT=16)
        # fiber bit 3 fires in 12 rows, fiber bit 4 fires in 4 rows
        # → centroid should be pulled toward bit 3 (higher TOT)
        x_rib = 1 << 2          # ribbon bit 2
        x_fib_3 = 1 << 3       # fiber bit 3
        x_fib_4 = 1 << 4       # fiber bit 4
        gen = 0

        # Word with both fiber bits set (used for the OR'd result)
        word_full = (
            y_rib | (y_fib << 10)
            | (x_rib << 32) | ((x_fib_3 | x_fib_4) << 42)
            | (gen << 52)
        )
        # Word with only fiber bit 3 (TOT=12 rows)
        word_f3_only = (
            y_rib | (y_fib << 10)
            | (x_rib << 32) | (x_fib_3 << 42)
            | (gen << 52)
        )

        # Rows 0-3: fiber bit 4 fires (TOT=4); rows 0-15: fiber bit 3 fires
        # Total: fiber 3 → TOT=16, fiber 4 → TOT=4
        raw = struct.pack('<I', 16) + struct.pack('<I', 1)
        for row in range(16):
            w = word_full if row < 4 else word_f3_only
            raw += struct.pack('<Q', w)

        bin_path = tmp_path / 'cluster_tot.bin'
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        pos_paths = [bin_path]

        hit_plain  = decode_position(ref, pos_paths, n_cols=1)[0]
        hit_weight = decode_position(ref, pos_paths, n_cols=1,
                                     tot_weights=True)[0]

        # Unweighted centroid: (ch=23 + ch=24) / 2 = 23.5
        assert hit_plain.x_mm == pytest.approx((23.5 + 0.5) * STRIP_MM)
        # TOT-weighted: fiber3 TOT=16, fiber4 TOT=4; ribbon2 TOT=16 both
        # weights = ribbon2*fiber3=256, ribbon2*fiber4=64
        # centroid = (256*23 + 64*24) / (256+64) = (5888+1536)/320 = 23.2
        assert hit_weight.x_mm < hit_plain.x_mm, (
            "TOT-weighted centroid should shift toward the higher-TOT bit"
        )

    def test_thresh_removes_single_row_noise(self, tmp_path):
        """A bit that fires in only 1 of 16 rows is removed by thresh=2."""
        import struct
        from monrad.synth import _ch_to_u64
        from monrad.stage1 import PosRef

        # Build a minimal *.bin with one event:
        # row 0: a clean golden hit (ch=5 in X, ch=3 in Y)
        # rows 1-15: the same golden hit PLUS a spurious noise bit (fiber_X
        #             bit 7) that fires only in row 0.
        n_cols = 1
        clean_word  = _ch_to_u64(5, 3, gen=0)
        # Noise bit 7 in fiber_X (bit 42+7=49 of the u64)
        noisy_word  = clean_word | (1 << (42 + 7))  # fiber_X bit 7

        rows = [noisy_word] + [clean_word] * 15
        raw = struct.pack('<I', 16) + struct.pack('<I', 1)
        for w in rows:
            raw += struct.pack('<Q', w)
        bin_path = tmp_path / 'noise.bin'
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        pos_paths = [bin_path]

        # thresh=1: noise bit survives → may affect quality
        hits1 = decode_position(ref, pos_paths, n_cols=1, tot_thresh=1)
        # thresh=2: noise bit is removed, only the clean golden hit remains
        hits2 = decode_position(ref, pos_paths, n_cols=1, tot_thresh=2)

        assert hits2[0].quality in ('golden', 'cluster'), (
            f'tot_thresh=2 should recover golden hit, got {hits2[0].quality}'
        )
        # The clean hit decodes to x=55mm, y=35mm
        assert hits2[0].x_mm == pytest.approx(5.5 * STRIP_MM)
        assert hits2[0].y_mm == pytest.approx(3.5 * STRIP_MM)


# ── fold-encoded synthetic data ────────────────────────────────────────

@pytest.fixture(scope='module')
def synth_fold(tmp_path_factory):
    out = tmp_path_factory.mktemp('synth_stage3_fold')
    result = generate(
        out_dir=out,
        t_x=50.0, t_y=-30.0,
        theta=0.29671, z_p=300.0,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
        fold=True,   # encode as folded-fiber MAROC wiring
    )
    return result, out


class TestFoldDecoder:
    """Verify that fold-encoded synthetic hits are recovered correctly."""

    def test_fold_decoded_all_good(self, synth_fold):
        """Every fold-encoded hit should decode to 'golden' with fold=True."""
        result, out = synth_fold
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        n_bad = 0
        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits = decode_position(ref, pos_paths, n_cols=_N_COLS,
                                   fold=True)
            for h in hits:
                if h.quality not in ('golden', 'cluster'):
                    n_bad += 1
        assert n_bad == 0, f'{n_bad} hits not decoded with fold=True'

    def test_fold_off_gives_unresolved(self, synth_fold):
        """With fold=False the same data should come back as 'unresolved'."""
        result, out = synth_fold
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        n_unresolved = 0
        n_total = 0
        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits = decode_position(ref, pos_paths, n_cols=_N_COLS,
                                   fold=False)
            for h in hits:
                n_total += 1
                if h.quality == 'unresolved':
                    n_unresolved += 1
        # fold=False on folded data → almost all hits unresolved
        assert n_unresolved / n_total > 0.9, (
            f'Expected >90% unresolved with fold=False on folded data, '
            f'got {n_unresolved}/{n_total}'
        )

    def test_fold_and_tot_combined(self, synth_fold):
        """fold=True + tot_thresh=16 still decodes all synthetic events
        (they fire the same bit in all 16 rows)."""
        result, out = synth_fold
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        n_bad = 0
        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits = decode_position(ref, pos_paths, n_cols=_N_COLS,
                                   fold=True, tot_thresh=16)
            for h in hits:
                if h.quality not in ('golden', 'cluster'):
                    n_bad += 1
        assert n_bad == 0, (
            f'{n_bad} hits failed with fold=True, tot_thresh=16'
        )

    def test_fold_decoded_coordinates(self, synth_fold):
        """Fold-decoded coordinates are consistent with the fold mapping.

        Fold-decoding maps each (ribbon_bit, fiber_bit) pair to
        (min(r, 9-r), min(f, 9-f)), which is NOT the same as the
        original channel for bits > 4.  The test checks against the
        expected fold-decoded value rather than the original channel.
        """
        def _fold_ch(c: int) -> int:
            r = min(c // 10, 9 - c // 10)
            f = min(c % 10, 9 - c % 10)
            return 10 * r + f

        result, out = synth_fold
        tracks = result['tracks']
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        mismatches = 0
        for i, (_ev, ref) in enumerate(
            reconstruct_stream(gps_paths, pos_paths, utc0, f0)
        ):
            hits = decode_position(ref, pos_paths, n_cols=_N_COLS,
                                   fold=True)
            ax, bx, ay, by = tracks[i]
            for k, (hit, z) in enumerate(zip(hits, Z_TEL)):
                cx = _fold_ch(int((ax + bx * z) / STRIP_MM))
                cy = _fold_ch(int((ay + by * z) / STRIP_MM))
                exp_x = (cx + 0.5) * STRIP_MM
                exp_y = (cy + 0.5) * STRIP_MM
                if (abs(hit.x_mm - exp_x) > 1e-6
                        or abs(hit.y_mm - exp_y) > 1e-6):
                    mismatches += 1
        assert mismatches == 0, (
            f'{mismatches} coordinate mismatches in fold-decoded data'
        )


# ── Tests for disambiguate_telescope_hits ─────────────────────────────────

import numpy as np

_Z3 = np.array([0.0, 400.0, 800.0])
_SIGMA = STRIP_MM / math.sqrt(12)


def _h(x, y, quality='golden', cx=None, cy=None):
    return Hit(x, y, _SIGMA, _SIGMA, quality, cx, cy)


class TestDisambiguateHits:
    """Verify disambiguate_telescope_hits() resolves candidates correctly."""

    def test_all_golden_unchanged(self):
        """Three golden hits → returned unchanged (no unresolved plane)."""
        hits = [_h(100.0, 50.0), _h(300.0, 250.0), _h(500.0, 450.0)]
        result = disambiguate_telescope_hits(hits, _Z3)
        assert result == hits

    def test_wrong_length_unchanged(self):
        """Input with != 3 planes → returned unchanged."""
        hits = [_h(100.0, 50.0), _h(300.0, 250.0)]
        assert disambiguate_telescope_hits(hits, _Z3) == hits

    def test_middle_plane_disambiguated(self):
        """
        Middle plane unresolved; 2 candidates — one near the track prediction,
        one far away.  The near candidate is selected.

        Track: x0=100, x2=500 → prediction at z=400 is x=300 mm.
        Candidates: ch=29.0 → 295 mm (Δ=5 mm < tol=15 mm) ✓
                    ch=0.0  → 5 mm   (Δ=295 mm)             ✗
        """
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, 'unresolved', cx=[0.0, 29.0], cy=[0.0, 29.0])
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[0] == h0
        assert result[2] == h2
        assert result[1].quality == 'cluster'
        assert abs(result[1].x_mm - 295.0) < 1e-6
        assert abs(result[1].y_mm - 295.0) < 1e-6

    def test_outer_plane_disambiguated(self):
        """
        Plane 0 unresolved; planes 1 and 2 are good.
        Prediction at z=0: extrapolated from z=400 and z=800.
        Track: x1=300, x2=500 → slope=(500-300)/(800-400)=0.5 mm/mm
               x_pred at z=0: t=(0-400)/(800-400)=-1.0 → x=300-1*(500-300)=100 mm.
        """
        h0 = _h(0.0, 0.0, 'unresolved', cx=[0.0, 9.0], cy=[0.0, 9.0])
        h1 = _h(300.0, 300.0)
        h2 = _h(500.0, 500.0)
        # ch=9: x_mm=(9.0+0.5)*10=95 mm; |95-100|=5<15 ✓
        # ch=0: x_mm=5 mm; |5-100|=95>15 ✗
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1] == h1
        assert result[2] == h2
        assert result[0].quality == 'cluster'
        assert abs(result[0].x_mm - 95.0) < 1e-6

    def test_no_match_candidate_out_of_range(self):
        """Nearest candidate is more than 1.5 strips away → hit unchanged."""
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, 'unresolved', cx=[0.0], cy=[0.0])
        h2 = _h(500.0, 500.0)
        # x_pred at z=400: 300 mm; ch=0 → 5 mm; Δ=295 mm > 15 mm
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == 'unresolved'

    def test_x_matches_y_does_not_unchanged(self):
        """If only one axis resolves, the hit quality stays 'unresolved'."""
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, 'unresolved',
                cx=[29.0],   # x_mm=295 ≈ pred 300 → within tol
                cy=[0.0])    # y_mm=5   far from pred 300 → out of tol
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == 'unresolved'

    def test_two_unresolved_planes_unchanged(self):
        """Two unresolved planes → can't form a 2-good-plane predictor."""
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, 'unresolved', cx=[29.0], cy=[29.0])
        h2 = _h(0.0, 0.0, 'unresolved', cx=[49.0], cy=[49.0])
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == 'unresolved'
        assert result[2].quality == 'unresolved'

    def test_no_candidates_unchanged(self):
        """Unresolved hit with empty candidate list → left unchanged."""
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, 'unresolved', cx=[], cy=[])
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == 'unresolved'


class TestCandidatesPopulated:
    """Unresolved hits from decode_position() carry non-empty candidate lists."""

    def test_fold_off_candidates_present(self, synth_fold):
        """fold=False → some hits are unresolved; their candidate lists are set."""
        result, out = synth_fold
        tel_dir = out / 'telescope'
        utc0, f0 = load_header_params(next(tel_dir.glob('*_header.txt')))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        found_unresolved = 0
        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits = decode_position(ref, pos_paths, n_cols=_N_COLS, fold=False)
            for h in hits:
                if h.quality == 'unresolved':
                    assert h.candidates_x is not None, \
                        'unresolved hit missing candidates_x'
                    assert h.candidates_y is not None, \
                        'unresolved hit missing candidates_y'
                    assert len(h.candidates_x) > 0 or len(h.candidates_y) > 0
                    found_unresolved += 1
        assert found_unresolved > 0, 'no unresolved hits found in fold=False run'
