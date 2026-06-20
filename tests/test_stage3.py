"""
Tests for stage 3 — position decoding.

Decodes every telescope event from the synthetic dataset and verifies
that hits are 'golden', coordinates match ground truth, and sigmas are
correct.
"""

import math
import pytest
import numpy as np
from datetime import datetime

from monrad.stage1 import (
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.stage3 import (
    decode_position,
    disambiguate_telescope_hits,
    reconstruct_plane_candidates,
    Hit,
)
from monrad.synth import generate, F0, Z_TEL, STRIP_MM

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS = 1000
_N_COLS = 3  # telescope planes
_SIGMA_GOLD = STRIP_MM / math.sqrt(12)


# ── fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_stage3")
    result = generate(
        out_dir=out,
        t_x=50.0,
        t_y=-30.0,
        theta=0.29671,
        z_p=300.0,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
    )
    return result, out


@pytest.fixture(scope="module")
def tel_decoded(synth):
    """
    Run stage-1 on the telescope data, then decode position for every
    event.  Returns (decoded, tracks):
      decoded : list of list[Hit] — one inner list (3 Hits) per event
      tracks  : list of (a_x, b_x, a_y, b_y) ground-truth slopes
    """
    result, out = synth
    tel_dir = out / "telescope"

    utc0, f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    gps_paths, pos_paths = find_file_pairs(tel_dir)

    decoded = []
    for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
        hits = decode_position(ref, pos_paths, n_cols=_N_COLS)
        decoded.append(hits)

    return decoded, result["tracks"]


# ── tests ─────────────────────────────────────────────────────────────


class TestDecodePosition:
    def test_event_count(self, tel_decoded):
        decoded, _ = tel_decoded
        assert len(decoded) == _N_TRACKS

    def test_hits_per_event(self, tel_decoded):
        decoded, _ = tel_decoded
        for i, hits in enumerate(decoded):
            assert len(hits) == _N_COLS, (
                f"event {i}: expected {_N_COLS} hits, got {len(hits)}"
            )

    def test_all_golden(self, tel_decoded):
        decoded, _ = tel_decoded
        for i, hits in enumerate(decoded):
            for k, hit in enumerate(hits):
                assert hit.quality == "golden", (
                    f'event {i} plane {k}: quality={hit.quality!r}, expected "golden"'
                )

    def test_sigma(self, tel_decoded):
        decoded, _ = tel_decoded
        for i, hits in enumerate(decoded):
            for k, hit in enumerate(hits):
                assert hit.sigma_x == pytest.approx(_SIGMA_GOLD), (
                    f"event {i} plane {k}: sigma_x={hit.sigma_x}"
                )
                assert hit.sigma_y == pytest.approx(_SIGMA_GOLD), (
                    f"event {i} plane {k}: sigma_y={hit.sigma_y}"
                )

    def test_coordinates(self, tel_decoded):
        """
        For event i, plane k the ground-truth channel is:
          cx = int((a_x + b_x * Z_TEL[k]) / STRIP_MM)
        and the decoded coordinate must equal (cx + 0.5) * STRIP_MM.
        """
        decoded, tracks = tel_decoded
        for i, (hits, (ax, bx, ay, by)) in enumerate(zip(decoded, tracks)):
            for k, (hit, z) in enumerate(zip(hits, Z_TEL)):
                cx = int((ax + bx * z) / STRIP_MM)
                cy = int((ay + by * z) / STRIP_MM)
                exp_x = (cx + 0.5) * STRIP_MM
                exp_y = (cy + 0.5) * STRIP_MM
                assert hit.x_mm == pytest.approx(exp_x), (
                    f"event {i} plane {k}: x_mm={hit.x_mm}, expected {exp_x} (cx={cx})"
                )
                assert hit.y_mm == pytest.approx(exp_y), (
                    f"event {i} plane {k}: y_mm={hit.y_mm}, expected {exp_y} (cy={cy})"
                )


# ── TOT threshold tests ────────────────────────────────────────────────


class TestTOTThreshold:
    def test_thresh1_same_as_default(self, synth):
        """tot_thresh=1 must be identical to the default behaviour."""
        result, out = synth
        tel_dir = out / "telescope"
        utc0, f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits_default = decode_position(ref, pos_paths, n_cols=_N_COLS)
            hits_thresh1 = decode_position(ref, pos_paths, n_cols=_N_COLS, tot_thresh=1)
            assert hits_default == hits_thresh1

    def test_thresh16_passes_all_synthetic(self, synth):
        """All 16 rows of a synthetic golden hit carry the same bit, so
        tot_thresh=16 must not degrade the quality."""
        result, out = synth
        tel_dir = out / "telescope"
        utc0, f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        n_bad = 0
        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            hits = decode_position(ref, pos_paths, n_cols=_N_COLS, tot_thresh=16)
            for h in hits:
                if h.quality not in ("golden", "cluster"):
                    n_bad += 1
        assert n_bad == 0, f"{n_bad} golden hits lost with tot_thresh=16"

    def test_tot_weights_golden_unchanged(self, synth):
        """tot_weights=True must not change golden-hit coordinates (width=1,
        weighting only affects width > 1 clusters)."""
        result, out = synth
        tel_dir = out / "telescope"
        utc0, f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
        gps_paths, pos_paths = find_file_pairs(tel_dir)

        for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
            h_plain = decode_position(ref, pos_paths, n_cols=_N_COLS)
            h_weight = decode_position(ref, pos_paths, n_cols=_N_COLS, tot_weights=True)
            assert h_plain == h_weight

    def test_tot_weighted_centroid_shifts_cluster(self, tmp_path):
        """A two-bit cluster where one bit fires more rows should pull
        the centroid toward the higher-TOT bit."""
        import struct
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
        # y: golden (ribbon=1, fiber=1 → ch=11)
        y_rib = 1 << 1
        y_fib = 1 << 1
        # x: cluster over fiber bits 3 and 4 with ribbon bit 2
        # ribbon bit 2 fires in all 16 rows (TOT=16)
        # fiber bit 3 fires in 12 rows, fiber bit 4 fires in 4 rows
        # → centroid should be pulled toward bit 3 (higher TOT)
        x_rib = 1 << 2  # ribbon bit 2
        x_fib_3 = 1 << 3  # fiber bit 3
        x_fib_4 = 1 << 4  # fiber bit 4
        gen = 0

        # Word with both fiber bits set (used for the OR'd result)
        word_full = (
            y_rib
            | (y_fib << 10)
            | (x_rib << 32)
            | ((x_fib_3 | x_fib_4) << 42)
            | (gen << 52)
        )
        # Word with only fiber bit 3 (TOT=12 rows)
        word_f3_only = (
            y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib_3 << 42) | (gen << 52)
        )

        # Rows 0-3: fiber bit 4 fires (TOT=4); rows 0-15: fiber bit 3 fires
        # Total: fiber 3 → TOT=16, fiber 4 → TOT=4
        raw = struct.pack("<I", 16) + struct.pack("<I", 1)
        for row in range(16):
            w = word_full if row < 4 else word_f3_only
            raw += struct.pack("<Q", w)

        bin_path = tmp_path / "cluster_tot.bin"
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        pos_paths = [bin_path]

        hit_plain = decode_position(ref, pos_paths, n_cols=1)[0]
        hit_weight = decode_position(ref, pos_paths, n_cols=1, tot_weights=True)[0]

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
        clean_word = _ch_to_u64(5, 3, gen=0)
        # Noise bit 7 in fiber_X (bit 42+7=49 of the u64)
        noisy_word = clean_word | (1 << (42 + 7))  # fiber_X bit 7

        rows = [noisy_word] + [clean_word] * 15
        raw = struct.pack("<I", 16) + struct.pack("<I", 1)
        for w in rows:
            raw += struct.pack("<Q", w)
        bin_path = tmp_path / "noise.bin"
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        pos_paths = [bin_path]

        # thresh=2: noise bit is removed, only the clean golden hit remains
        hits2 = decode_position(ref, pos_paths, n_cols=1, tot_thresh=2)

        assert hits2[0].quality in ("golden", "cluster"), (
            f"tot_thresh=2 should recover golden hit, got {hits2[0].quality}"
        )
        # The clean hit decodes to x=55mm, y=35mm
        assert hits2[0].x_mm == pytest.approx(5.5 * STRIP_MM)
        assert hits2[0].y_mm == pytest.approx(3.5 * STRIP_MM)


# ── Tests for disambiguate_telescope_hits ─────────────────────────────────

_Z3 = np.array([0.0, 400.0, 800.0])
_SIGMA = STRIP_MM / math.sqrt(12)


def _h(x, y, quality="golden", cx=None, cy=None):
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
        h1 = _h(
            0.0, 0.0, "unresolved", cx=[(0.0, 1), (29.0, 1)], cy=[(0.0, 1), (29.0, 1)]
        )
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[0] == h0
        assert result[2] == h2
        assert result[1].quality == "cluster"
        assert abs(result[1].x_mm - 295.0) < 1e-6
        assert abs(result[1].y_mm - 295.0) < 1e-6

    def test_outer_plane_disambiguated(self):
        """
        Plane 0 unresolved; planes 1 and 2 are good.
        Prediction at z=0: extrapolated from z=400 and z=800.
        Track: x1=300, x2=500 → slope=(500-300)/(800-400)=0.5 mm/mm
               x_pred at z=0: t=(0-400)/(800-400)=-1.0 → x=300-1*(500-300)=100 mm.
        """
        h0 = _h(
            0.0, 0.0, "unresolved", cx=[(0.0, 1), (9.0, 1)], cy=[(0.0, 1), (9.0, 1)]
        )
        h1 = _h(300.0, 300.0)
        h2 = _h(500.0, 500.0)
        # ch=9: x_mm=(9.0+0.5)*10=95 mm; |95-100|=5<15 ✓
        # ch=0: x_mm=5 mm; |5-100|=95>15 ✗
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1] == h1
        assert result[2] == h2
        assert result[0].quality == "cluster"
        assert abs(result[0].x_mm - 95.0) < 1e-6

    def test_sigma_carries_cluster_width(self):
        """Disambiguated sigma reflects each axis's cluster width, not always 1 strip."""
        h0 = _h(100.0, 100.0)
        # cx: width-2 cluster centroid near prediction; cy: width-1
        h1 = _h(0.0, 0.0, "unresolved", cx=[(29.0, 2)], cy=[(29.0, 1)])
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == "cluster"
        assert abs(result[1].sigma_x - STRIP_MM * 2 / math.sqrt(12)) < 1e-9
        assert abs(result[1].sigma_y - STRIP_MM * 1 / math.sqrt(12)) < 1e-9

    def test_no_match_candidate_out_of_range(self):
        """Nearest candidate is more than 1.5 strips away → hit unchanged."""
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, "unresolved", cx=[(0.0, 1)], cy=[(0.0, 1)])
        h2 = _h(500.0, 500.0)
        # x_pred at z=400: 300 mm; ch=0 → 5 mm; Δ=295 mm > 15 mm
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == "unresolved"

    def test_x_matches_y_does_not_unchanged(self):
        """If only one axis resolves, the hit quality stays 'unresolved'."""
        h0 = _h(100.0, 100.0)
        h1 = _h(
            0.0,
            0.0,
            "unresolved",
            cx=[(29.0, 1)],  # x_mm=295 ≈ pred 300 → within tol
            cy=[(0.0, 1)],
        )  # y_mm=5   far from pred 300 → out of tol
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == "unresolved"

    def test_two_unresolved_planes_unchanged(self):
        """Two unresolved planes → can't form a 2-good-plane predictor."""
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, "unresolved", cx=[(29.0, 1)], cy=[(29.0, 1)])
        h2 = _h(0.0, 0.0, "unresolved", cx=[(49.0, 1)], cy=[(49.0, 1)])
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == "unresolved"
        assert result[2].quality == "unresolved"

    def test_no_candidates_unchanged(self):
        """Unresolved hit with empty candidate list → left unchanged."""
        h0 = _h(100.0, 100.0)
        h1 = _h(0.0, 0.0, "unresolved", cx=[], cy=[])
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == "unresolved"

    def test_single_axis_unresolved_recovered(self):
        """
        Single-axis failure: x failed (two candidates), y was resolved and is
        carried as a one-element candidate.  The failed x is filled from the
        projection and the known y matches trivially → the plane is recovered.

        Track: x0=100, x2=500 → x_pred at z=400 is 300 mm.
               y0=100, y2=500 → y_pred at z=400 is 300 mm.
        cx: ch=29.0 → 295 mm (Δ=5 < 15) ✓ ; ch=0.0 → 5 mm (Δ=295) ✗
        cy: ch=29.5 → 300 mm (the resolved axis, Δ=0)            ✓
        """
        h0 = _h(100.0, 100.0)
        h1 = _h(
            0.0,
            0.0,
            "unresolved",
            cx=[(0.0, 1), (29.0, 1)],  # failed axis: real hypotheses
            cy=[(29.5, 1)],  # resolved axis: single kept candidate
        )
        h2 = _h(500.0, 500.0)
        result = disambiguate_telescope_hits([h0, h1, h2], _Z3)
        assert result[1].quality == "cluster"
        assert abs(result[1].x_mm - 295.0) < 1e-6
        assert abs(result[1].y_mm - 300.0) < 1e-6


class TestCandidatesPopulated:
    """Unresolved hits from decode_position() carry non-empty candidate lists."""

    def test_unresolved_candidates_present(self, tmp_path):
        """An unresolved hit (multiple clusters) carries a non-empty candidate list."""
        import struct
        from monrad.stage1 import PosRef

        # X: two disconnected fiber clusters (bits 0 and 5 both set, ribbon bit 2)
        # → candidates [20, 25], which are non-contiguous → unresolved
        x_rib = 1 << 2
        x_fib = (1 << 0) | (1 << 5)
        y_rib = 1 << 1
        y_fib = 1 << 1
        gen = 0
        word = y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)
        raw = struct.pack("<I", 16) + struct.pack("<I", 1)
        for _ in range(16):
            raw += struct.pack("<Q", word)
        bin_path = tmp_path / "unresolved.bin"
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        hits = decode_position(ref, [bin_path], n_cols=1)
        h = hits[0]
        assert h.quality == "unresolved"
        assert h.candidates_x is not None and len(h.candidates_x) > 0

    def test_single_axis_resolved_kept_as_candidate(self, tmp_path):
        """
        When only one axis fails, the axis that resolved is retained as a
        single candidate (its centroid + width) so the plane can still be
        recovered by the two-plane projection.

        Same word as above: x is unresolved (fiber bits 0,5 on ribbon bit 2),
        y is golden (ribbon bit 1, fiber bit 1 → channel 10*1+1 = 11).
        """
        import struct
        from monrad.stage1 import PosRef

        x_rib = 1 << 2
        x_fib = (1 << 0) | (1 << 5)
        y_rib = 1 << 1
        y_fib = 1 << 1
        gen = 0
        word = y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)
        raw = struct.pack("<I", 16) + struct.pack("<I", 1)
        for _ in range(16):
            raw += struct.pack("<Q", word)
        bin_path = tmp_path / "single_axis.bin"
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        h = decode_position(ref, [bin_path], n_cols=1)[0]
        assert h.quality == "unresolved"
        # Failed axis: real multi-candidate hypotheses.
        assert h.candidates_x is not None and len(h.candidates_x) > 0
        # Resolved axis: kept as one golden candidate at channel 11, width 1.
        assert h.candidates_y == [(11.0, 1)]


# ── Tests for reconstruct_plane_candidates ─────────────────────────────────


class TestPlaneCandidates:
    """
    reconstruct_plane_candidates() enumerates per-plane candidate (x, y)
    positions instead of collapsing each plane to a single resolved Hit —
    the basis of the Stage 5 combinatorial track finder.
    """

    def _write_block(self, tmp_path, name, word, n_cols=1):
        import struct

        raw = struct.pack("<I", 16) + struct.pack("<I", n_cols)
        for _ in range(16):
            raw += struct.pack("<Q", word)
        bin_path = tmp_path / name
        bin_path.write_bytes(raw)
        return bin_path

    def test_mirror_fold_axis_yields_multiple_candidates(self, tmp_path):
        """
        X is mirror-fold ambiguous: ribbon bits {2, 7} and fiber bits {3, 6}
        both fire (the folded-fiber MAROC wiring, DESIGN.md §10), giving 4
        ribbon×fiber combinations.  Y is golden (ribbon=1, fiber=1 → ch=11).
        Expect exactly the 4 X-candidates × 1 Y-candidate = 4 points.
        """
        from monrad.stage1 import PosRef

        y_rib = 1 << 1
        y_fib = 1 << 1
        x_rib = (1 << 2) | (1 << 7)
        x_fib = (1 << 3) | (1 << 6)
        gen = 0
        word = y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)
        bin_path = self._write_block(tmp_path, "mirror_fold.bin", word)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        res = reconstruct_plane_candidates(ref, [bin_path], n_cols=1)
        assert len(res) == 1
        cands = res[0]
        assert len(cands) == 4

        x_mm_set = {round(c.x_mm, 6) for c in cands}
        expected_chs = [10 * r + f for r in (2, 7) for f in (3, 6)]
        assert x_mm_set == {(ch + 0.5) * STRIP_MM for ch in expected_chs}
        for c in cands:
            assert c.y_mm == pytest.approx((11 + 0.5) * STRIP_MM)
            assert c.sigma_x == pytest.approx(_SIGMA_GOLD)
            assert c.sigma_y == pytest.approx(_SIGMA_GOLD)
            # Every bit fires in all 16 rows (the word is repeated unchanged),
            # so every candidate's TOT score is 16*16 regardless of which
            # mirror-pair channel it picks, and width-1 axes are "golden".
            assert c.quality == "golden"
            assert c.tot_x == 16 * 16
            assert c.tot_y == 16 * 16

    def test_cap_limits_candidate_count(self, tmp_path):
        """
        Both axes mirror-fold ambiguous → 4×4 = 16 candidate points, all of
        equal compactness (width 1 on each axis), so the cap keeps exactly
        max_per_plane of them.
        """
        from monrad.stage1 import PosRef

        y_rib = (1 << 1) | (1 << 8)
        y_fib = (1 << 1) | (1 << 8)
        x_rib = (1 << 2) | (1 << 7)
        x_fib = (1 << 3) | (1 << 6)
        gen = 0
        word = y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)
        bin_path = self._write_block(tmp_path, "mirror_fold_both.bin", word)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        res = reconstruct_plane_candidates(ref, [bin_path], n_cols=1, max_per_plane=8)
        assert len(res[0]) == 8

    def test_invalid_plane_returns_empty_list(self, tmp_path):
        """X_ribbon=0 (no ribbon channel fired) → invalid → empty list."""
        from monrad.stage1 import PosRef

        y_rib = 1 << 1
        y_fib = 1 << 1
        x_rib = 0
        x_fib = 1 << 3
        gen = 0
        word = y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)
        bin_path = self._write_block(tmp_path, "invalid.bin", word)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        res = reconstruct_plane_candidates(ref, [bin_path], n_cols=1)
        assert res == [[]]

    def test_golden_plane_yields_one_candidate(self, tmp_path):
        """A clean golden hit on both axes yields exactly one candidate."""
        from monrad.stage1 import PosRef

        y_rib = 1 << 1
        y_fib = 1 << 1
        x_rib = 1 << 2
        x_fib = 1 << 3
        gen = 0
        word = y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)
        bin_path = self._write_block(tmp_path, "golden.bin", word)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        res = reconstruct_plane_candidates(ref, [bin_path], n_cols=1)
        assert len(res[0]) == 1
        c = res[0][0]
        assert c.x_mm == pytest.approx((23 + 0.5) * STRIP_MM)
        assert c.y_mm == pytest.approx((11 + 0.5) * STRIP_MM)
        assert c.quality == "golden"
        assert c.tot_x == 16 * 16
        assert c.tot_y == 16 * 16

    def test_tot_reflects_per_bit_row_counts_and_cluster_quality(self, tmp_path):
        """
        X is a 2-wide cluster (fiber bits 3 and 4, ribbon bit 2 fixed) where
        fiber bit 3 fires in all 16 rows but fiber bit 4 only fires in 4 —
        mirrors test_tot_weighted_centroid_shifts_cluster's row pattern.
        The single resulting X candidate must report tot_x as the *sum* of
        each contributing (ribbon, fiber) pair's TOT product, and its width
        (2 on X) must downgrade quality from "golden" to "cluster" even
        though Y stays golden.
        """
        import struct

        from monrad.stage1 import PosRef

        y_rib = 1 << 1
        y_fib = 1 << 1
        x_rib = 1 << 2
        x_fib_3 = 1 << 3
        x_fib_4 = 1 << 4
        gen = 0

        word_full = (
            y_rib
            | (y_fib << 10)
            | (x_rib << 32)
            | ((x_fib_3 | x_fib_4) << 42)
            | (gen << 52)
        )
        word_f3_only = (
            y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib_3 << 42) | (gen << 52)
        )

        raw = struct.pack("<I", 16) + struct.pack("<I", 1)
        raw += b"".join(
            struct.pack("<Q", word_full if row < 4 else word_f3_only)
            for row in range(16)
        )

        bin_path = tmp_path / "tot_cluster.bin"
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        res = reconstruct_plane_candidates(ref, [bin_path], n_cols=1)
        assert len(res[0]) == 1
        c = res[0][0]

        # ribbon bit 2: TOT=16 throughout.  fiber bit 3: TOT=16.  fiber bit 4: TOT=4.
        # tot_x = ribbon[2]*fiber[3] + ribbon[2]*fiber[4] = 16*16 + 16*4 = 320.
        assert c.tot_x == 16 * 16 + 16 * 4
        assert c.tot_y == 16 * 16
        assert c.quality == "cluster"

    def test_adjacent_ribbons_split_into_separate_runs(self, tmp_path):
        """
        A single contiguous ribbon cluster {2,3} crossed with fiber cluster
        {3,4} gives combined channels {23,24,33,34}.  These are NOT one
        width-4 hit straddling the 25..32 gap — adjacent ribbons are 10
        channels apart, so the cross-product splits into two gap-free runs,
        each a distinct 2-wide cluster candidate: [23,24] and [33,34].
        """
        from monrad.stage1 import PosRef

        y_rib = 1 << 1
        y_fib = 1 << 1
        x_rib = (1 << 2) | (1 << 3)  # adjacent ribbons → one cluster [2,3]
        x_fib = (1 << 3) | (1 << 4)  # adjacent fibers  → one cluster [3,4]
        gen = 0
        word = y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib << 42) | (gen << 52)
        bin_path = self._write_block(tmp_path, "adjacent_ribbons.bin", word)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)
        res = reconstruct_plane_candidates(ref, [bin_path], n_cols=1)
        cands = res[0]
        # Two X-runs × one Y-candidate = 2 points, each a 2-wide X cluster.
        assert len(cands) == 2
        x_mm_set = {round(c.x_mm, 6) for c in cands}
        assert x_mm_set == {
            round((23.5 + 0.5) * STRIP_MM, 6),  # centroid of [23,24] = 23.5
            round((33.5 + 0.5) * STRIP_MM, 6),  # centroid of [33,34] = 33.5
        }
        for c in cands:
            assert c.quality == "cluster"  # width 2 on X
            assert c.sigma_x == pytest.approx(2 * STRIP_MM / math.sqrt(12))

    def test_tot_weights_shifts_cluster_centroid(self, tmp_path):
        """
        With tot_weights=True the X cluster centroid is TOT-weighted toward the
        stronger fiber bit, exactly as decode_position's tot_weights path does —
        otherwise the telescope candidates silently ignore the pipeline's
        --tot-weights flag (the probe decode honours it).

        X cluster = fiber bits 3,4 (ribbon 2) → channels 23,24.  Fiber bit 3
        fires in all 16 rows, fiber bit 4 in only 4.  Weights: ch23 = 16*16,
        ch24 = 16*4.  Weighted centroid = (256*23 + 64*24)/320 = 23.2, vs the
        unweighted 23.5.
        """
        import struct

        from monrad.stage1 import PosRef

        y_rib = 1 << 1
        y_fib = 1 << 1
        x_rib = 1 << 2
        x_fib_3 = 1 << 3
        x_fib_4 = 1 << 4
        gen = 0
        word_full = (
            y_rib
            | (y_fib << 10)
            | (x_rib << 32)
            | ((x_fib_3 | x_fib_4) << 42)
            | (gen << 52)
        )
        word_f3_only = (
            y_rib | (y_fib << 10) | (x_rib << 32) | (x_fib_3 << 42) | (gen << 52)
        )
        raw = struct.pack("<I", 16) + struct.pack("<I", 1)
        raw += b"".join(
            struct.pack("<Q", word_full if row < 4 else word_f3_only)
            for row in range(16)
        )
        bin_path = tmp_path / "tot_weighted.bin"
        bin_path.write_bytes(raw)

        ref = PosRef(file_idx=0, row_offset=0, split_rows=0)

        unweighted = reconstruct_plane_candidates(ref, [bin_path], n_cols=1)[0][0]
        assert unweighted.x_mm == pytest.approx((23.5 + 0.5) * STRIP_MM)

        weighted = reconstruct_plane_candidates(
            ref, [bin_path], n_cols=1, tot_weights=True
        )[0][0]
        assert weighted.x_mm == pytest.approx((23.2 + 0.5) * STRIP_MM)
        # Width/quality unchanged — tot_weights only moves the centroid.
        assert weighted.quality == "cluster"
        assert weighted.sigma_x == pytest.approx(2 * STRIP_MM / math.sqrt(12))
