"""
Tests for stage 4 — telescope internal alignment.

Inject small known per-plane translational misalignments into a fresh
synthetic dataset, run AlignmentAccumulator over all 1000 telescope
events, and assert that the recovered corrections lie within 3σ of the
injected values.

Statistical basis (all hits golden → sigma = STRIP_MM / sqrt(12)):
  Plane 1 (middle, interpolated):
    var(res_1) = sigma^2 * 1.5  →  sigma_mean = sigma*sqrt(1.5)/sqrt(N)
  Planes 0, 2 (outer, extrapolated):
    var(res_k) = sigma^2 * 6    →  sigma_mean = sigma*sqrt(6)/sqrt(N)
"""

import math
from datetime import datetime

import pytest

from monrad.stage1 import (
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.stage3 import decode_position
from monrad.stage4 import (
    AlignmentAccumulator,
    AlignmentCorrection,
    PlaneCorrection,
    fit_telescope_alignment,
)
from monrad.synth import generate, F0, STRIP_MM

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS  = 1000

# Injected offsets: plane 1 only (middle plane).
_DX = 5.0   # mm
_DY = 3.0   # mm
_PLANE_OFFSETS = {1: (_DX, _DY)}

_SIGMA_STRIP = STRIP_MM / math.sqrt(12)   # ≈ 2.887 mm

# Expected sigma of mean residual at each plane (N = _N_TRACKS):
#   k=1 (interpolated): var_factor = 1.5
#   k=0,2 (extrapolated): var_factor = 6
_SIGMA_MEAN_MID   = _SIGMA_STRIP * math.sqrt(1.5) / math.sqrt(_N_TRACKS)
_SIGMA_MEAN_OUTER = _SIGMA_STRIP * math.sqrt(6.0) / math.sqrt(_N_TRACKS)


# ── fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp('synth_stage4')
    result = generate(
        out_dir=out,
        t_x=50.0, t_y=-30.0,
        theta=0.29671, z_p=300.0,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
        plane_offsets=_PLANE_OFFSETS,
    )
    return result, out


@pytest.fixture(scope='module')
def correction(synth):
    """
    Run stage-1 + stage-3 + stage-4 on the telescope stream and return
    the AlignmentCorrection from flushing the accumulator.
    """
    result, out = synth
    tel_dir = out / 'telescope'

    utc0, f0  = load_header_params(next(tel_dir.glob('*_header.txt')))
    gps_paths, pos_paths = find_file_pairs(tel_dir)

    accum = AlignmentAccumulator(flush_every=_N_TRACKS + 1)

    for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
        hits = decode_position(ref, pos_paths, n_cols=3)
        accum.add(hits)

    return accum.flush()


# ── unit tests for AlignmentCorrection and AlignmentAccumulator ─────────

class TestAlignmentCorrectionIdentity:

    def test_identity_shape(self):
        ac = AlignmentCorrection.identity()
        assert len(ac.planes) == 3

    def test_identity_zeros(self):
        ac = AlignmentCorrection.identity()
        for pc in ac.planes:
            assert pc.delta_x    == 0.0
            assert pc.delta_y    == 0.0
            assert pc.rotation_z == 0.0

    def test_identity_no_correction(self):
        ac = AlignmentCorrection.identity()
        assert not ac.needs_correction


class TestAccumulatorFiltering:

    def test_add_wrong_length_ignored(self):
        from monrad.stage3 import Hit
        accum = AlignmentAccumulator(flush_every=1)
        h = Hit(5.0, 5.0, 2.0, 2.0, 'golden')
        result = accum.add([h, h])   # only 2 planes — should be ignored
        assert result is None
        assert len(accum._hits) == 0

    def test_add_invalid_quality_ignored(self):
        from monrad.stage3 import Hit
        accum = AlignmentAccumulator(flush_every=1)
        g = Hit(5.0, 5.0, 2.0, 2.0, 'golden')
        b = Hit(0.0, 0.0, 0.0, 0.0, 'invalid')
        result = accum.add([g, b, g])
        assert result is None
        assert len(accum._hits) == 0

    def test_flush_empty_returns_identity(self):
        accum = AlignmentAccumulator()
        ac = accum.flush()
        assert not ac.needs_correction

    def test_auto_flush_on_full(self):
        from monrad.stage3 import Hit
        accum = AlignmentAccumulator(flush_every=3)
        h = [Hit(float(5 + k*10), 5.0, 2.0, 2.0, 'golden') for k in range(3)]
        # Two events — no flush yet
        assert accum.add(h) is None
        assert accum.add(h) is None
        # Third event — triggers flush
        result = accum.add(h)
        assert isinstance(result, AlignmentCorrection)


class TestFitTelescopeAlignment:

    def test_identity_on_empty(self):
        ac = fit_telescope_alignment([])
        assert not ac.needs_correction

    def test_identity_on_aligned(self):
        """Zero-offset hits → corrections near zero, no flag."""
        from monrad.stage3 import Hit
        from monrad.synth import Z_TEL
        import numpy as np
        rng = np.random.default_rng(0)
        hits = []
        for _ in range(200):
            ax, bx = rng.uniform(50, 500), rng.uniform(-0.3, 0.3)
            ay, by = rng.uniform(50, 500), rng.uniform(-0.3, 0.3)
            row = [
                Hit((ax + bx * z + 0.5) * 1.0,
                    (ay + by * z + 0.5) * 1.0,
                    STRIP_MM / math.sqrt(12),
                    STRIP_MM / math.sqrt(12),
                    'golden')
                for z in Z_TEL
            ]
            hits.append(row)
        ac = fit_telescope_alignment(hits)
        for pc in ac.planes:
            assert abs(pc.delta_x) < 1.0
            assert abs(pc.delta_y) < 1.0
        assert not ac.needs_correction


# ── integration tests: offset recovery ──────────────────────────────────

class TestOffsetRecovery:

    def test_correction_has_three_planes(self, correction):
        assert len(correction.planes) == 3

    def test_needs_correction_flag(self, correction):
        """Injected 5 mm offset on plane 1 exceeds the 1 mm threshold."""
        assert correction.needs_correction

    def test_plane1_delta_x_within_3sigma(self, correction):
        """Two-plane predictor for middle plane → delta_x ≈ injected DX."""
        assert abs(correction.planes[1].delta_x - _DX) < 3 * _SIGMA_MEAN_MID, (
            f"delta_x[1]={correction.planes[1].delta_x:.4f} mm, "
            f"expected {_DX} ± {3*_SIGMA_MEAN_MID:.4f} mm"
        )

    def test_plane1_delta_y_within_3sigma(self, correction):
        """Two-plane predictor for middle plane → delta_y ≈ injected DY."""
        assert abs(correction.planes[1].delta_y - _DY) < 3 * _SIGMA_MEAN_MID, (
            f"delta_y[1]={correction.planes[1].delta_y:.4f} mm, "
            f"expected {_DY} ± {3*_SIGMA_MEAN_MID:.4f} mm"
        )

    def test_plane0_delta_x_within_3sigma(self, correction):
        """
        Outer plane (k=0) extrapolates from planes 1 and 2.  With only
        plane 1 shifted by DX, the predictor for plane 0 reads
        x_pred_0 = 2*x[1] - x[2] ≈ x_true[0] + 2*DX, so the mean
        residual converges to -2*DX.
        """
        expected = -2.0 * _DX
        assert abs(correction.planes[0].delta_x - expected) < 3 * _SIGMA_MEAN_OUTER, (
            f"delta_x[0]={correction.planes[0].delta_x:.4f} mm, "
            f"expected {expected} ± {3*_SIGMA_MEAN_OUTER:.4f} mm"
        )

    def test_plane2_delta_x_within_3sigma(self, correction):
        """By symmetry with plane 0, mean residual for plane 2 → -2*DX."""
        expected = -2.0 * _DX
        assert abs(correction.planes[2].delta_x - expected) < 3 * _SIGMA_MEAN_OUTER, (
            f"delta_x[2]={correction.planes[2].delta_x:.4f} mm, "
            f"expected {expected} ± {3*_SIGMA_MEAN_OUTER:.4f} mm"
        )

    def test_rotation_near_zero(self, correction):
        """No rotation was injected; all three plane rotations should be tiny."""
        for k, pc in enumerate(correction.planes):
            assert abs(pc.rotation_z) < 5e-3, (
                f"rotation_z[{k}]={pc.rotation_z:.6f} rad, expected ≈ 0"
            )

    def test_delta_z_near_zero_no_injection(self, correction):
        """No Z offset was injected; plane 1 delta_z should be near zero."""
        assert abs(correction.planes[1].delta_z) < 10.0, (
            f"delta_z[1]={correction.planes[1].delta_z:.4f} mm "
            f"unexpectedly large (no Z offset injected)"
        )
        assert correction.planes[0].delta_z == 0.0
        assert correction.planes[2].delta_z == 0.0


# ── Z-offset injection + recovery ───────────────────────────────────────

_DZ_INJECT = 30.0  # mm — middle-plane Z offset to inject

# σ of the slope-residual regression estimator at N events.
# The slope (track angle) variance ≈ (mean b² over cosmic rays).
# With a 10 Hz 1000-event run at strip_mm = 10 mm the regression noise
# is small; we allow ±15 mm (half the injection) as a loose bound.
_DZ_TOL = 15.0


@pytest.fixture(scope='module')
def synth_z(tmp_path_factory):
    out = tmp_path_factory.mktemp('synth_stage4_z')
    result = generate(
        out_dir=out,
        t_x=50.0, t_y=-30.0,
        theta=0.29671, z_p=300.0,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
        z_tel_offsets={1: _DZ_INJECT},
    )
    return result, out


@pytest.fixture(scope='module')
def correction_z(synth_z):
    result, out = synth_z
    tel_dir = out / 'telescope'
    utc0, f0  = load_header_params(next(tel_dir.glob('*_header.txt')))
    gps_paths, pos_paths = find_file_pairs(tel_dir)

    accum = AlignmentAccumulator(flush_every=_N_TRACKS + 1)
    for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
        hits = decode_position(ref, pos_paths, n_cols=3)
        accum.add(hits)
    return accum.flush()


class TestZOffsetRecovery:

    def test_plane1_delta_z_detected(self, correction_z):
        """Injected 30 mm Z offset on middle plane must be recovered."""
        dz = correction_z.planes[1].delta_z
        assert abs(dz - _DZ_INJECT) < _DZ_TOL, (
            f"delta_z[1]={dz:.2f} mm, expected {_DZ_INJECT} ± {_DZ_TOL} mm"
        )

    def test_outer_planes_delta_z_zero(self, correction_z):
        """Outer planes always have delta_z = 0 (not fitted)."""
        assert correction_z.planes[0].delta_z == 0.0
        assert correction_z.planes[2].delta_z == 0.0

    def test_needs_correction_flag(self, correction_z):
        """A 30 mm Z offset exceeds _Z_THRESH = 5 mm."""
        assert correction_z.needs_correction

    def test_corrected_z_tel(self, correction_z):
        """corrected_z_tel() shifts only the middle plane."""
        import numpy as np
        from monrad.synth import Z_TEL
        z_corr = correction_z.corrected_z_tel(Z_TEL)
        assert abs(z_corr[0] - Z_TEL[0]) < 1e-6
        assert abs(z_corr[2] - Z_TEL[2]) < 1e-6
        # Middle plane shifted by ≈ DZ_INJECT
        assert abs(z_corr[1] - (Z_TEL[1] + _DZ_INJECT)) < _DZ_TOL
