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

from monrad.timing import (
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.reconstruction import decode_position
from monrad.alignment import (
    AlignmentAccumulator,
    AlignmentCorrection,
    fit_telescope_alignment,
)
from monrad.synthetic import generate, F0, STRIP_MM

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS = 1000

# Injected offsets: plane 1 only (middle plane).
_DX = 5.0  # mm
_DY = 3.0  # mm
_PLANE_OFFSETS = {1: (_DX, _DY)}

_SIGMA_STRIP = STRIP_MM / math.sqrt(12)  # ≈ 2.887 mm

# Expected sigma of mean residual at each plane (N = _N_TRACKS):
#   k=1 (interpolated): var_factor = 1.5
#   k=0,2 (extrapolated): var_factor = 6
_SIGMA_MEAN_MID = _SIGMA_STRIP * math.sqrt(1.5) / math.sqrt(_N_TRACKS)
_SIGMA_MEAN_OUTER = _SIGMA_STRIP * math.sqrt(6.0) / math.sqrt(_N_TRACKS)


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_stage4")
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
        plane_offsets=_PLANE_OFFSETS,
    )
    return result, out


@pytest.fixture(scope="module")
def correction(synth):
    """
    Run stage-1 + stage-3 + stage-4 on the telescope stream and return
    the AlignmentCorrection from flushing the accumulator.
    """
    result, out = synth
    tel_dir = out / "telescope"

    utc0, f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
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
            assert pc.delta_x == 0.0
            assert pc.delta_y == 0.0
            assert pc.rotation_z == 0.0

    def test_identity_no_correction(self):
        ac = AlignmentCorrection.identity()
        assert not ac.needs_correction


class TestAccumulatorFiltering:
    def test_add_wrong_length_ignored(self):
        from monrad.reconstruction import Hit

        accum = AlignmentAccumulator(flush_every=1)
        h = Hit(5.0, 5.0, 2.0, 2.0, "golden")
        result = accum.add([h, h])  # only 2 planes — should be ignored
        assert result is None
        assert len(accum._hits) == 0

    def test_add_invalid_quality_ignored(self):
        from monrad.reconstruction import Hit

        accum = AlignmentAccumulator(flush_every=1)
        g = Hit(5.0, 5.0, 2.0, 2.0, "golden")
        b = Hit(0.0, 0.0, 0.0, 0.0, "invalid")
        result = accum.add([g, b, g])
        assert result is None
        assert len(accum._hits) == 0

    def test_flush_empty_returns_identity(self):
        accum = AlignmentAccumulator()
        ac = accum.flush()
        assert not ac.needs_correction

    def test_auto_flush_on_full(self):
        from monrad.reconstruction import Hit

        accum = AlignmentAccumulator(flush_every=3)
        h = [Hit(float(5 + k * 10), 5.0, 2.0, 2.0, "golden") for k in range(3)]
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
        from monrad.reconstruction import Hit
        from monrad.synthetic import Z_TEL
        import numpy as np

        rng = np.random.default_rng(0)
        hits = []
        for _ in range(200):
            ax, bx = rng.uniform(50, 500), rng.uniform(-0.3, 0.3)
            ay, by = rng.uniform(50, 500), rng.uniform(-0.3, 0.3)
            row = [
                Hit(
                    (ax + bx * z + 0.5) * 1.0,
                    (ay + by * z + 0.5) * 1.0,
                    STRIP_MM / math.sqrt(12),
                    STRIP_MM / math.sqrt(12),
                    "golden",
                )
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
            f"expected {_DX} ± {3 * _SIGMA_MEAN_MID:.4f} mm"
        )

    def test_plane1_delta_y_within_3sigma(self, correction):
        """Two-plane predictor for middle plane → delta_y ≈ injected DY."""
        assert abs(correction.planes[1].delta_y - _DY) < 3 * _SIGMA_MEAN_MID, (
            f"delta_y[1]={correction.planes[1].delta_y:.4f} mm, "
            f"expected {_DY} ± {3 * _SIGMA_MEAN_MID:.4f} mm"
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
            f"expected {expected} ± {3 * _SIGMA_MEAN_OUTER:.4f} mm"
        )

    def test_plane2_delta_x_within_3sigma(self, correction):
        """By symmetry with plane 0, mean residual for plane 2 → -2*DX."""
        expected = -2.0 * _DX
        assert abs(correction.planes[2].delta_x - expected) < 3 * _SIGMA_MEAN_OUTER, (
            f"delta_x[2]={correction.planes[2].delta_x:.4f} mm, "
            f"expected {expected} ± {3 * _SIGMA_MEAN_OUTER:.4f} mm"
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


@pytest.fixture(scope="module")
def synth_z(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_stage4_z")
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
        z_tel_offsets={1: _DZ_INJECT},
    )
    return result, out


@pytest.fixture(scope="module")
def correction_z(synth_z):
    result, out = synth_z
    tel_dir = out / "telescope"
    utc0, f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
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
        from monrad.synthetic import Z_TEL

        z_corr = correction_z.corrected_z_tel(Z_TEL)
        assert abs(z_corr[0] - Z_TEL[0]) < 1e-6
        assert abs(z_corr[2] - Z_TEL[2]) < 1e-6
        # Middle plane shifted by ≈ DZ_INJECT
        assert abs(z_corr[1] - (Z_TEL[1] + _DZ_INJECT)) < _DZ_TOL


# ── Tilt injection + recovery ───────────────────────────────────────────

# A tilt about the y-axis on the middle plane displaces its x hit by
# tilt_y·b_x·x — the slope×lever-arm residual of DESIGN.md §7.3.  At ~1000
# tracks the b·x regression resolves the tilt to ≈3 mrad, so a 30 mrad
# injection sits well clear of noise; allow a loose half-injection bound.
_TILT_INJECT = 0.03  # rad — middle-plane tilt about the y-axis
_TILT_TOL = 0.015  # rad


@pytest.fixture(scope="module")
def synth_tilt(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_stage4_tilt")
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
        z_tel_tilts={1: (0.0, _TILT_INJECT)},  # (tilt_x, tilt_y)
    )
    return result, out


@pytest.fixture(scope="module")
def correction_tilt(synth_tilt):
    result, out = synth_tilt
    tel_dir = out / "telescope"
    utc0, f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    gps_paths, pos_paths = find_file_pairs(tel_dir)

    accum = AlignmentAccumulator(flush_every=_N_TRACKS + 1)
    for _ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
        hits = decode_position(ref, pos_paths, n_cols=3)
        accum.add(hits)
    return accum.flush()


class TestTiltRecovery:
    def test_plane1_tilt_y_detected(self, correction_tilt):
        """Injected 30 mrad tilt about y on the middle plane is recovered."""
        ty = correction_tilt.planes[1].tilt_y
        assert abs(ty - _TILT_INJECT) < _TILT_TOL, (
            f"tilt_y[1]={ty:.5f} rad, expected {_TILT_INJECT} ± {_TILT_TOL} rad"
        )

    def test_plane1_tilt_x_near_zero(self, correction_tilt):
        """No tilt about x was injected; the orthogonal axis stays ≈ 0."""
        tx = correction_tilt.planes[1].tilt_x
        assert abs(tx) < _TILT_TOL, f"tilt_x[1]={tx:.5f} rad, expected ≈ 0"

    def test_tilt_not_absorbed_into_delta_z(self, correction_tilt):
        """The joint b/b·x fit keeps a pure tilt out of the Z offset."""
        assert abs(correction_tilt.planes[1].delta_z) < _DZ_TOL, (
            f"delta_z[1]={correction_tilt.planes[1].delta_z:.2f} mm "
            f"should stay near zero for a pure tilt"
        )

    def test_outer_planes_tilt_zero(self, correction_tilt):
        """Outer planes never get a tilt fitted."""
        for k in (0, 2):
            assert correction_tilt.planes[k].tilt_x == 0.0
            assert correction_tilt.planes[k].tilt_y == 0.0

    def test_needs_correction_flag(self, correction_tilt):
        """A 30 mrad tilt exceeds _TILT_THRESH = 5 mrad."""
        assert correction_tilt.needs_correction


# ── Middle plane selected by z, not by file-column index ────────────────
#
# Regression for the 0_testLab_20210723 geometry: the telescope columns are
# not stored in z order (column 1 is the *far* plane, column 2 the middle).
# fit_telescope_alignment must fit delta_z/tilt on the geometric middle
# (argsort(z)[1]), not on hardcoded column 1.  Build straight-track hits
# directly — a displaced middle plane reports x = a + b·(z_mid + dz), so the
# two-plane predictor leaves a residual b·dz that the fit must attribute to
# the correct column.


def _straight_track_hits(z_cols, dz_col, dz, n=600, seed=7):
    """N events of 3 collinear hits at z_cols, with column dz_col shifted by dz."""
    import numpy as np

    from monrad.reconstruction import Hit

    rng = np.random.default_rng(seed)
    z = np.asarray(z_cols, dtype=float)
    z_eff = z.copy()
    z_eff[dz_col] += dz  # the displaced plane samples the track at z + dz
    hits = []
    for _ in range(n):
        # Random straight tracks; spread of slopes gives the b-regression
        # something to bite on.  Intercepts keep coords on the 0–1000 mm area.
        a_x, b_x = rng.uniform(300.0, 700.0), rng.uniform(-0.3, 0.3)
        a_y, b_y = rng.uniform(300.0, 700.0), rng.uniform(-0.3, 0.3)
        event = [
            Hit(
                float(a_x + b_x * z_eff[k]),
                float(a_y + b_y * z_eff[k]),
                _SIGMA_STRIP,
                _SIGMA_STRIP,
                "golden",
            )
            for k in range(3)
        ]
        hits.append(event)
    return hits


class TestMiddlePlaneByZOrder:
    # Columns in file order map to z = [0, 800, 400]: column 2 is the middle.
    _Z_SHUFFLED = [0.0, 800.0, 400.0]
    _MID_COL = 2  # argsort([0,800,400])[1]
    _DZ = 30.0

    @pytest.fixture(scope="class")
    def correction_shuffled(self):
        hits = _straight_track_hits(self._Z_SHUFFLED, self._MID_COL, self._DZ)
        import numpy as np

        return fit_telescope_alignment(hits, np.asarray(self._Z_SHUFFLED))

    def test_delta_z_on_geometric_middle(self, correction_shuffled):
        """delta_z is fitted on column 2 (the middle by z), not column 1."""
        dz = correction_shuffled.planes[self._MID_COL].delta_z
        assert abs(dz - self._DZ) < _DZ_TOL, (
            f"delta_z[{self._MID_COL}]={dz:.2f} mm, expected {self._DZ} ± {_DZ_TOL}"
        )

    def test_outer_columns_delta_z_zero(self, correction_shuffled):
        """The two non-middle columns (0 and 1) leave delta_z untouched."""
        assert correction_shuffled.planes[0].delta_z == 0.0
        assert correction_shuffled.planes[1].delta_z == 0.0

    def test_far_column_one_not_treated_as_middle(self, correction_shuffled):
        """The old code fit column 1; it is an outer plane here and stays 0."""
        assert correction_shuffled.planes[1].tilt_x == 0.0
        assert correction_shuffled.planes[1].tilt_y == 0.0
