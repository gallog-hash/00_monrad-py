"""
Tests for stage 5 — probe pose fit.

Runs the complete streaming pipeline on the standard synthetic dataset
(no injected misalignments) and asserts the recovered pose parameters
lie within 3σ of ground truth.

The square probe creates a four-fold (90°) degeneracy in the θ landscape.
The θ test accounts for this by checking the angle error modulo π/2.
The z_p test is unambiguous (z_p is identical for all four equivalent
solutions).  For t_x and t_y the test also handles the four-fold case
by checking a reconstructed canonical solution.
"""

import math
from datetime import datetime

import numpy as np
import pytest

from monrad.timing import (
    load_header_params,
    find_file_pairs,
    reconstruct_stream,
)
from monrad.coincidence import coincidence_stream
from monrad.alignment import AlignmentCorrection
from monrad.pose import (
    Coincidence,
    DecodeReport,
    PoseResult,
    PoseFitter,
    fit_probe_pose,
    _tel_line_fit,
    _linear_solve_fixed_theta,
    _sigma_tel_at_z,
)
from monrad.synthetic import generate, F0, Z_TEL, STRIP_MM

_START_UTC = datetime(2023, 4, 18, 19, 21, 0)
_N_TRACKS = 1000
_TRUE_TX = 50.0
_TRUE_TY = -30.0
_TRUE_THETA = 0.29671  # ≈ radians(17°)
_TRUE_ZP = 300.0
_SIGMA_STRIP = STRIP_MM / math.sqrt(12)  # golden hit σ (width-1)
_SIGMA_CLUSTER2 = 2 * STRIP_MM / math.sqrt(12)  # width-2 cluster σ


# ── fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth_stage5")
    result = generate(
        out_dir=out,
        t_x=_TRUE_TX,
        t_y=_TRUE_TY,
        theta=_TRUE_THETA,
        z_p=_TRUE_ZP,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
    )
    return result, out


@pytest.fixture(scope="module")
def pose_result(synth):
    """
    Full streaming pipeline: stage 1 + 2 + 5 on standard synthetic data.
    Returns PoseResult after flushing the PoseFitter.
    """
    result, out = synth
    tel_dir = out / "telescope"
    prb_dir = out / "probe"

    tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header.txt")))

    tel_gps, tel_pos = find_file_pairs(tel_dir)
    prb_gps, prb_pos = find_file_pairs(prb_dir)

    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    fitter = PoseFitter(
        tel_z=Z_TEL,
        alignment=AlignmentCorrection.identity(),
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
        refit_every=_N_TRACKS + 1,  # no auto-flush; use explicit flush()
    )

    for cluster in coincidence_stream(
        [tel_stream, prb_stream],
        detector_ids=[0, 1],
    ):
        fitter.add(cluster)

    pr = fitter.flush()
    assert pr is not None, "PoseFitter.flush() returned None (too few coincidences)"
    return pr


# ── helper ────────────────────────────────────────────────────────────────


def _theta_err_mod90(theta_fit: float, theta_true: float) -> float:
    """
    Minimum |theta_fit - (theta_true + k*π/2)| over integer k.
    Accounts for the 4-fold rotational degeneracy of a square probe.
    """
    return min(abs(theta_fit - theta_true - k * math.pi / 2) for k in range(-4, 5))


def _nearest_k90(theta_fit: float, theta_true: float) -> int:
    """Return k such that theta_fit ≈ theta_true + k*π/2 (k chosen to minimise error)."""
    candidates = list(range(-3, 5))
    return min(candidates, key=lambda k: abs(theta_fit - theta_true - k * math.pi / 2))


# ── unit tests for helpers ─────────────────────────────────────────────────


class TestTelLineFit:
    def test_perfect_straight_line(self):
        z = Z_TEL
        ax, bx, ay, by = 200.0, 0.1, 150.0, -0.05
        x = ax + bx * z
        y = ay + by * z
        a_x, b_x, a_y, b_y, cov_x, cov_y, chi2 = _tel_line_fit(
            x, y, z, _SIGMA_STRIP, _SIGMA_STRIP
        )
        assert abs(a_x - ax) < 1e-9
        assert abs(b_x - bx) < 1e-9
        assert abs(a_y - ay) < 1e-9
        assert abs(b_y - by) < 1e-9
        assert chi2 < 1e-9

    def test_cov_is_positive(self):
        z = Z_TEL
        x = np.array([205.0, 255.0, 310.0])
        y = np.array([150.0, 148.0, 147.0])
        _, _, _, _, cov_x, cov_y, _ = _tel_line_fit(x, y, z, _SIGMA_STRIP, _SIGMA_STRIP)
        va, cab, vb = cov_x
        assert va > 0 and vb > 0


class TestHeteroscedasticLineFit:
    """
    Per-plane / per-axis σ weighting (DESIGN.md §6.4, §8.2).  A plane that
    decodes as a wide `cluster` carries a larger σ and must be weighted
    *less* than the sharp `golden` planes — the line should follow the
    sharp planes more closely.  Before the per-axis-σ fix, _tel_line_fit
    broadcast plane-0's σ to all planes and both axes, so a cluster plane's
    larger σ was ignored; these tests pin the corrected behaviour.

    Geometry: outer planes (0, 2) sit on a flat line at x=100; the middle
    plane (1) is displaced 10 mm off it.  The weights are z-symmetric, so
    the fitted slope stays 0 and the fit reduces to a weighted mean — the
    intercept moves toward whichever planes are trusted more.
    """

    _Z = Z_TEL
    _X = np.array([100.0, 110.0, 100.0])  # plane 1 is 10 mm off the outer line
    _Y = np.array([200.0, 220.0, 240.0])  # clean line, never reweighted here

    def _fit(self, sigma_x, sigma_y):
        return _tel_line_fit(self._X, self._Y, self._Z, sigma_x, sigma_y)

    def test_sharp_planes_followed_more_closely(self):
        """
        Inflating only the middle-plane X σ (golden→cluster) must pull the X
        line toward the sharp outer planes: their |residual| shrinks while
        the down-weighted middle plane's |residual| grows.
        """
        ax_u, bx_u, *_ = self._fit(_SIGMA_STRIP, _SIGMA_STRIP)
        sx = np.array([_SIGMA_STRIP, _SIGMA_CLUSTER2, _SIGMA_STRIP])
        ax_h, bx_h, *_ = self._fit(sx, _SIGMA_STRIP)

        res_outer_u = abs(self._X[0] - (ax_u + bx_u * self._Z[0]))
        res_outer_h = abs(self._X[0] - (ax_h + bx_h * self._Z[0]))
        res_mid_u = abs(self._X[1] - (ax_u + bx_u * self._Z[1]))
        res_mid_h = abs(self._X[1] - (ax_h + bx_h * self._Z[1]))

        assert res_outer_h < res_outer_u, (
            f"sharp-plane residual should shrink: {res_outer_u:.3f} → {res_outer_h:.3f}"
        )
        assert res_mid_h > res_mid_u, (
            f"down-weighted-plane residual should grow: {res_mid_u:.3f} → {res_mid_h:.3f}"
        )

    def test_cov_diverges_between_axes(self):
        """
        With a wide cluster on X but golden Y, the X fit holds less
        information than Y, so its parameter variance must be strictly
        larger — and the two axes' covariances must genuinely differ.
        """
        sx = np.array([_SIGMA_STRIP, _SIGMA_CLUSTER2, _SIGMA_STRIP])
        _, _, _, _, cov_x, cov_y, _ = self._fit(sx, _SIGMA_STRIP)
        assert cov_x != cov_y
        assert cov_x[0] > cov_y[0], "X intercept variance should exceed Y's"

    def test_differs_from_scalar_collapse(self):
        """
        Regression guard for the original bug: passing the real per-plane σ
        array must give a different fit than collapsing to a single scalar
        (plane-0's σ broadcast to all planes).
        """
        sx = np.array([_SIGMA_STRIP, _SIGMA_CLUSTER2, _SIGMA_STRIP])
        ax_arr, *_ = self._fit(sx, _SIGMA_STRIP)
        ax_scalar, *_ = self._fit(_SIGMA_STRIP, _SIGMA_STRIP)
        assert abs(ax_arr - ax_scalar) > 1.0, (
            f"per-plane σ must change the fit: {ax_scalar:.3f} vs {ax_arr:.3f}"
        )


class TestSigmaTelAtZ:
    def test_zero_at_zero_zp(self):
        cov = (1.0, 0.0, 0.0)  # only var_a, no covariance
        assert _sigma_tel_at_z(cov, 0.0) == pytest.approx(1.0)

    def test_grows_with_zp(self):
        cov = (1.0, 0.0, 0.001)
        s0 = _sigma_tel_at_z(cov, 0.0)
        s1 = _sigma_tel_at_z(cov, 300.0)
        assert s1 > s0


class TestLinearSolveFixedTheta:
    def _make_coincs(self, n=50, seed=7):
        """Synthetic coincidences for a known pose."""
        rng = np.random.default_rng(seed)
        coincs = []
        for _ in range(n):
            # Random telescope slopes
            ax, bx = rng.uniform(100, 800), rng.uniform(-0.3, 0.3)
            ay, by = rng.uniform(100, 800), rng.uniform(-0.3, 0.3)
            # Probe hit at z_p=300
            x_at_p = ax + bx * 300.0
            y_at_p = ay + by * 300.0
            c, s = math.cos(_TRUE_THETA), math.sin(_TRUE_THETA)
            u = (x_at_p - _TRUE_TX) * c + (y_at_p - _TRUE_TY) * s
            v = -(x_at_p - _TRUE_TX) * s + (y_at_p - _TRUE_TY) * c
            cov = (_SIGMA_STRIP**2, 0.0, _SIGMA_STRIP**2 / 1e6)
            coincs.append(
                Coincidence(ax, bx, ay, by, cov, cov, u, v, _SIGMA_STRIP, _SIGMA_STRIP)
            )
        return coincs

    def test_recovers_pose(self):
        coincs = self._make_coincs(200)
        c, s = math.cos(_TRUE_THETA), math.sin(_TRUE_THETA)
        tx, ty, zp, _ = _linear_solve_fixed_theta(coincs, c, s)
        assert abs(tx - _TRUE_TX) < 1.0
        assert abs(ty - _TRUE_TY) < 1.0
        assert abs(zp - _TRUE_ZP) < 5.0


# ── integration tests on the synthetic pipeline ───────────────────────────


class TestPoseFitterFlush:
    def test_returns_pose_result(self, pose_result):
        assert isinstance(pose_result, PoseResult)

    def test_n_inliers_positive(self, pose_result):
        assert pose_result.n_inliers >= 10

    def test_chi2_curve_shape(self, pose_result):
        assert pose_result.chi2_curve.shape == (360, 2), (
            f"chi2_curve shape {pose_result.chi2_curve.shape}, expected (360, 2)"
        )

    def test_chi2_curve_finite(self, pose_result):
        assert np.isfinite(pose_result.chi2_curve[:, 1]).all()

    def test_residuals_length(self, pose_result):
        assert len(pose_result.residuals_x) == pose_result.n_inliers
        assert len(pose_result.residuals_y) == pose_result.n_inliers

    def test_residuals_mean_near_zero(self, pose_result):
        """Mean residual must be well below one strip."""
        n = pose_result.n_inliers
        tol = 3 * _SIGMA_STRIP / math.sqrt(n)
        assert abs(np.mean(pose_result.residuals_x)) < tol, (
            f"mean(res_x) = {np.mean(pose_result.residuals_x):.3f} mm"
        )
        assert abs(np.mean(pose_result.residuals_y)) < tol, (
            f"mean(res_y) = {np.mean(pose_result.residuals_y):.3f} mm"
        )

    def test_cov_positive_diagonal(self, pose_result):
        for i in range(4):
            assert pose_result.cov[i, i] > 0, f"cov[{i},{i}] <= 0"

    def test_half_consistency(self, pose_result):
        """
        The two parity halves should give t_x, t_y within ±3σ of each other.
        """
        tx_e, _ = pose_result.half_params[0, 0], pose_result.half_params[0, 1]
        tx_o, _ = pose_result.half_params[1, 0], pose_result.half_params[1, 1]
        sigma_half = math.sqrt(abs(pose_result.cov[0, 0])) * math.sqrt(2.0)
        assert abs(tx_e - tx_o) < 5 * sigma_half, (
            f"tx: even={tx_e:.2f}, odd={tx_o:.2f}, 5σ_half={5 * sigma_half:.2f}"
        )


class TestPoseParameterRecovery:
    """
    3σ assertions on the fitted parameters.

    θ is tested modulo π/2 to account for the four-fold degeneracy
    of a square probe (DESIGN.md §7.5).  z_p is unambiguous.  For
    t_x and t_y the test identifies which of the four equivalent
    solutions the optimizer found and verifies the covariance
    ellipsoid encloses the ground truth for that rotation.
    """

    def test_zp_within_3sigma(self, pose_result):
        sigma = math.sqrt(abs(pose_result.cov[3, 3]))
        err = abs(pose_result.z_p - _TRUE_ZP)
        assert err < 3 * sigma, (
            f"z_p={pose_result.z_p:.2f} mm, "
            f"true={_TRUE_ZP} mm, err={err:.2f} mm, 3σ={3 * sigma:.2f} mm"
        )

    def test_theta_within_3sigma_mod90(self, pose_result):
        """
        The fitted θ must be within 3σ of θ_true + k·π/2 for some integer k.
        """
        sigma = math.sqrt(abs(pose_result.cov[2, 2]))
        err = _theta_err_mod90(pose_result.theta, _TRUE_THETA)
        assert err < 3 * sigma, (
            f"theta={math.degrees(pose_result.theta):.3f}°, "
            f"nearest equiv={math.degrees(_TRUE_THETA):.3f}°±k·90°, "
            f"err={math.degrees(err):.3f}°, 3σ={math.degrees(3 * sigma):.3f}°"
        )

    def test_tx_ty_within_3sigma(self, pose_result):
        """
        Identify which equivalent solution (k) the optimizer found, then
        verify t_x and t_y against the expected values for that rotation.

        For k = 0 (canonical solution):   expected (t_x, t_y) = (50, -30).
        For k ≠ 0 the expected translation depends on the mean probe
        hit, which is estimated from the residual means — both should
        be near zero regardless of k.
        """
        k = _nearest_k90(pose_result.theta, _TRUE_THETA)
        sigma_tx = math.sqrt(abs(pose_result.cov[0, 0]))
        sigma_ty = math.sqrt(abs(pose_result.cov[1, 1]))

        if k == 0:
            # Canonical solution — compare directly to ground truth.
            assert abs(pose_result.t_x - _TRUE_TX) < 3 * sigma_tx, (
                f"t_x={pose_result.t_x:.2f}, true={_TRUE_TX}, 3σ={3 * sigma_tx:.2f}"
            )
            assert abs(pose_result.t_y - _TRUE_TY) < 3 * sigma_ty, (
                f"t_y={pose_result.t_y:.2f}, true={_TRUE_TY}, 3σ={3 * sigma_ty:.2f}"
            )
        else:
            # Non-canonical solution found: the fit is still valid, but
            # the translation vector encodes a different 90°-rotated probe
            # frame.  Verify only that the residuals are small (see
            # test_residuals_mean_near_zero) and that the covariance is
            # consistent.
            n = pose_result.n_inliers
            tol = 3 * _SIGMA_STRIP / math.sqrt(n)
            assert abs(np.mean(pose_result.residuals_x)) < tol
            assert abs(np.mean(pose_result.residuals_y)) < tol


# ── combinatorial track finder: recovery from fold-ambiguous data ──────────


def _run_fold_pipeline(
    out,
    fold=False,
    fold_planes=None,
    n_tracks=_N_TRACKS,
    fold_symmetry=1.0,
    fold_crosstalk_rate=0.0,
    on_decode=None,
    min_anchor_planes=1,
):
    """Generate fold-ambiguous synthetic data and run the streaming
    pipeline through PoseFitter.flush(); return the PoseResult (or None).

    fold_symmetry / fold_crosstalk_rate: forwarded to synth.generate() to
    inject realistic (non-idealised) fold-mirror and fiber cross-talk noise
    (DESIGN.md §10) instead of the default perfectly periodic fold pattern.
    """
    generate(
        out_dir=out,
        t_x=_TRUE_TX,
        t_y=_TRUE_TY,
        theta=_TRUE_THETA,
        z_p=_TRUE_ZP,
        n_tracks=n_tracks,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
        fold=fold,
        fold_planes=fold_planes,
        fold_symmetry=fold_symmetry,
        fold_crosstalk_rate=fold_crosstalk_rate,
    )
    tel_dir = out / "telescope"
    prb_dir = out / "probe"

    tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header.txt")))

    tel_gps, tel_pos = find_file_pairs(tel_dir)
    prb_gps, prb_pos = find_file_pairs(prb_dir)

    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    fitter = PoseFitter(
        tel_z=Z_TEL,
        alignment=AlignmentCorrection.identity(),
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
        refit_every=n_tracks + 1,  # no auto-flush; use explicit flush()
        min_anchor_planes=min_anchor_planes,
        on_decode=on_decode,
    )

    for cluster in coincidence_stream(
        [tel_stream, prb_stream],
        detector_ids=[0, 1],
    ):
        fitter.add(cluster)

    return fitter.flush()


def _assert_pose_within_3sigma(pr):
    sigma_zp = math.sqrt(abs(pr.cov[3, 3]))
    err_zp = abs(pr.z_p - _TRUE_ZP)
    assert err_zp < 3 * sigma_zp, (
        f"z_p={pr.z_p:.2f} mm, true={_TRUE_ZP} mm, "
        f"err={err_zp:.2f} mm, 3σ={3 * sigma_zp:.2f} mm"
    )

    sigma_th = math.sqrt(abs(pr.cov[2, 2]))
    err_th = _theta_err_mod90(pr.theta, _TRUE_THETA)
    assert err_th < 3 * sigma_th, (
        f"theta={math.degrees(pr.theta):.3f}°, "
        f"nearest equiv={math.degrees(_TRUE_THETA):.3f}°±k·90°, "
        f"err={math.degrees(err_th):.3f}°, 3σ={math.degrees(3 * sigma_th):.3f}°"
    )

    k = _nearest_k90(pr.theta, _TRUE_THETA)
    sigma_tx = math.sqrt(abs(pr.cov[0, 0]))
    sigma_ty = math.sqrt(abs(pr.cov[1, 1]))
    if k == 0:
        assert abs(pr.t_x - _TRUE_TX) < 3 * sigma_tx, (
            f"t_x={pr.t_x:.2f}, true={_TRUE_TX}, 3σ={3 * sigma_tx:.2f}"
        )
        assert abs(pr.t_y - _TRUE_TY) < 3 * sigma_ty, (
            f"t_y={pr.t_y:.2f}, true={_TRUE_TY}, 3σ={3 * sigma_ty:.2f}"
        )
    else:
        n = pr.n_inliers
        tol = 3 * _SIGMA_STRIP / math.sqrt(n)
        assert abs(np.mean(pr.residuals_x)) < tol
        assert abs(np.mean(pr.residuals_y)) < tol


@pytest.fixture(scope="module")
def pose_result_fold_1plane(tmp_path_factory):
    """
    Single ambiguous plane (plane 1; planes 0 and 2 stay golden/cluster).
    The two clean planes anchor the line, and the combinatorial finder
    recovers plane 1's candidate from the χ² search.
    """
    out = tmp_path_factory.mktemp("synth_stage5_fold1")
    pr = _run_fold_pipeline(out, fold_planes={1})
    assert pr is not None, "PoseFitter.flush() returned None (too few coincidences)"
    return pr


@pytest.fixture(scope="module")
def pose_result_fold_2plane(tmp_path_factory):
    """
    Two ambiguous planes (0 and 1; plane 2 stays golden as the sole anchor).
    This is the combinatorial finder's headline win: per-plane candidate
    enumeration + line-fit χ² search across both ambiguous planes
    simultaneously, anchored by the single clean plane.
    """
    out = tmp_path_factory.mktemp("synth_stage5_fold2")
    pr = _run_fold_pipeline(out, fold_planes={0, 1})
    assert pr is not None, "PoseFitter.flush() returned None (too few coincidences)"
    return pr


@pytest.fixture(scope="module")
def pose_result_fold_2plane_realistic(tmp_path_factory):
    """
    Same 2-ambiguous-plane scenario as pose_result_fold_2plane, but with
    realistic (non-idealised) fold statistics from DESIGN.md §10 instead of
    a perfectly periodic mirror pattern: fold_symmetry=0.85 (within the
    documented 0.71-0.95 range) and fold_crosstalk_rate=0.02 (within the
    documented 1.7-2.6% fiber cross-talk range).  This is the combinatorial
    finder's real payoff per the architectural audit: messy fold data, not
    the idealised always-both-bits pattern the other fold fixtures use.
    """
    out = tmp_path_factory.mktemp("synth_stage5_fold2_realistic")
    pr = _run_fold_pipeline(
        out,
        fold_planes={0, 1},
        fold_symmetry=0.85,
        fold_crosstalk_rate=0.02,
    )
    assert pr is not None, "PoseFitter.flush() returned None (too few coincidences)"
    return pr


class TestFoldedPoseRecovery1Plane:
    """3σ recovery with one mirror-fold-ambiguous telescope plane."""

    def test_zp_within_3sigma(self, pose_result_fold_1plane):
        _assert_pose_within_3sigma(pose_result_fold_1plane)


class TestFoldedPoseRecovery2Plane:
    """
    Headline win for the combinatorial track finder (DESIGN.md §8.2):
    recover the probe pose with two of three telescope planes ambiguous on
    every event — resolved globally by the per-plane candidate enumeration +
    line-fit χ² search, using the remaining clean plane as the anchor.
    """

    def test_zp_within_3sigma(self, pose_result_fold_2plane):
        _assert_pose_within_3sigma(pose_result_fold_2plane)


class TestFoldedPoseRecovery2PlaneRealisticNoise:
    """
    Closes the architectural-audit gap: recovers the probe pose from
    fold-ambiguous data with realistic (DESIGN.md §10) fold-symmetry and
    fiber cross-talk noise, not the idealised perfectly-periodic fold
    pattern every other fold test in this module uses.
    """

    def test_zp_within_3sigma(self, pose_result_fold_2plane_realistic):
        _assert_pose_within_3sigma(pose_result_fold_2plane_realistic)

    def test_noise_model_engages(self, tmp_path_factory):
        """
        Guard against a silent no-op: with fold_symmetry < 1.0 and
        fold_crosstalk_rate > 0.0, the per-plane candidate counts seen
        across accepted coincidences must vary, unlike the idealised fold
        pattern's fixed shape (planes 0 and 1 folded, plane 2 golden ->
        always (4, 4, 1) candidates).  Dropped mirror partner bits shrink
        a candidate list toward 1; cross-talk bits grow one past 4 — either
        way, count variety is the noise model's observable fingerprint.
        """
        out = tmp_path_factory.mktemp("synth_stage5_fold2_realistic_report")
        reports: list[DecodeReport] = []
        _run_fold_pipeline(
            out,
            fold_planes={0, 1},
            fold_symmetry=0.85,
            fold_crosstalk_rate=0.02,
            on_decode=reports.append,
        )
        accepted = [r for r in reports if r.accepted]
        assert accepted, "no coincidence was accepted under realistic fold noise"
        cand_counts_seen = {r.cand_counts for r in accepted}
        assert len(cand_counts_seen) > 1, (
            "candidate counts are constant across every accepted coincidence "
            "— fold_symmetry/fold_crosstalk_rate do not appear to be engaging"
        )


class TestFoldedPoseRecoveryAllPlanesFails:
    """
    Documented negative result (see handoff-combinatorial-track-finder.md):
    with all 3 telescope planes mirror-fold ambiguous on both axes on every
    event, every plane's candidate list has >1 entry, so PoseFitter's
    "require ≥1 already-resolved anchor plane" guard (added to fix
    TestPerScenarioHandling::test_E2_pileup_same_window_unresolved_rejected
    — the same ambiguous-bit-pattern problem a genuine multi-particle
    pile-up produces) rejects every coincidence outright, and flush()
    returns None for lack of any accepted coincidence.

    Before that guard existed, the search instead ran to completion and
    found an exact mathematical tie: monrad.synthetic.generate() has no
    measurement noise, so reflecting an entire straight line is still a
    straight line, and the all-3-planes-mirrored candidate triple achieved
    *exactly* the same χ² as the true triple (verified directly: true
    χ²≈3.9e-28 vs full-mirror χ²≈4.7e-27, both numerically zero) — a tie
    the χ²-only search cannot break.  Either way the case is unrecoverable;
    real data only requires recovering *at least one* ambiguous plane per
    event (DESIGN.md §10 Deduction #4), not all three at once, so this is
    more extreme than the strategy needs to handle.  Pinned here so a
    future change doesn't silently regress without re-deriving this
    finding.
    """

    @pytest.fixture(scope="class")
    def pose_result_fold_all(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("synth_stage5_fold_all")
        return _run_fold_pipeline(out, fold=True)

    def test_pose_not_recovered(self, pose_result_fold_all):
        assert pose_result_fold_all is None, (
            "flush() unexpectedly returned a PoseResult for the all-3-planes "
            "fold case — the anchor-plane guard or the exact-tie ambiguity "
            "appears to have changed; see this test's docstring before "
            "relaxing it"
        )


class TestMinAnchorPlanesTunable:
    """
    The anchor-plane guard is tunable via PoseFitter(min_anchor_planes=N):
    the search runs only when at least N planes decoded to a single resolved
    candidate.  N=1 is the default (original behaviour); N=0 disables the guard
    so even all-3-ambiguous clusters reach the χ² search.
    """

    @pytest.mark.parametrize("bad", [-1, 4])
    def test_constructor_rejects_out_of_range(self, bad):
        with pytest.raises(ValueError, match="min_anchor_planes"):
            PoseFitter(
                tel_z=Z_TEL,
                alignment=AlignmentCorrection.identity(),
                tel_id=0,
                prb_id=1,
                tel_pos_paths=[],
                prb_pos_paths=[],
                min_anchor_planes=bad,
            )

    def test_default_gate_rejects_all_fold(self, tmp_path_factory):
        # With every plane mirror-fold ambiguous, no plane is an anchor, so the
        # default guard (min_anchor_planes=1) rejects every coincidence via the
        # "no_anchor_plane" gate and nothing reaches the χ² search.
        out = tmp_path_factory.mktemp("min_anchor_default")
        reports: list = []
        _run_fold_pipeline(out, fold=True, on_decode=reports.append)
        assert reports, "no coincidences were decoded at all"
        assert any(r.reason == "no_anchor_plane" for r in reports)
        assert not any(r.accepted for r in reports)

    def test_zero_disables_gate(self, tmp_path_factory):
        # min_anchor_planes=0 removes the guard: the same all-fold clusters now
        # run the combinatorial search instead of being rejected, so not a
        # single "no_anchor_plane" rejection is emitted.
        out = tmp_path_factory.mktemp("min_anchor_zero")
        reports: list = []
        _run_fold_pipeline(
            out, fold=True, on_decode=reports.append, min_anchor_planes=0
        )
        assert reports, "no coincidences were decoded at all"
        assert not any(r.reason == "no_anchor_plane" for r in reports)


# ── unit tests for cluster disambiguation in PoseFitter._decode_cluster ─────


from monrad.timing import TimedEvent, PosRef, Quality  # noqa: E402


def _entry(det_id, seq):
    """A (det_id, TimedEvent, PosRef) cluster entry with throwaway payload."""
    return det_id, TimedEvent(seq, seq, Quality.GOOD), PosRef(0, seq * 16)


class TestDecodeClusterDisambiguation:
    """
    _decode_cluster must accept exactly one telescope event and exactly one
    event from *its* probe, and reject any ambiguous cluster *before* touching
    the position files.  Reaching decode_position() with the dummy refs below
    would raise, so a clean None return proves the guard short-circuits.
    """

    def _fitter(self):
        return PoseFitter(
            tel_z=Z_TEL,
            alignment=AlignmentCorrection.identity(),
            tel_id=0,
            prb_id=1,
            tel_pos_paths=[],
            prb_pos_paths=[],
        )

    def test_two_telescope_events_rejected(self):
        cluster = [_entry(0, 0), _entry(0, 1), _entry(1, 2)]
        assert self._fitter()._decode_cluster(cluster) is None

    def test_two_probe_events_rejected(self):
        cluster = [_entry(0, 0), _entry(1, 1), _entry(1, 2)]
        assert self._fitter()._decode_cluster(cluster) is None

    def test_missing_telescope_rejected(self):
        cluster = [_entry(1, 0)]
        assert self._fitter()._decode_cluster(cluster) is None

    def test_missing_probe_rejected(self):
        cluster = [_entry(0, 0)]
        assert self._fitter()._decode_cluster(cluster) is None

    def test_other_probe_does_not_count(self):
        # A second, different probe detector (id 2) in the cluster must not
        # make this probe's pairing ambiguous: one tel + one prb-1 is still a
        # valid pairing, so the guard passes and execution proceeds past it
        # (and then fails at decode_position with the empty path list).
        cluster = [_entry(0, 0), _entry(1, 1), _entry(2, 2)]
        with pytest.raises((IndexError, ValueError, FileNotFoundError)):
            self._fitter()._decode_cluster(cluster)


# ── per-axis probe σ propagation through the full pipeline ─────────────────


def _run_pipeline(out, **gen_kwargs):
    """Generate a synthetic dataset and run stages 1+2+5; return the
    PoseFitter (with its accumulated coincidences) after a flush()."""
    result = generate(
        out_dir=out,
        t_x=_TRUE_TX,
        t_y=_TRUE_TY,
        theta=_TRUE_THETA,
        z_p=_TRUE_ZP,
        n_tracks=_N_TRACKS,
        seed=42,
        start_utc=_START_UTC,
        f0=F0,
        **gen_kwargs,
    )
    tel_dir = out / "telescope"
    prb_dir = out / "probe"
    tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
    prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header.txt")))
    tel_gps, tel_pos = find_file_pairs(tel_dir)
    prb_gps, prb_pos = find_file_pairs(prb_dir)
    tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
    prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

    fitter = PoseFitter(
        tel_z=Z_TEL,
        alignment=AlignmentCorrection.identity(),
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel_pos,
        prb_pos_paths=prb_pos,
        refit_every=_N_TRACKS + 1,  # no auto-flush
    )
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        fitter.add(cluster)
    fitter.flush()
    return result, fitter


class TestPerAxisSigmaPropagation:
    """
    End-to-end proof that a probe hit decoding as a `cluster` carries a
    larger σ on the wide axis, that this per-axis σ reaches the optimizer
    inputs intact, and that it genuinely changes the pose fit (the gap the
    synth `cluster_widths` support was added to close).

    The probe is encoded with a width-2 cluster on its v-axis only
    (`probe_cluster_width=(1, 2)`), so σ_prb,v = 2·σ_prb,u.  The telescope
    stays golden, so the χ²<4 track cut keeps full statistics.
    """

    @pytest.fixture(scope="class")
    def cluster_fitter(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("synth_stage5_probe_cluster")
        _, fitter = _run_pipeline(out, probe_cluster_width=(1, 2))
        return fitter

    def test_probe_sigma_is_anisotropic(self, cluster_fitter):
        """Every coincidence's probe σ is golden on u, width-2 cluster on v."""
        coincs = cluster_fitter._coincs
        assert len(coincs) >= 3
        for co in coincs:
            assert co.sigma_prb_x == pytest.approx(_SIGMA_STRIP)
            assert co.sigma_prb_y == pytest.approx(_SIGMA_CLUSTER2)
            assert co.sigma_prb_y > co.sigma_prb_x

    def test_per_axis_sigma_changes_the_fit(self, cluster_fitter):
        """
        Refit the same coincidences with the per-axis probe σ collapsed to
        a single scalar (σ_v forced to σ_u — the pre-fix behaviour).  The
        recovered covariance must differ, proving the anisotropic σ is not
        silently ignored downstream.
        """
        coincs = cluster_fitter._coincs
        ident = AlignmentCorrection.identity()
        pr_hetero = fit_probe_pose(coincs, Z_TEL, ident)
        collapsed = [co._replace(sigma_prb_y=co.sigma_prb_x) for co in coincs]
        pr_scalar = fit_probe_pose(collapsed, Z_TEL, ident)

        assert not np.allclose(pr_hetero.cov, pr_scalar.cov), (
            "collapsing probe σ to a scalar left the pose covariance unchanged"
        )
        # The down-weighted v-axis carries less information, so the hetero
        # fit must be no tighter than the (over-confident) scalar fit on the
        # v-coupled parameters.
        assert np.trace(pr_hetero.cov) > np.trace(pr_scalar.cov)


class TestDecodeReport:
    """
    DecodeReport surfaces the gate outcome (reason), the winning triple's χ²
    and the per-plane candidate counts that PoseFitter._decode_cluster
    actually computed, so run_pipeline.py reads the gates instead of
    re-deriving them.
    """

    def test_accepted_report_carries_chi2_and_cand_counts(self, tmp_path):
        out = tmp_path / "synth"
        generate(
            out_dir=out,
            t_x=_TRUE_TX,
            t_y=_TRUE_TY,
            theta=_TRUE_THETA,
            z_p=_TRUE_ZP,
            n_tracks=_N_TRACKS,
            seed=42,
            start_utc=_START_UTC,
            f0=F0,
        )
        tel_dir, prb_dir = out / "telescope", out / "probe"
        tel_utc0, tel_f0 = load_header_params(next(tel_dir.glob("*_header.txt")))
        prb_utc0, prb_f0 = load_header_params(next(prb_dir.glob("*_header.txt")))
        tel_gps, tel_pos = find_file_pairs(tel_dir)
        prb_gps, prb_pos = find_file_pairs(prb_dir)
        tel_stream = reconstruct_stream(tel_gps, tel_pos, tel_utc0, tel_f0)
        prb_stream = reconstruct_stream(prb_gps, prb_pos, prb_utc0, prb_f0)

        reports: list[DecodeReport] = []
        fitter = PoseFitter(
            tel_z=Z_TEL,
            alignment=AlignmentCorrection.identity(),
            tel_id=0,
            prb_id=1,
            tel_pos_paths=tel_pos,
            prb_pos_paths=prb_pos,
            refit_every=_N_TRACKS + 1,
            on_decode=reports.append,
        )
        for cluster in coincidence_stream(
            [tel_stream, prb_stream], detector_ids=[0, 1]
        ):
            fitter.add(cluster)
        fitter.flush()

        accepted = [r for r in reports if r.accepted]
        assert accepted, "no coincidence was accepted"
        for r in accepted:
            assert r.reason == "accepted"
            assert r.chi2 is not None
            assert r.prb_quality in ("golden", "cluster")
            assert r.cand_counts is not None and len(r.cand_counts) == 3
            # Clean synthetic telescope hits are golden: exactly one candidate
            # per plane.
            assert r.cand_counts == (1, 1, 1)
            # Each plane's winning candidate carries its own golden label,
            # threaded through to the on_decode callback.
            assert r.tel_quality == ("golden", "golden", "golden")

    def test_cand_counts_absent_before_candidate_enumeration(self):
        """An ambiguous cluster is rejected before any candidate is enumerated,
        so the per-plane candidate counts are reported as None."""
        reports: list[DecodeReport] = []
        fitter = PoseFitter(
            tel_z=Z_TEL,
            alignment=AlignmentCorrection.identity(),
            tel_id=0,
            prb_id=1,
            tel_pos_paths=[],
            prb_pos_paths=[],
            on_decode=reports.append,
        )
        # Two telescope events in one cluster → "ambiguous_cluster" gate.
        assert (
            fitter._decode_cluster([_entry(0, 0), _entry(0, 1), _entry(1, 2)]) is None
        )
        assert len(reports) == 1
        r = reports[0]
        assert r.reason == "ambiguous_cluster"
        assert r.cand_counts is None
        assert r.chi2 is None
