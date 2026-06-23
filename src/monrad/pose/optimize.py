"""
Stage 5 (part) — the probe pose optimizer.

fit_probe_pose(coincidences, tel_z, alignment) -> PoseResult
    Implements DESIGN.md §8.4 four-step optimizer plus the Mahalanobis outlier
    cut and stratified-half consistency test.
"""

import math

import numpy as np
from scipy.optimize import least_squares

from ..alignment import AlignmentCorrection
from .types import Coincidence, PoseResult

_MAHAL_CUT = 4.0  # Mahalanobis distance outlier threshold — DESIGN.md §8.4


# ── Helpers ───────────────────────────────────────────────────────────────


def _sigma_tel_at_z(
    cov_ab: tuple[float, float, float],
    z_p: float,
) -> float:
    """Variance of telescope line prediction x_pred = a + b*z at z = z_p."""
    va, cab, vb = cov_ab
    return va + 2.0 * z_p * cab + z_p * z_p * vb


def _tel_line_fit(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    sigma_x: float | np.ndarray,
    sigma_y: float | np.ndarray,
    z_y_arr: np.ndarray | None = None,
) -> tuple[
    float,
    float,
    float,
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]:
    """
    Weighted least-squares fit of x(z) and y(z) through n telescope planes.

    Each plane carries its own per-axis position uncertainty (DESIGN.md §6.4),
    so the X and Y fits use independent diagonal weight matrices and yield
    distinct covariances (DESIGN.md §8.2).

    sigma_x and sigma_y may each be a scalar (uniform across planes) or an
    (n,) array of per-plane sigmas.

    z_arr is the plane z used for the x(z) fit.  z_y_arr, if given, is the
    plane z for the y(z) fit; it defaults to z_arr.  The two differ only when
    an out-of-plane tilt has been folded in (DESIGN.md §7.3/§10): a tilt about
    the y-axis shifts the effective z of the x measurement and a tilt about
    the x-axis shifts it for y, so each axis is fit in its own corrected frame.

    Returns (a_x, b_x, a_y, b_y, cov_x, cov_y, chi2_total).
    chi2_total is the combined x+y chi² (ndof = 2*(n-2)).
    cov_x / cov_y = (var_a, cov_ab, var_b) for the respective axis.
    """
    n = len(z_arr)
    z_y = z_arr if z_y_arr is None else z_y_arr
    wx = 1.0 / np.broadcast_to(np.asarray(sigma_x, dtype=float), (n,)) ** 2
    wy = 1.0 / np.broadcast_to(np.asarray(sigma_y, dtype=float), (n,)) ** 2

    A_x = np.column_stack([np.ones(n), z_arr])  # (n, 2)
    A_y = np.column_stack([np.ones(n), z_y])
    AtA_x = A_x.T @ (wx[:, None] * A_x)  # (2, 2)
    AtA_y = A_y.T @ (wy[:, None] * A_y)
    Atx = A_x.T @ (wx * x_arr)
    Aty = A_y.T @ (wy * y_arr)

    px = np.linalg.solve(AtA_x, Atx)  # [a_x, b_x]
    py = np.linalg.solve(AtA_y, Aty)  # [a_y, b_y]

    cov2_x = np.linalg.inv(AtA_x)  # (2, 2) covariance of [a_x, b_x]
    cov2_y = np.linalg.inv(AtA_y)
    cov_x = (float(cov2_x[0, 0]), float(cov2_x[0, 1]), float(cov2_x[1, 1]))
    cov_y = (float(cov2_y[0, 0]), float(cov2_y[0, 1]), float(cov2_y[1, 1]))

    rx = x_arr - A_x @ px
    ry = y_arr - A_y @ py
    chi2 = float(np.sum(wx * rx**2) + np.sum(wy * ry**2))

    return (
        float(px[0]),
        float(px[1]),
        float(py[0]),
        float(py[1]),
        cov_x,
        cov_y,
        chi2,
    )


def _fit_triple(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    sigma_x: np.ndarray,
    sigma_y: np.ndarray,
    alignment: AlignmentCorrection,
    z_corr: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]:
    """
    Apply per-plane alignment correction and fit a straight line through one
    telescope (x, y) triple — one candidate position per plane.

    Mirrors the alignment block PoseFitter._decode_cluster used to apply to
    its single resolved hit per plane: each plane's raw position is shifted
    by its fitted (delta_x, delta_y), and each axis is fit at its own
    z-frame, shifted by tilt_y·x / tilt_x·y (DESIGN.md §7.3/§10) so an
    out-of-plane tilt is removed exactly. Returns the same tuple as
    _tel_line_fit: (a_x, b_x, a_y, b_y, cov_x, cov_y, chi2).

    z_corr is the alignment-corrected plane z (alignment.corrected_z_tel(tel_z));
    it is constant across every triple of one event, so the caller computes it
    once and passes it in rather than re-deriving it per triple.
    """
    corr = alignment
    x_arr = x_raw - np.array([corr.planes[k].delta_x for k in range(3)])
    y_arr = y_raw - np.array([corr.planes[k].delta_y for k in range(3)])

    z_x_arr = z_corr + np.array([corr.planes[k].tilt_y * x_arr[k] for k in range(3)])
    z_y_arr = z_corr + np.array([corr.planes[k].tilt_x * y_arr[k] for k in range(3)])

    return _tel_line_fit(x_arr, y_arr, z_x_arr, sigma_x, sigma_y, z_y_arr=z_y_arr)


def _linear_solve_fixed_theta(
    coincs: list[Coincidence],
    c: float,  # cos θ
    s: float,  # sin θ
) -> tuple[float, float, float, float]:
    """
    Solve for (t_x, t_y, z_p) at fixed θ using probe-only weights.

    The model (residual = 0):
      t_x - b_x·z_p = a_x - (u·c - v·s)
      t_y - b_y·z_p = a_y - (u·s + v·c)

    Each equation is weighted by the per-axis probe uncertainty
    (DESIGN.md §8.4 step 1: W_i held at the probe-only weight Σ_probe,i⁻¹).
    The x and y rows therefore carry independent weights, and chi2 is the
    already-normalised weighted sum of squares.

    Returns (t_x, t_y, z_p, chi2).
    """
    N = len(coincs)
    A = np.zeros((2 * N, 3))
    b_vec = np.zeros(2 * N)
    wsqrt = np.zeros(2 * N)
    for i, co in enumerate(coincs):
        A[2 * i, 0] = 1.0
        A[2 * i, 2] = -co.b_x
        b_vec[2 * i] = co.a_x - (co.u * c - co.v * s)
        wsqrt[2 * i] = 1.0 / co.sigma_prb_x
        A[2 * i + 1, 1] = 1.0
        A[2 * i + 1, 2] = -co.b_y
        b_vec[2 * i + 1] = co.a_y - (co.u * s + co.v * c)
        wsqrt[2 * i + 1] = 1.0 / co.sigma_prb_y

    Aw = A * wsqrt[:, None]
    bw = b_vec * wsqrt
    params, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    res = Aw @ params - bw
    chi2 = float(np.dot(res, res))

    return float(params[0]), float(params[1]), float(params[2]), chi2


def _weighted_residuals(
    params: np.ndarray,
    coincs: list[Coincidence],
) -> np.ndarray:
    """
    Normalised residuals for scipy.optimize.least_squares (LM step).

    params = [t_x, t_y, theta, z_p].
    Each coincidence contributes two elements: r_x/σ_x and r_y/σ_y,
    where σ² = σ_prb² + σ_tel²(z_p) (full weight, not probe-only).
    """
    tx, ty, theta, zp = params
    c, s = math.cos(theta), math.sin(theta)
    res = np.empty(2 * len(coincs))
    for i, co in enumerate(coincs):
        x_pred = co.a_x + co.b_x * zp
        y_pred = co.a_y + co.b_y * zp
        x_meas = tx + co.u * c - co.v * s
        y_meas = ty + co.u * s + co.v * c
        var_x = co.sigma_prb_x**2 + _sigma_tel_at_z(co.cov_ab_x, zp)
        var_y = co.sigma_prb_y**2 + _sigma_tel_at_z(co.cov_ab_y, zp)
        res[2 * i] = (x_meas - x_pred) / math.sqrt(max(var_x, 1e-12))
        res[2 * i + 1] = (y_meas - y_pred) / math.sqrt(max(var_y, 1e-12))
    return res


# ── Main fit function ─────────────────────────────────────────────────────


def fit_probe_pose(
    coincidences: list[Coincidence],
    tel_z: np.ndarray,
    alignment: AlignmentCorrection,
) -> PoseResult:
    """
    Four-step probe pose optimizer.  Implements DESIGN.md §8.4.

    Step 1 — coarse θ scan at 1° over [−180°, 180°).
    Step 2 — diagnostic χ²(θ) curve stored in PoseResult.
    Step 3 — fine θ scan at 0.01° over ±2° around global minimum.
    Step 4 — Levenberg-Marquardt polish on all four parameters,
              with full σ²(z_p) weights.

    After step 4: Mahalanobis outlier cut at d > 4, one-pass refit.

    Parameters
    ----------
    coincidences : pre-decoded and quality-filtered telescope-probe pairs
    tel_z        : telescope plane z-coordinates (mm), shape (3,)
    alignment    : applied before building coincidences in PoseFitter;
                   carried here for API completeness
    """
    coincs = coincidences
    if len(coincs) < 3:
        raise ValueError(f"fit_probe_pose needs ≥ 3 coincidences, got {len(coincs)}")

    # ── Step 1: coarse θ scan at 1° ──────────────────────────────
    theta_coarse = np.deg2rad(np.arange(-180.0, 180.0, 1.0))
    chi2_curve = np.empty((len(theta_coarse), 2))
    best_chi2 = math.inf
    best_theta = 0.0
    best_tx = 0.0
    best_ty = 0.0
    best_zp = float(np.mean(tel_z))  # neutral starting guess

    for j, th in enumerate(theta_coarse):
        c, s = math.cos(th), math.sin(th)
        tx, ty, zp, chi2 = _linear_solve_fixed_theta(coincs, c, s)
        chi2_curve[j, 0] = th
        chi2_curve[j, 1] = chi2
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_theta = th
            best_tx, best_ty, best_zp = tx, ty, zp

    # ── Step 2: χ²(θ) diagnostic curve stored above ───────────────

    # ── Step 3: fine θ scan at 0.01° over ±2° ────────────────────
    theta_fine = best_theta + np.deg2rad(np.arange(-2.0, 2.01, 0.01))
    for th in theta_fine:
        c, s = math.cos(th), math.sin(th)
        tx, ty, zp, chi2 = _linear_solve_fixed_theta(coincs, c, s)
        if chi2 < best_chi2:
            best_chi2 = chi2
            best_theta = th
            best_tx, best_ty, best_zp = tx, ty, zp

    # ── Step 4: LM polish on all four parameters ──────────────────
    x0 = np.array([best_tx, best_ty, best_theta, best_zp])
    opt = least_squares(
        _weighted_residuals,
        x0,
        args=(coincs,),
        method="lm",
    )
    tx_lm, ty_lm, theta_lm, zp_lm = opt.x

    # Covariance from the Gram matrix of normalised-residual Jacobian.
    J = opt.jac  # (2N, 4)
    JtJ = J.T @ J
    try:
        cov = np.linalg.inv(JtJ)
    except np.linalg.LinAlgError:
        cov = np.full((4, 4), np.nan)

    # ── Mahalanobis outlier cut ───────────────────────────────────
    c_lm, s_lm = math.cos(theta_lm), math.sin(theta_lm)
    maha = np.empty(len(coincs))
    for i, co in enumerate(coincs):
        x_pred = co.a_x + co.b_x * zp_lm
        y_pred = co.a_y + co.b_y * zp_lm
        x_meas = tx_lm + co.u * c_lm - co.v * s_lm
        y_meas = ty_lm + co.u * s_lm + co.v * c_lm
        var_x = max(co.sigma_prb_x**2 + _sigma_tel_at_z(co.cov_ab_x, zp_lm), 1e-12)
        var_y = max(co.sigma_prb_y**2 + _sigma_tel_at_z(co.cov_ab_y, zp_lm), 1e-12)
        maha[i] = math.sqrt(
            (x_meas - x_pred) ** 2 / var_x + (y_meas - y_pred) ** 2 / var_y
        )

    mask = maha <= _MAHAL_CUT
    inliers = [co for co, m in zip(coincs, mask) if m]
    if len(inliers) < 3:
        inliers = list(coincs)  # fallback: keep all
    n_inliers = len(inliers)

    if n_inliers < len(coincs):
        # One-pass refit on inliers
        x0_in = np.array([tx_lm, ty_lm, theta_lm, zp_lm])
        opt2 = least_squares(
            _weighted_residuals,
            x0_in,
            args=(inliers,),
            method="lm",
        )
        tx_lm, ty_lm, theta_lm, zp_lm = opt2.x
        JtJ2 = opt2.jac.T @ opt2.jac
        try:
            cov = np.linalg.inv(JtJ2)
        except np.linalg.LinAlgError:
            cov = np.full((4, 4), np.nan)

    # ── Final residuals ───────────────────────────────────────────
    c_f, s_f = math.cos(theta_lm), math.sin(theta_lm)
    res_x = np.empty(n_inliers)
    res_y = np.empty(n_inliers)
    for i, co in enumerate(inliers):
        res_x[i] = (tx_lm + co.u * c_f - co.v * s_f) - (co.a_x + co.b_x * zp_lm)
        res_y[i] = (ty_lm + co.u * s_f + co.v * c_f) - (co.a_y + co.b_y * zp_lm)

    # ── Stratified-half consistency test ─────────────────────────
    # Split inliers by event-index parity; fit each half at theta_lm
    # using the fast linear solve.
    half_params = np.zeros((2, 4))
    even = [co for i, co in enumerate(inliers) if i % 2 == 0]
    odd = [co for i, co in enumerate(inliers) if i % 2 == 1]
    for j, half in enumerate((even, odd)):
        if len(half) >= 3:
            tx_h, ty_h, zp_h, _ = _linear_solve_fixed_theta(half, c_f, s_f)
            half_params[j] = [tx_h, ty_h, theta_lm, zp_h]

    return PoseResult(
        t_x=float(tx_lm),
        t_y=float(ty_lm),
        theta=float(theta_lm),
        z_p=float(zp_lm),
        cov=cov,
        chi2_curve=chi2_curve,
        residuals_x=res_x,
        residuals_y=res_y,
        n_inliers=n_inliers,
        half_params=half_params,
        inliers=inliers,
    )
