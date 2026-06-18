"""
Stage 5 — probe pose fit.

Public API
----------
PoseResult
    Dataclass: fitted (t_x, t_y, theta, z_p), 4×4 covariance,
    chi²(θ) curve, final residuals, n_inliers, half-consistency params.

PoseFitter
    .add(cluster)               -> PoseResult | None
    .flush()                    -> PoseResult | None
    .update_alignment(corr)     -> None

fit_probe_pose(coincidences, tel_z, alignment) -> PoseResult
    Implements DESIGN.md §7.4 four-step optimizer.
"""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
from scipy.optimize import least_squares

from .stage3 import (
    GOOD_QUALITIES,
    decode_position,
    reconstruct_plane_candidates,
)
from .stage4 import AlignmentCorrection

_MAHAL_CUT = 4.0  # Mahalanobis distance outlier threshold — DESIGN.md §7.4
_CHI2_TRACK = 4.0  # telescope line-fit χ² threshold — DESIGN.md §7.2


# ── Internal data structure ───────────────────────────────────────────────


class Coincidence(NamedTuple):
    """Per-coincidence data for the pose optimizer."""

    a_x: float  # telescope x(z) = a_x + b_x*z
    b_x: float
    a_y: float  # telescope y(z) = a_y + b_y*z
    b_y: float
    # Covariance of (a, b) per axis: (var_a, cov_ab, var_b).  X and Y differ
    # because each plane's hit sigma is per-axis (DESIGN.md §6.4, §8.2): the
    # two axes share the same z values but not the same weights.
    cov_ab_x: tuple[float, float, float]
    cov_ab_y: tuple[float, float, float]
    u: float  # probe u-coordinate (mm)
    v: float  # probe v-coordinate (mm)
    sigma_prb_x: float  # probe position uncertainty along x (mm)
    sigma_prb_y: float  # probe position uncertainty along y (mm)


class DecodeReport(NamedTuple):
    """
    Outcome of one PoseFitter._decode_cluster call, for instrumentation
    (e.g. scripts/run_pipeline.py's Stage 3 table, track_coincidence_loss.py)
    so diagnostics read the gates _decode_cluster actually applied instead
    of re-deriving their own copy of its logic.

    reason is one of: "ambiguous_cluster", "zero_candidate_plane",
    "no_anchor_plane", "chi2_track_cut", "probe_quality", "accepted".
    cand_counts/chi2/prb_quality/winning_quality/winning_tot are None when
    the cluster was rejected before that quantity was computed.

    winning_quality/winning_tot describe the per-plane PlaneCandidate the
    χ² search actually picked (stage3.PlaneCandidate.quality/tot_x/tot_y) —
    available whenever a best triple was found, i.e. from "chi2_track_cut"
    onward, even if that triple was then rejected by a later gate.
    """

    accepted: bool
    reason: str
    cand_counts: tuple[int, int, int] | None
    chi2: float | None
    prb_quality: str | None
    winning_quality: tuple[str, str, str] | None
    winning_tot: tuple[tuple[int, int], tuple[int, int], tuple[int, int]] | None


# ── Result bundle ─────────────────────────────────────────────────────────


@dataclass
class PoseResult:
    """
    Full output of fit_probe_pose().  Implements DESIGN.md §7.7.

    Parameter order in `cov`: [t_x, t_y, theta, z_p].
    """

    t_x: float  # mm
    t_y: float  # mm
    theta: float  # rad
    z_p: float  # mm
    cov: np.ndarray  # (4, 4) covariance, order [t_x, t_y, θ, z_p]
    chi2_curve: np.ndarray  # (N_angles, 2): columns [theta_rad, chi2_min]
    residuals_x: np.ndarray  # (n_inliers,) final x residuals (mm)
    residuals_y: np.ndarray  # (n_inliers,) final y residuals (mm)
    n_inliers: int
    # half_params[0] = [tx, ty, theta, zp] for even-index inliers,
    # half_params[1] = same for odd-index inliers (stratified consistency §7.7)
    half_params: np.ndarray  # (2, 4)


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
    tel_z: np.ndarray,
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
    """
    corr = alignment
    x_arr = x_raw - np.array([corr.planes[k].delta_x for k in range(3)])
    y_arr = y_raw - np.array([corr.planes[k].delta_y for k in range(3)])

    z_arr = alignment.corrected_z_tel(tel_z)
    z_x_arr = z_arr + np.array([corr.planes[k].tilt_y * x_arr[k] for k in range(3)])
    z_y_arr = z_arr + np.array([corr.planes[k].tilt_x * y_arr[k] for k in range(3)])

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
    Four-step probe pose optimizer.  Implements DESIGN.md §7.4.

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
    )


# ── Accumulator ───────────────────────────────────────────────────────────


class PoseFitter:
    """
    Accumulates telescope-probe coincidences and refits the probe pose
    every refit_every new coincidences.  Implements DESIGN_UPDATE.md §6.1.
    """

    MIN_FIT = 30
    REFIT_EVERY = 500

    def __init__(
        self,
        tel_z: np.ndarray,
        alignment: AlignmentCorrection,
        tel_id: int,
        prb_id: int,
        tel_pos_paths: list[Path],
        prb_pos_paths: list[Path],
        refit_every: int = REFIT_EVERY,
        tot_thresh: int = 1,
        tot_weights: bool = False,
        on_decode: Callable[[DecodeReport], None] | None = None,
    ) -> None:
        self.tel_z = tel_z
        self.alignment = alignment
        self.tel_id = tel_id
        self.prb_id = prb_id
        self.tel_pos_paths = tel_pos_paths
        self.prb_pos_paths = prb_pos_paths
        self.refit_every = refit_every
        self.tot_thresh = tot_thresh
        self.tot_weights = tot_weights
        self.on_decode = on_decode
        self._coincs: list[Coincidence] = []
        self._since_last = 0
        self.result: PoseResult | None = None

    def update_alignment(self, correction: AlignmentCorrection) -> None:
        self.alignment = correction

    def add(
        self,
        cluster: list[tuple[int, object, object]],
    ) -> "PoseResult | None":
        """
        Decode positions for the cluster and accumulate the coincidence.
        Returns a new PoseResult when a refit is triggered; otherwise None.
        """
        co = self._decode_cluster(cluster)
        if co is None:
            return None
        self._coincs.append(co)
        self._since_last += 1
        if len(self._coincs) >= self.MIN_FIT and self._since_last >= self.refit_every:
            return self._refit()
        return None

    def flush(self) -> "PoseResult | None":
        """Force a fit on whatever is buffered."""
        if len(self._coincs) < self.MIN_FIT:
            return None
        return self._refit()

    def _decode_cluster(
        self,
        cluster: list,
    ) -> "Coincidence | None":
        """
        Extract telescope and probe hits from a coincidence cluster,
        apply alignment correction, fit a telescope line, apply the
        track quality cut, and return a Coincidence or None.
        """

        def _report(
            reason: str,
            cand_counts: tuple[int, int, int] | None = None,
            chi2: float | None = None,
            prb_quality: str | None = None,
            winning_quality: tuple[str, str, str] | None = None,
            winning_tot: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
            | None = None,
        ) -> None:
            if self.on_decode is not None:
                self.on_decode(
                    DecodeReport(
                        accepted=(reason == "accepted"),
                        reason=reason,
                        cand_counts=cand_counts,
                        chi2=chi2,
                        prb_quality=prb_quality,
                        winning_quality=winning_quality,
                        winning_tot=winning_tot,
                    )
                )

        # A genuine coincidence pairs exactly one telescope track with exactly
        # one hit in *this* probe.  A cluster carrying two or more events from
        # either of those two detectors is ambiguous (two particles inside the
        # window, or a random coincidence) — reject it rather than silently
        # picking one and fabricating a pairing.  Events belonging to *other*
        # probe detectors are ignored here: a single telescope event may
        # legitimately be in coincidence with several distinct probes, each
        # handled by its own PoseFitter.
        tel_refs = [ref for det_id, _ev, ref in cluster if det_id == self.tel_id]
        prb_refs = [ref for det_id, _ev, ref in cluster if det_id == self.prb_id]
        if len(tel_refs) != 1 or len(prb_refs) != 1:
            _report("ambiguous_cluster")
            return None
        tel_ref = tel_refs[0]
        prb_ref = prb_refs[0]

        # Enumerate per-plane candidate positions (golden/cluster axes give
        # one candidate; mirror-fold-ambiguous axes give their full
        # ribbon×fiber cross-product) and search every one-candidate-per-
        # plane triple for the lowest-χ² straight line.  This resolves the
        # mirror-fold ambiguity globally from which combination actually
        # lies on a track, instead of needing two already-clean planes to
        # bootstrap a third (replaces disambiguate_telescope_hits +
        # recover_efficiency_hits in this path; see DESIGN.md §10
        # Deduction #4 and the combinatorial-track-finder plan).
        cands = reconstruct_plane_candidates(
            tel_ref,
            self.tel_pos_paths,
            n_cols=3,
            max_per_plane=16,
            tot_thresh=self.tot_thresh,
        )
        cand_counts = (len(cands[0]), len(cands[1]), len(cands[2]))
        if any(len(c) == 0 for c in cands):
            # A triple needs all 3 planes; single-half dropouts are out of
            # scope for this phase-1 combinatorial search.
            _report("zero_candidate_plane", cand_counts=cand_counts)
            return None
        if all(len(c) > 1 for c in cands):
            # No plane decoded as a single, already-resolved candidate: the
            # mirror-fold/pile-up ambiguity is identical at the bit level
            # for both causes, so a candidate search over all 3 ambiguous
            # planes at once can't tell "one particle, fold-mirrored" from
            # "two particles overlapping in the same window" — it can only
            # find whichever combination happens to minimise χ², which a
            # genuine pile-up can do by coincidence (see
            # TestPerScenarioHandling::test_E2_pileup_same_window_unresolved_rejected).
            # Require at least one already-resolved plane as an anchor,
            # matching the old disambiguate_telescope_hits requirement of
            # ≥2 clean planes to bootstrap a third (here relaxed to ≥1,
            # since the combinatorial search needs only one true reference
            # rather than two independent ones).
            _report("no_anchor_plane", cand_counts=cand_counts)
            return None

        best_chi2 = math.inf
        best_fit = None
        best_triple = None
        for c0, c1, c2 in itertools.product(*cands):
            x_raw = np.array([c0.x_mm, c1.x_mm, c2.x_mm])
            y_raw = np.array([c0.y_mm, c1.y_mm, c2.y_mm])
            sigma_x_arr = np.array([c0.sigma_x, c1.sigma_x, c2.sigma_x])
            sigma_y_arr = np.array([c0.sigma_y, c1.sigma_y, c2.sigma_y])
            fit = _fit_triple(
                x_raw, y_raw, sigma_x_arr, sigma_y_arr, self.alignment, self.tel_z
            )
            if fit[-1] < best_chi2:
                best_chi2 = fit[-1]
                best_fit = fit
                best_triple = (c0, c1, c2)

        if best_triple is not None:
            winning_quality = (
                best_triple[0].quality,
                best_triple[1].quality,
                best_triple[2].quality,
            )
            winning_tot = (
                (best_triple[0].tot_x, best_triple[0].tot_y),
                (best_triple[1].tot_x, best_triple[1].tot_y),
                (best_triple[2].tot_x, best_triple[2].tot_y),
            )
        else:
            winning_quality = None
            winning_tot = None

        if best_fit is None or best_chi2 >= _CHI2_TRACK:
            _report(
                "chi2_track_cut",
                cand_counts=cand_counts,
                chi2=(best_chi2 if best_fit is not None else None),
                winning_quality=winning_quality,
                winning_tot=winning_tot,
            )
            return None
        a_x, b_x, a_y, b_y, cov_x, cov_y, _ = best_fit

        # Decode probe (1 plane)
        prb_hits = decode_position(
            prb_ref,
            self.prb_pos_paths,
            n_cols=1,
            tot_thresh=self.tot_thresh,
            tot_weights=self.tot_weights,
        )
        prb_hit = prb_hits[0]
        if prb_hit.quality not in GOOD_QUALITIES:
            _report(
                "probe_quality",
                cand_counts=cand_counts,
                chi2=best_chi2,
                prb_quality=prb_hit.quality,
                winning_quality=winning_quality,
                winning_tot=winning_tot,
            )
            return None

        _report(
            "accepted",
            cand_counts=cand_counts,
            chi2=best_chi2,
            prb_quality=prb_hit.quality,
            winning_quality=winning_quality,
            winning_tot=winning_tot,
        )
        return Coincidence(
            a_x=a_x,
            b_x=b_x,
            a_y=a_y,
            b_y=b_y,
            cov_ab_x=cov_x,
            cov_ab_y=cov_y,
            u=prb_hit.x_mm,
            v=prb_hit.y_mm,
            sigma_prb_x=prb_hit.sigma_x,
            sigma_prb_y=prb_hit.sigma_y,
        )

    def _refit(self) -> "PoseResult":
        result = fit_probe_pose(
            self._coincs,
            self.alignment.corrected_z_tel(self.tel_z),
            self.alignment,
        )
        self._since_last = 0
        self.result = result
        return result
