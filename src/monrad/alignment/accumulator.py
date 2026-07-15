"""
Stage 4 — telescope internal alignment.

Public API
----------
PlaneCorrection
    NamedTuple: (delta_x, delta_y, rotation_z, delta_z, tilt_x, tilt_y)

AlignmentCorrection
    NamedTuple: (planes, needs_correction)

AlignmentAccumulator
    .add(hits)  -> AlignmentCorrection | None
    .flush()    -> AlignmentCorrection

fit_telescope_alignment(hits) -> AlignmentCorrection
    Implements DESIGN.md §7.3 two-diagnostic approach.
"""

from typing import NamedTuple

import numpy as np

from ..reconstruction import Hit, disambiguate_telescope_hits

_Z_TEL = np.array([0.0, 400.0, 800.0])  # mm
_OFFSET_THRESH = 1.0  # mm  — DESIGN.md §7.4
_ROTATION_THRESH = 1e-3  # rad — DESIGN.md §7.4
_Z_THRESH = 5.0  # mm  — middle-plane Z-offset significance cut
# Middle-plane tilt significance cut.  Set above the ~3 mrad statistical
# floor of the b·coord regression at ~1000 tracks so oscillator/strip noise
# does not routinely trip it, while still catching the small (≲1°) plane
# non-parallelism the telescope mechanics actually permit (DESIGN.md §10).
_TILT_THRESH = 5e-3  # rad

# Other-plane indices used by the two-plane predictor for each plane k.
_OTHERS = [(1, 2), (0, 2), (0, 1)]


class PlaneCorrection(NamedTuple):
    delta_x: float  # mm — mean translational offset in x
    delta_y: float  # mm — mean translational offset in y
    rotation_z: float  # rad — rotation about z (small angle)
    delta_z: float = 0.0  # mm — Z offset (non-zero only for middle plane)
    # Out-of-plane tilts (small angle), non-zero only for the middle plane.
    # tilt_x = rotation about the x-axis → tips the plane in y-z, so it
    # perturbs the *y* measurement; tilt_y = rotation about the y-axis →
    # perturbs the *x* measurement.  These break plane parallelism, the
    # misalignment the telescope mechanics actually permit (DESIGN.md §10).
    tilt_x: float = 0.0  # rad — tilt about x-axis (affects y)
    tilt_y: float = 0.0  # rad — tilt about y-axis (affects x)


class AlignmentCorrection(NamedTuple):
    planes: list[PlaneCorrection]  # length 3, one per telescope plane
    needs_correction: bool

    @classmethod
    def identity(cls) -> "AlignmentCorrection":
        p = PlaneCorrection(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls([p, p, p], False)

    def corrected_z_tel(self, base_z: np.ndarray) -> np.ndarray:
        """Return base_z adjusted by each plane's fitted delta_z."""
        return base_z + np.array([p.delta_z for p in self.planes])


def _fit_dz_and_tilt(
    r_c: np.ndarray,
    b: np.ndarray,
    coord_pred: np.ndarray,
) -> tuple[float, float]:
    """
    Jointly separate a middle-plane Z offset from an out-of-plane tilt.

    Both leave a slope-dependent two-plane residual, but with distinct
    shapes (DESIGN.md §7.3):

        Z offset δz : r ≈ δz · b              (∝ track slope)
        tilt    φ   : r ≈ φ  · b · coord      (∝ slope × lever arm)

    The two regressors ``b`` and ``b·coord`` are strongly correlated for a
    cosmic-ray sample (coord is bounded to one side of the origin), so a
    pair of univariate fits would cross-contaminate.  Solve the 2×2 OLS
    jointly on the centred regressors instead; the partial coefficients are
    δz (on b) and φ (on b·coord).  ``r_c`` is the translation-subtracted
    residual.

    Returns (delta_z, tilt).  Falls back to a univariate δz (tilt = 0) when
    the regressors are degenerate.
    """
    p1 = b - float(np.mean(b))
    p2 = b * coord_pred
    p2 = p2 - float(np.mean(p2))
    s11 = float(p1 @ p1)
    s22 = float(p2 @ p2)
    s12 = float(p1 @ p2)
    det = s11 * s22 - s12 * s12
    if det <= 0.0:
        return (float(p1 @ r_c) / s11 if s11 > 0.0 else 0.0), 0.0
    g1 = float(p1 @ r_c)
    g2 = float(p2 @ r_c)
    delta_z = (s22 * g1 - s12 * g2) / det
    tilt = (s11 * g2 - s12 * g1) / det
    return delta_z, tilt


def fit_telescope_alignment(
    hits: list[list[Hit]],
    z_tel: np.ndarray | None = None,
    disambiguated: list[tuple[bool, bool, bool]] | None = None,
) -> AlignmentCorrection:
    """
    Fit telescope internal alignment from a batch of 3-plane hits.

    Implements DESIGN.md §7.3 using two complementary diagnostics:

    (a) Three-plane straight-line fit — fit x(z) and y(z) through all
        three planes and compute per-plane residuals.

    (b) Two-plane prediction — for each plane k, fit a line through the
        other two planes and predict the hit on plane k.  These residuals
        are the primary output: plane k does not contribute to its own
        predictor, so the result is unbiased.

    Translation (delta_x, delta_y) is taken as the mean two-plane
    residual.  Rotation (rotation_z) is estimated from the OLS slope of
    the centred residual against the predicted perpendicular coordinate.

    Parameters
    ----------
    hits : list of N events; each event is a list of exactly 3 Hits
           (one per telescope plane in z order).
    disambiguated : optional per-event (plane_0, plane_1, plane_2) flags,
        same length as hits, marking which planes were recovered by
        disambiguate_telescope_hits() rather than naturally golden/cluster.
        A disambiguated plane k's position is manufactured from the same
        two-plane (j1, j2) predictor used below to compute plane k's own
        residual, which would otherwise bias that plane's fitted offset
        toward zero.  Such events are excluded from plane k's own
        residual sample (but still usable as j1/j2 predictor input for
        other planes).  Defaults to "nothing was disambiguated".
    """
    n = len(hits)
    if n < 3:
        return AlignmentCorrection.identity()

    # Build position arrays — shape (N, 3)
    x = np.array([[h[k].x_mm for k in range(3)] for h in hits], dtype=float)
    y = np.array([[h[k].y_mm for k in range(3)] for h in hits], dtype=float)
    z = z_tel if z_tel is not None else _Z_TEL
    disamb = (
        np.zeros((n, 3), dtype=bool)
        if disambiguated is None
        else np.array(disambiguated, dtype=bool)
    )

    # ── Two-plane prediction ──────────────────────────────────────────
    # The geometric middle plane is the column with the median z, regardless
    # of the order columns appear in the *.bin file.  Only this plane gets a
    # delta_z/tilt fit; for it, _OTHERS[mid] is exactly the two outer planes.
    mid = int(np.argsort(z)[1])
    planes: list[PlaneCorrection] = []
    needs = False

    for k, (j1, j2) in enumerate(_OTHERS):
        # Exclude events where plane k itself was disambiguated: its
        # position was manufactured from this exact (j1, j2) predictor, so
        # its own residual here would be artificially near zero.
        keep = ~disamb[:, k]
        xk, yk = x[keep, k], y[keep, k]
        xj1, xj2 = x[keep, j1], x[keep, j2]
        yj1, yj2 = y[keep, j1], y[keep, j2]

        if xk.size == 0:
            planes.append(PlaneCorrection(0.0, 0.0, 0.0))
            continue

        # Interpolation / extrapolation fraction along z.
        t = (z[k] - z[j1]) / (z[j2] - z[j1])

        x_pred = xj1 + t * (xj2 - xj1)  # (n_keep,)
        y_pred = yj1 + t * (yj2 - yj1)

        rx = xk - x_pred  # two-plane residuals
        ry = yk - y_pred

        # Translation: mean residual.
        dx = float(np.mean(rx))
        dy = float(np.mean(ry))

        # Rotation about z: r_x = Δx − α·y_pred  →  slope of rx_c on y_pred
        #                   r_y = α·x_pred + Δy   →  slope of ry_c on x_pred
        rx_c = rx - dx
        ry_c = ry - dy
        var_y = float(np.var(y_pred))
        var_x = float(np.var(x_pred))

        # OLS slope: cov(r, p) / var(p) using population estimators.
        alpha_x = (
            -float(np.mean(rx_c * (y_pred - np.mean(y_pred)))) / var_y
            if var_y > 0
            else 0.0
        )
        alpha_y = (
            float(np.mean(ry_c * (x_pred - np.mean(x_pred)))) / var_x
            if var_x > 0
            else 0.0
        )
        rotation_z = (alpha_x + alpha_y) / 2.0

        # ── Z-offset and tilt for the middle plane (k == mid) only ─────
        # Both a Z offset and an out-of-plane tilt give plane k a
        # slope-dependent two-plane residual.  Their shapes differ:
        #   Z offset δz : r ≈ δz·b            (b = track slope)
        #   tilt    φ   : r ≈ φ·b·coord       (slope × lever arm)
        # so a joint regression of the translation-subtracted residual on
        # (b, b·coord) recovers both without cross-contamination (see
        # _fit_dz_and_tilt).  tilt_y (about the y-axis) perturbs x, tilt_x
        # (about the x-axis) perturbs y.  Outer planes (k=0,2) have these
        # degrees of freedom degenerate with track slope and are left at 0.
        delta_z = 0.0
        tilt_x = 0.0
        tilt_y = 0.0
        if k == mid:
            b_x = (xj2 - xj1) / (z[j2] - z[j1])
            b_y = (yj2 - yj1) / (z[j2] - z[j1])
            dz_x, tilt_y = _fit_dz_and_tilt(rx_c, b_x, x_pred)
            dz_y, tilt_x = _fit_dz_and_tilt(ry_c, b_y, y_pred)
            delta_z = (dz_x + dz_y) / 2.0

        planes.append(PlaneCorrection(dx, dy, rotation_z, delta_z, tilt_x, tilt_y))
        if abs(dx) > _OFFSET_THRESH or abs(dy) > _OFFSET_THRESH:
            needs = True
        if abs(rotation_z) > _ROTATION_THRESH:
            needs = True
        if abs(delta_z) > _Z_THRESH:
            needs = True
        if abs(tilt_x) > _TILT_THRESH or abs(tilt_y) > _TILT_THRESH:
            needs = True

    return AlignmentCorrection(planes, needs)


class AlignmentAccumulator:
    """
    Collects decoded 3-plane telescope hits into a buffer and fits an
    AlignmentCorrection every flush_every valid events.

    Implements DESIGN.md §7.5.
    """

    def __init__(
        self,
        flush_every: int = 10_000,
        z_tel: np.ndarray | None = None,
    ) -> None:
        self.flush_every = flush_every
        self._z_tel = z_tel
        self._hits: list[list[Hit]] = []
        self._disambiguated: list[tuple[bool, bool, bool]] = []
        self.current_correction = AlignmentCorrection.identity()

    @property
    def n_buffered(self) -> int:
        """Number of valid events currently buffered, awaiting the next fit.

        Read this just before :meth:`flush` to report how many events a fit
        was computed from (``flush`` clears the buffer).  With a very large
        ``flush_every`` -- so ``add`` never mid-flushes -- this is the full
        event count going into a single whole-acquisition fit.
        """
        return len(self._hits)

    def add(self, hits: list[Hit]) -> AlignmentCorrection | None:
        """
        Add one decoded 3-plane hit.

        Events where any plane quality is not 'golden' or 'cluster' are
        silently dropped (after attempting disambiguation).  Which planes
        were recovered by disambiguation (vs. naturally golden/cluster) is
        recorded alongside the event, so fit_telescope_alignment can
        exclude a disambiguated plane from its own residual sample (see
        that function's docstring).  Returns a new AlignmentCorrection when
        the buffer reaches flush_every events; otherwise None.
        """
        if len(hits) != 3:
            return None
        z = self._z_tel if self._z_tel is not None else _Z_TEL
        raw_hits = hits
        hits = disambiguate_telescope_hits(hits, z)
        if any(h.quality not in ("golden", "cluster") for h in hits):
            return None
        self._hits.append(hits)
        self._disambiguated.append(tuple(hits[k] is not raw_hits[k] for k in range(3)))
        if len(self._hits) >= self.flush_every:
            return self._fit_and_flush()
        return None

    def flush(self) -> AlignmentCorrection:
        """Force a fit on whatever is buffered and return the correction."""
        if not self._hits:
            return self.current_correction
        return self._fit_and_flush()

    def _fit_and_flush(self) -> AlignmentCorrection:
        correction = fit_telescope_alignment(
            self._hits, self._z_tel, self._disambiguated
        )
        self.current_correction = correction
        self._hits.clear()
        self._disambiguated.clear()
        return correction
