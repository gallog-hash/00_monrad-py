"""
Stage 4 — telescope internal alignment.

Public API
----------
PlaneCorrection
    NamedTuple: (delta_x, delta_y, rotation_z)

AlignmentCorrection
    NamedTuple: (planes, needs_correction)

AlignmentAccumulator
    .add(hits)  -> AlignmentCorrection | None
    .flush()    -> AlignmentCorrection

fit_telescope_alignment(hits) -> AlignmentCorrection
    Implements DESIGN.md §6.3 two-diagnostic approach.
"""

from typing import NamedTuple

import numpy as np

from .stage3 import Hit, disambiguate_telescope_hits

_Z_TEL = np.array([0.0, 400.0, 800.0])  # mm
_OFFSET_THRESH = 1.0  # mm  — DESIGN.md §6.4
_ROTATION_THRESH = 1e-3  # rad — DESIGN.md §6.4
_Z_THRESH = 5.0  # mm  — middle-plane Z-offset significance cut

# Other-plane indices used by the two-plane predictor for each plane k.
_OTHERS = [(1, 2), (0, 2), (0, 1)]


class PlaneCorrection(NamedTuple):
    delta_x: float  # mm — mean translational offset in x
    delta_y: float  # mm — mean translational offset in y
    rotation_z: float  # rad — rotation about z (small angle)
    delta_z: float = 0.0  # mm — Z offset (non-zero only for middle plane)


class AlignmentCorrection(NamedTuple):
    planes: list[PlaneCorrection]  # length 3, one per telescope plane
    needs_correction: bool

    @classmethod
    def identity(cls) -> "AlignmentCorrection":
        p = PlaneCorrection(0.0, 0.0, 0.0, 0.0)
        return cls([p, p, p], False)

    def corrected_z_tel(self, base_z: np.ndarray) -> np.ndarray:
        """Return base_z adjusted by each plane's fitted delta_z."""
        return base_z + np.array([p.delta_z for p in self.planes])


def fit_telescope_alignment(
    hits: list[list[Hit]],
    z_tel: np.ndarray | None = None,
) -> AlignmentCorrection:
    """
    Fit telescope internal alignment from a batch of 3-plane hits.

    Implements DESIGN.md §6.3 using two complementary diagnostics:

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
    """
    n = len(hits)
    if n < 3:
        return AlignmentCorrection.identity()

    # Build position arrays — shape (N, 3)
    x = np.array([[h[k].x_mm for k in range(3)] for h in hits], dtype=float)
    y = np.array([[h[k].y_mm for k in range(3)] for h in hits], dtype=float)
    z = z_tel if z_tel is not None else _Z_TEL

    # ── Two-plane prediction ──────────────────────────────────────────
    planes: list[PlaneCorrection] = []
    needs = False

    for k, (j1, j2) in enumerate(_OTHERS):
        # Interpolation / extrapolation fraction along z.
        t = (z[k] - z[j1]) / (z[j2] - z[j1])

        x_pred = x[:, j1] + t * (x[:, j2] - x[:, j1])  # (N,)
        y_pred = y[:, j1] + t * (y[:, j2] - y[:, j1])

        rx = x[:, k] - x_pred  # two-plane residuals
        ry = y[:, k] - y_pred

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

        # ── Z-offset for the middle plane (k=1) only ──────────────
        # If plane k is at z[k] + δz rather than z[k], its two-plane
        # residuals acquire a slope-dependent term: r_k = dx + δz·b,
        # where b ≈ (x_j2 - x_j1) / (z[j2] - z[j1]).  Regress the
        # translation-subtracted residual on the track slope to recover
        # δz.  Outer planes (k=0,2) have their Z offset degenerate with
        # track slope and are left at 0.
        delta_z = 0.0
        if k == 1:
            b_x = (x[:, j2] - x[:, j1]) / (z[j2] - z[j1])
            b_y = (y[:, j2] - y[:, j1]) / (z[j2] - z[j1])
            b_x_c = b_x - float(np.mean(b_x))
            b_y_c = b_y - float(np.mean(b_y))
            var_bx = float(np.var(b_x))
            var_by = float(np.var(b_y))
            dz_x = float(np.mean(rx_c * b_x_c)) / var_bx if var_bx > 0 else 0.0
            dz_y = float(np.mean(ry_c * b_y_c)) / var_by if var_by > 0 else 0.0
            delta_z = (dz_x + dz_y) / 2.0

        planes.append(PlaneCorrection(dx, dy, rotation_z, delta_z))
        if abs(dx) > _OFFSET_THRESH or abs(dy) > _OFFSET_THRESH:
            needs = True
        if abs(rotation_z) > _ROTATION_THRESH:
            needs = True
        if abs(delta_z) > _Z_THRESH:
            needs = True

    return AlignmentCorrection(planes, needs)


class AlignmentAccumulator:
    """
    Collects decoded 3-plane telescope hits into a buffer and fits an
    AlignmentCorrection every flush_every valid events.

    Implements DESIGN_UPDATE.md §5.1.
    """

    def __init__(
        self,
        flush_every: int = 10_000,
        z_tel: np.ndarray | None = None,
    ) -> None:
        self.flush_every = flush_every
        self._z_tel = z_tel
        self._hits: list[list[Hit]] = []
        self.current_correction = AlignmentCorrection.identity()

    def add(self, hits: list[Hit]) -> AlignmentCorrection | None:
        """
        Add one decoded 3-plane hit.

        Events where any plane quality is not 'golden' or 'cluster' are
        silently dropped (after attempting disambiguation).  Returns a new
        AlignmentCorrection when the buffer reaches flush_every events;
        otherwise None.
        """
        if len(hits) != 3:
            return None
        z = self._z_tel if self._z_tel is not None else _Z_TEL
        hits = disambiguate_telescope_hits(hits, z)
        if any(h.quality not in ("golden", "cluster") for h in hits):
            return None
        self._hits.append(hits)
        if len(self._hits) >= self.flush_every:
            return self._fit_and_flush()
        return None

    def flush(self) -> AlignmentCorrection:
        """Force a fit on whatever is buffered and return the correction."""
        if not self._hits:
            return self.current_correction
        return self._fit_and_flush()

    def _fit_and_flush(self) -> AlignmentCorrection:
        correction = fit_telescope_alignment(self._hits, self._z_tel)
        self.current_correction = correction
        self._hits.clear()
        return correction
