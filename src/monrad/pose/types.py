"""
Stage 5 (part) — data structures for the probe pose fit.

Coincidence   per-coincidence telescope-line + probe-hit bundle fed to the
              optimizer.
PoseResult    full optimizer output (fitted pose, covariance, diagnostics).
DecodeReport / GATE_ORDER   instrumentation of the
              PoseFitter._decode_cluster funnel.
TelescopeTrackResult   outcome of the telescope-only half of that funnel,
              shareable across PoseFitters watching the same cluster.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


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
    # Per-plane quality ("golden"/"cluster") of the winning telescope triple,
    # taken from each winning candidate's own label (DESIGN.md §8.2).  The
    # default keeps positional construction in tests working unchanged.
    tel_quality: tuple[str, str, str] = ("golden", "golden", "golden")
    # Telescope event time (integer ns) of this coincidence, set by
    # PoseFitter._decode_cluster from the telescope TimedEvent.t_ns.  Lets
    # downstream monitoring bucket coincidences into time windows without
    # re-reading the stage-1 stream.  The default keeps positional
    # Coincidence(...) construction in tests working unchanged.
    t_ns: int = 0
    # Label of the AlignmentCorrection active when this coincidence was
    # decoded (e.g. "20210723_060000" for a time-varying AlignmentSchedule
    # window, or the static alignment source's name).  Set by the monitor
    # drivers (monrad.monitor.io.stream_coincidences /
    # monrad.monitor.multiprobe), not by PoseFitter itself, since only the
    # caller knows the schedule/label -- PoseFitter only holds the
    # AlignmentCorrection object.  "" when no label is available.  The
    # default keeps positional Coincidence(...) construction in tests
    # working unchanged.
    alignment_label: str = ""
    # Probe-minus-telescope event time (integer ns): prb_ev.t_ns - tel_ev.t_ns,
    # set by PoseFitter._finish_decode.  Negative means the probe fired first,
    # which is the expected sign when the probe sits above the telescope.
    #
    # Instrumentation only -- nothing in the pipeline gates on it.  The true
    # time of flight is ~3.3 ns/m at beta~1, far below the measurement floor
    # (an 80 ns acquisition window per event, DESIGN.md SS3.3; a 10 ns clock
    # tick, timing.F0_DEFAULT; and 20-50 ns of GPS accEst per detector,
    # DESIGN.md SS5), so the *per-event* sign is near-random and must not be
    # cut on.  What is measurable is the distribution: its mean is the constant
    # inter-detector offset (cable/trigger latency + per-receiver GPS bias) and
    # its width is the combined timing resolution, which is what DESIGN.md SS5
    # wants before tightening window_ns.  The default keeps positional
    # Coincidence(...) construction in tests working unchanged.
    dt_ns: int = 0


class DecodeReport(NamedTuple):
    """
    Outcome of one PoseFitter._decode_cluster call, for instrumentation
    (e.g. scripts/run_pipeline.py's Stage 3 table, track_coincidence_loss.py)
    so diagnostics read the gates _decode_cluster actually applied instead
    of re-deriving their own copy of its logic.

    reason is one of: "ambiguous_cluster", "zero_candidate_plane",
    "no_anchor_plane", "chi2_track_cut", "probe_quality", "accepted".
    cand_counts/chi2/prb_quality are None when the cluster was rejected before
    that quantity was computed.
    """

    accepted: bool
    reason: str
    cand_counts: tuple[int, int, int] | None
    chi2: float | None
    prb_quality: str | None
    # Per-plane quality ("golden"/"cluster") of the winning telescope triple
    # (see Coincidence).  None until the cluster reaches the
    # telescope-classification step (i.e. on the "probe_quality" and "accepted"
    # paths).
    tel_quality: tuple[str, str, str] | None = None
    # Probe-minus-telescope event time in ns (see Coincidence.dt_ns).  Set on
    # every path where the cluster held exactly one telescope *and* exactly one
    # of this probe's events -- i.e. all of "zero_candidate_plane",
    # "no_anchor_plane", "chi2_track_cut", "probe_quality" and "accepted", so
    # the timing distribution can be measured over the full one-tel/one-probe
    # population rather than only over pose-fit survivors.  None only for
    # "ambiguous_cluster", where no unique pairing exists to difference.
    dt_ns: int | None = None


# The rejection gates _decode_cluster applies, in the order it checks them
# (the "accepted" terminal is the success path, not a gate).  This is the
# single source of truth for the funnel ordering: diagnostics import it
# instead of hard-coding their own copy, so they can't drift from
# _decode_cluster.  A DecodeReport.reason outside this tuple and not
# "accepted" should be caught by callers as a catch-all, not silently lost.
GATE_ORDER = (
    "ambiguous_cluster",
    "zero_candidate_plane",
    "no_anchor_plane",
    "chi2_track_cut",
    "probe_quality",
)


class TelescopeTrackResult(NamedTuple):
    """
    Outcome of PoseFitter._decode_telescope_track: the combinatorial
    telescope-track search for one coincidence cluster, computed from the
    cluster's telescope entry alone — independent of which probe (if any)
    is asking.

    Shareable across every PoseFitter watching the same cluster, as long as
    they agree on tel_id/tel_z/alignment/tot_thresh/tot_weights/
    min_anchor_planes/tel_pos_paths (finding 9 — see
    ``monrad.monitor.multiprobe``).

    accepted=False marks a telescope-side rejection; reason is one of the
    telescope-side entries of GATE_ORDER ("ambiguous_cluster" for >1 or 0
    telescope entries in the cluster, "zero_candidate_plane",
    "no_anchor_plane", "chi2_track_cut"). The line-fit / covariance /
    tel_quality fields are only meaningful when accepted=True.

    t_ns is the exception: it is set whenever the cluster held exactly one
    telescope event, rejection paths included, so DecodeReport.dt_ns can be
    reported for clusters the telescope side rejected.  It stays 0 only for
    the "ambiguous_cluster" return, where there is no unique telescope event.
    """

    accepted: bool
    reason: str
    cand_counts: tuple[int, int, int] | None = None
    chi2: float | None = None
    a_x: float = 0.0
    b_x: float = 0.0
    a_y: float = 0.0
    b_y: float = 0.0
    cov_ab_x: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cov_ab_y: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tel_quality: tuple[str, str, str] | None = None
    t_ns: int = 0


# ── Result bundle ─────────────────────────────────────────────────────────


@dataclass
class PoseResult:
    """
    Full output of fit_probe_pose().  Implements DESIGN.md §8.7.

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
    # half_params[1] = same for odd-index inliers (stratified consistency §8.7)
    half_params: np.ndarray  # (2, 4)
    inliers: list[Coincidence]  # the n_inliers Coincidences kept after the
    # Mahalanobis cut (used for the final refit) — exposed for diagnostics
    # such as 3D track plots.
    outliers: list[Coincidence] = field(default_factory=list)
    # the Coincidences rejected by the Mahalanobis cut (d_i > 4) — exposed so
    # diagnostics can render the LM-polish-removed tracks distinctly from the
    # inliers (DESIGN.md §8.7).  Empty when the cut was bypassed (the
    # len(inliers) < 3 fallback keeps all coincidences).
