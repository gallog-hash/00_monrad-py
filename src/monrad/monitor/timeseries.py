"""Probe pose monitoring over an acquisition (monitoring Step 2).

Console script ``monrad-monitor``.  Streams an acquisition and emits one
probe pose per batch, tracking the probe's position over time with
per-batch uncertainty.

Two batching modes, both driven by ``min_fit`` — the minimum number of
coincidences *surviving the configured geometric gates* fed to
:func:`~monrad.pose.fit_probe_pose`:

* **Count-based (default, ``window_s`` omitted).**  Buffer decoded
  :class:`~monrad.pose.Coincidence` objects until at least ``min_fit`` are
  collected, fit once, emit a :class:`WindowResult`, then reset.  The batch
  timestamps come from the first and last coincidence in the batch.
* **Hybrid (``window_s`` given).**  Each window grows until it spans at least
  ``window_s`` seconds *and* holds at least ``min_fit`` coincidences, whichever
  bound takes longer.  Batch timestamps are the first and last coincidence in
  the window.

In both modes, once the raw batch reaches ``min_fit`` (and, in hybrid mode,
spans ``window_s``), the configured pre-fit geometric gates
(``max_rigidity_resid_mm``, ``max_off_probe_mm``) run against it.  A gate only
ever removes coincidences, so a raw batch sized to exactly ``min_fit`` can
drop below the floor after gating; when that happens the batch keeps growing
— pulling in more raw coincidences and re-gating — until the *survivor* count
clears ``min_fit``.  If the raw batch grows past ``RAW_CAP_MULTIPLIER *
min_fit`` coincidences without enough survivors, the window is abandoned as
contaminated and dropped; the trailing remainder at end-of-stream is dropped
the same way.

Once ``min_fit`` gate-survivors are reached, ``fit_probe_pose`` runs and the
post-fit continuity gate (``max_pose_jump_mm``/``max_pose_jump_deg``) checks
the *fitted pose* against the previous accepted window's pose — a
contaminated raw batch can be internally consistent enough to fool
``fit_probe_pose``'s own Mahalanobis cut, converging on a spurious pose
instead of the genuine one.  Unlike the pre-fit gates, a rejection here is
terminal: the whole batch is dropped immediately (logged the same way as the
other drop paths) and the next window starts fresh from the following
coincidence, rather than growing the same batch and retrying.

Only the open batch is ever buffered in RAM.

In-plane uncertainties ``sigma_tx`` / ``sigma_ty`` are reported at the probe
**centre** (not the fit-parameter corner), consistent with the resolution
study (``monrad-resolution``).  The propagation uses
:func:`~monrad.monitor.io.centre_cov_2x2`.
"""

import argparse
import csv
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..alignment import AlignmentCorrection
from ..pose import (
    Coincidence,
    PoseFitter,
    PoseResult,
    _MIN_COINCS,
    filter_off_probe,
    filter_rigidity,
    fit_probe_pose,
)
from .io import (
    MacroArgumentParser,
    centre_cov_2x2,
    fit_alignment,
    load_detector,
    stream_coincidences,
    validate_probe_footprint,
)

logger = logging.getLogger(__name__)

# Default minimum decoded coincidences fed to a pose fit — the same floor the
# streaming PoseFitter applies, so monitored fits never diverge from it.  Used
# both as the count-based batch size and as the time-window floor.
MIN_FIT = PoseFitter.MIN_FIT

# A raw batch that still hasn't cleared min_fit survivors after growing to
# this many multiples of min_fit is treated as contaminated and dropped,
# rather than growing indefinitely chasing a floor a bad stretch will never
# reach.
RAW_CAP_MULTIPLIER = 5

# During cold start (no prev_pose yet), the rigidity gate's z_ref anchor is
# bootstrapped from a full fit_probe_pose call on the growing raw batch (see
# _run_gates). That's a throwaway anchor, not the committed pose, so refitting
# it on every single newly-appended coincidence is wasted work — it's instead
# cached and only refreshed once the raw batch has grown by this many
# coincidences since the last recompute.
COLD_START_REFIT_STRIDE = 10


def _window_resid_rms(pose: PoseResult) -> float:
    """Combined absolute-mm residual RMS over *all* coincidences fed to the fit.

    The honest window-quality signal.  ``pose.residuals_x/y`` cover only the
    inliers kept after the fit's Mahalanobis ``d>4`` cut, so a window
    contaminated by wide-angle "wild" telescope tracks looks clean there — the
    cut rejects the wild tracks, and the surviving good core has a normal RMS.
    The contamination shows up only when the *rejected* tracks are counted back
    in: their raw residual against the fitted pose is large.  This recomputes
    the residual of every inlier **and** outlier against the fitted pose
    (matching :func:`~monrad.pose.optimize._weighted_residuals`, without the
    per-point normalisation), so a burst of wild tracks lifts the RMS even
    though it evaded the outlier cut in the reported inlier residuals.
    """
    coincs = pose.inliers + pose.outliers
    if not coincs:
        return math.inf
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    sq = 0.0
    for co in coincs:
        rx = (co.a_x + co.b_x * pose.z_p) - (pose.t_x + co.u * c - co.v * s)
        ry = (co.a_y + co.b_y * pose.z_p) - (pose.t_y + co.u * s + co.v * c)
        sq += rx * rx + ry * ry
    return math.sqrt(sq / len(coincs))


def _pose_jump(pose: PoseResult, ref: PoseResult) -> tuple[float, float]:
    """How far *pose* has drifted from *ref* -- the post-fit continuity gate's signal.

    Uses the raw fit corner (t_x, t_y, z_p), not the probe centre.  For a
    rigid-body pose, a rotation can partly cancel a corner translation when
    viewed at the centre (half*(cos-sin, sin+cos) folds theta in), so a
    genuine large (theta, z_p) swing can leave the *centre* looking almost
    stationary -- exactly what happened in the testLab burst-tail window this
    gate targets (docs/handoffs/
    2026-07-09-post-fix-monitor-run-transition-window.md): the corner moved
    ~65 mm and theta by ~10.5 deg, but the centre moved only ~14 mm.  Checking
    the corner + z_p and theta separately catches both independently.

    Returns (pos_jump_mm, theta_jump_deg).
    """
    pos_jump = math.sqrt(
        (pose.t_x - ref.t_x) ** 2
        + (pose.t_y - ref.t_y) ** 2
        + (pose.z_p - ref.z_p) ** 2
    )
    theta_jump = math.degrees(abs(pose.theta - ref.theta)) % 360.0
    theta_jump = min(theta_jump, 360.0 - theta_jump)
    return pos_jump, theta_jump


@dataclass
class WindowResult:
    """Pose fit for one time window."""

    utc_start: datetime
    utc_end: datetime
    n_inliers: int
    t_x: float
    sigma_tx: float  # probe-centre σ_x (mm)
    t_y: float
    sigma_ty: float  # probe-centre σ_y (mm)
    z_p: float
    sigma_zp: float
    theta: float
    sigma_theta: float
    resid_rms: float  # combined absolute-mm residual RMS over all coincidences
    # fed to the fit (inliers + Mahalanobis-cut outliers); see _window_resid_rms


def _utc(t_ns: float) -> datetime:
    return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc)


class _WindowAccumulator:
    """Windowing/gating state machine for one probe's pose timeseries.

    Buffers a growing raw batch of decoded :class:`~monrad.pose.Coincidence`
    objects, applies the configured pre-fit geometric gates, fits a window
    once enough survivors accumulate, checks the post-fit continuity gate,
    and records a :class:`WindowResult` -- exactly the state machine
    documented in the module docstring and :func:`monitor_probe`'s
    docstring.  Multi-probe monitoring drives one independent instance per
    probe off a shared cluster stream; :func:`monitor_probe` itself is a
    thin single-instance wrapper so single-probe behavior is unchanged, not
    duplicated.

    ``label`` prefixes this accumulator's log messages so interleaved
    multi-probe logs are attributable to the probe they came from.
    """

    def __init__(
        self,
        *,
        z_corr: np.ndarray,
        alignment: AlignmentCorrection,
        n_probe_ch: int,
        fibers_per_ribbon: int = 10,
        window_ns: int | None = None,
        min_fit: int = MIN_FIT,
        max_rigidity_resid_mm: float | None = None,
        max_off_probe_mm: float | None = None,
        max_pose_jump_mm: float | None = None,
        max_pose_jump_deg: float | None = None,
        label: str = "",
    ) -> None:
        self.z_corr = z_corr
        self.alignment = alignment
        self.n_probe_ch = n_probe_ch
        self.fibers_per_ribbon = fibers_per_ribbon
        self.probe_size_mm = n_probe_ch * 10.0
        self.window_ns = window_ns
        self.min_fit = min_fit
        self.max_rigidity_resid_mm = max_rigidity_resid_mm
        self.max_off_probe_mm = max_off_probe_mm
        self.max_pose_jump_mm = max_pose_jump_mm
        self.max_pose_jump_deg = max_pose_jump_deg
        self.label = label
        self._prefix = f"{label}: " if label else ""
        self.raw_cap = min_fit * RAW_CAP_MULTIPLIER

        self.results: list[WindowResult] = []
        self.prev_pose: PoseResult | None = None
        self.cold_start_z_ref: float | None = None
        self.cold_start_n = 0
        self.batch: list[Coincidence] = []
        self.win_start_ns: int | None = None
        self._max_u_seen = 0.0
        self._max_v_seen = 0.0
        self._footprint_warned = False

    def _run_gates(
        self, coincs: list[Coincidence], utc_start: datetime, utc_end: datetime
    ) -> list[Coincidence]:
        """Apply the configured geometric gates once; return the survivors.

        Read-only w.r.t. ``prev_pose`` — callers decide whether/when to
        actually commit a fit from the result, so this can be called
        repeatedly on a growing raw batch without side effects.
        """
        working = coincs
        if self.max_rigidity_resid_mm is not None:
            if self.prev_pose is not None:
                z_ref = self.prev_pose.z_p
            else:
                # Cold start: no previous accepted pose to anchor z_ref. See
                # monitor_probe's max_rigidity_resid_mm docstring for why
                # mean(z_corr) is unusable and the bootstrap fit is cached
                # rather than recomputed on every growth step.
                if (
                    self.cold_start_z_ref is None
                    or len(coincs) - self.cold_start_n >= COLD_START_REFIT_STRIDE
                ):
                    self.cold_start_z_ref = fit_probe_pose(
                        working, self.z_corr, self.alignment
                    ).z_p
                    self.cold_start_n = len(coincs)
                z_ref = self.cold_start_z_ref
            working, dropped_rigid = filter_rigidity(
                working, z_ref, self.max_rigidity_resid_mm
            )
            if dropped_rigid:
                logger.info(
                    "%sWindow %s–%s: rigidity gate dropped %d/%d coincidence(s) "
                    "(z_ref=%.1f mm).",
                    self._prefix,
                    utc_start.isoformat(),
                    utc_end.isoformat(),
                    len(dropped_rigid),
                    len(coincs),
                    z_ref,
                )

        if self.max_off_probe_mm is not None and self.prev_pose is not None:
            n_before = len(working)
            working, dropped_fp = filter_off_probe(
                working, self.prev_pose, self.probe_size_mm, self.max_off_probe_mm
            )
            if dropped_fp:
                logger.info(
                    "%sWindow %s–%s: footprint gate dropped %d/%d coincidence(s).",
                    self._prefix,
                    utc_start.isoformat(),
                    utc_end.isoformat(),
                    len(dropped_fp),
                    n_before,
                )
        return working

    def _record(
        self,
        pose: PoseResult,
        utc_start: datetime,
        utc_end: datetime,
    ) -> WindowResult:
        # Absolute-mm residual RMS over ALL coincidences fed to the fit — the
        # honest window-quality signal.  The inlier-only residuals the fit
        # reports look clean even for a contaminated window, because the
        # Mahalanobis cut rejects the wild tracks; counting the rejected tracks
        # back in is what exposes the contamination (see _window_resid_rms).
        rms = _window_resid_rms(pose)

        cov_c = centre_cov_2x2(pose.cov, pose.theta, self.n_probe_ch)
        result = WindowResult(
            utc_start=utc_start,
            utc_end=utc_end,
            n_inliers=pose.n_inliers,
            t_x=pose.t_x,
            sigma_tx=math.sqrt(abs(cov_c[0, 0])),
            t_y=pose.t_y,
            sigma_ty=math.sqrt(abs(cov_c[1, 1])),
            z_p=pose.z_p,
            sigma_zp=math.sqrt(abs(pose.cov[3, 3])),
            theta=pose.theta,
            sigma_theta=math.sqrt(abs(pose.cov[2, 2])),
            resid_rms=rms,
        )
        self.results.append(result)
        self.prev_pose = pose
        return result

    def _reset_batch(self) -> None:
        self.batch = []
        self.win_start_ns = None
        self.cold_start_z_ref = None
        self.cold_start_n = 0

    def push(self, co: Coincidence) -> WindowResult | None:
        """Append one decoded coincidence; return a new WindowResult if a window closed.

        A window "closes" (the raw batch resets) whenever it is committed,
        dropped via the post-fit continuity gate, or abandoned at the raw
        cap -- in the latter two cases this returns ``None`` even though the
        batch reset, since no result was recorded.  While the raw batch is
        still short of survivors (and hasn't hit the raw cap), it keeps
        growing and this returns ``None`` without resetting anything.
        """
        if self.win_start_ns is None:
            self.win_start_ns = co.t_ns
        self.batch.append(co)

        if co.u > self._max_u_seen:
            self._max_u_seen = co.u
        if co.v > self._max_v_seen:
            self._max_v_seen = co.v
        if not self._footprint_warned and (
            co.u > self.probe_size_mm or co.v > self.probe_size_mm
        ):
            logger.warning(
                "%sDecoded probe hit at (u=%.1f, v=%.1f) mm exceeds the "
                "configured footprint [0, %.1f] mm (--n-probe-ch=%d, "
                "--fibers-per-ribbon=%d); either the channel count is too "
                "small for this probe, or --fibers-per-ribbon is wrong and "
                "channels are aliasing, which would bias the off-probe gate "
                "and the centre-covariance propagation either way. Further "
                "out-of-bounds hits this run are summarized at "
                "end-of-stream, not logged individually.",
                self._prefix,
                co.u,
                co.v,
                self.probe_size_mm,
                self.n_probe_ch,
                self.fibers_per_ribbon,
            )
            self._footprint_warned = True

        spanned = self.window_ns is None or (
            co.t_ns - self.win_start_ns >= self.window_ns
        )
        if not spanned or len(self.batch) < self.min_fit:
            return None

        utc_start, utc_end = _utc(self.win_start_ns), _utc(co.t_ns)
        working = self._run_gates(self.batch, utc_start, utc_end)

        if len(working) < self.min_fit:
            if len(self.batch) >= self.raw_cap:
                logger.warning(
                    "%sDropping window %s–%s: only %d/%d raw coincidence(s) "
                    "survive the geometric gates after reaching the %dx raw "
                    "cap (< min_fit=%d).",
                    self._prefix,
                    utc_start.isoformat(),
                    utc_end.isoformat(),
                    len(working),
                    len(self.batch),
                    RAW_CAP_MULTIPLIER,
                    self.min_fit,
                )
                self._reset_batch()
            # else: not enough survivors yet — keep growing the raw batch and
            # re-gate on the next coincidence.
            return None

        pose = fit_probe_pose(working, self.z_corr, self.alignment)
        if pose.outliers:
            logger.info(
                "%sWindow %s–%s: fit accepted %d/%d gate-survivor(s), rejected "
                "%d via Mahalanobis cut.",
                self._prefix,
                utc_start.isoformat(),
                utc_end.isoformat(),
                pose.n_inliers,
                len(working),
                len(pose.outliers),
            )

        continuity_ok = True
        if (
            self.max_pose_jump_mm is not None or self.max_pose_jump_deg is not None
        ) and self.prev_pose is not None:
            pos_jump, theta_jump = _pose_jump(pose, self.prev_pose)
            continuity_ok = (
                self.max_pose_jump_mm is None or pos_jump <= self.max_pose_jump_mm
            ) and (
                self.max_pose_jump_deg is None or theta_jump <= self.max_pose_jump_deg
            )
            if not continuity_ok:
                logger.warning(
                    "%sDropping window %s–%s: post-fit continuity gate rejected "
                    "the fitted pose (Δpos=%.1f mm, Δθ=%.2f° vs prev "
                    "z_p=%.1f mm, θ=%.2f°); %d gate-survivor(s) discarded.",
                    self._prefix,
                    utc_start.isoformat(),
                    utc_end.isoformat(),
                    pos_jump,
                    theta_jump,
                    self.prev_pose.z_p,
                    math.degrees(self.prev_pose.theta),
                    len(working),
                )

        result = self._record(pose, utc_start, utc_end) if continuity_ok else None
        self._reset_batch()
        return result

    def finalize(self) -> None:
        """Log the trailing raw batch, if any, as dropped at end-of-stream."""
        if self.batch:
            assert self.win_start_ns is not None  # set whenever batch is non-empty
            utc_start, utc_end = _utc(self.win_start_ns), _utc(self.batch[-1].t_ns)
            logger.warning(
                "%sDropping trailing window %s–%s: stream ended with only %d "
                "raw coincidence(s), never reaching min_fit=%d survivors.",
                self._prefix,
                utc_start.isoformat(),
                utc_end.isoformat(),
                len(self.batch),
                self.min_fit,
            )
        if self._footprint_warned:
            max_seen = max(self._max_u_seen, self._max_v_seen)
            logger.warning(
                "%s--n-probe-ch=%d (footprint %.0f mm) was exceeded by decoded "
                "probe hits during this run (max observed %.1f mm). This "
                "means either the channel count is too small (consider "
                "--n-probe-ch %d or higher) or --fibers-per-ribbon=%d is "
                "wrong for this probe (channel aliasing produces the same "
                "symptom) -- check the probe's actual wiring before raising "
                "--n-probe-ch.",
                self._prefix,
                self.n_probe_ch,
                self.probe_size_mm,
                max_seen,
                math.ceil(max_seen / 10.0),
                self.fibers_per_ribbon,
            )


def monitor_probe(
    tel_dir: Path,
    prb_dir: Path,
    *,
    window_s: float | None = None,
    z_tel: np.ndarray,
    n_probe_ch: int = 30,
    out_dir: Path | None = None,
    min_fit: int = MIN_FIT,
    min_anchor_planes: int = 1,
    max_rigidity_resid_mm: float | None = None,
    max_off_probe_mm: float | None = None,
    max_pose_jump_mm: float | None = None,
    max_pose_jump_deg: float | None = None,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    fibers_per_ribbon: int = 10,
    make_plots: bool = True,
) -> list[WindowResult]:
    """Stream an acquisition and fit the probe pose in successive batches.

    Emits one :class:`WindowResult` per batch of at least ``min_fit``
    gate-surviving coincidences.  Only the open batch is buffered — RAM usage
    is bounded to roughly ``RAW_CAP_MULTIPLIER * min_fit`` coincidences.

    Parameters
    ----------
    tel_dir, prb_dir:
        Acquisition directories for the telescope and probe.
    window_s:
        Window duration in seconds.  When ``None`` (the default), batches are
        count-based: the timestamps come from the first and last coincidence
        in the batch.  When given, the mode is hybrid: each window grows
        until it spans at least ``window_s`` seconds *and* holds at least
        ``min_fit`` gate-surviving coincidences, whichever bound takes
        longer.  In both modes a raw batch that hasn't yet cleared
        ``min_fit`` survivors keeps growing past its nominal size — see
        ``min_fit`` below.
    z_tel:
        Telescope plane z-positions (mm).
    n_probe_ch:
        Probe channel count; used to propagate the pose covariance to the
        physical probe centre (see :func:`~monrad.monitor.io.centre_cov_2x2`).
    out_dir:
        If given, write ``pose_timeseries.csv`` and (when ``make_plots``)
        ``pose_timeseries.png`` here.
    min_fit:
        Minimum coincidences *surviving the configured geometric gates*
        required to fit a window.  The raw batch is first filled to
        ``min_fit`` (count-based mode) or to ``window_s`` and ``min_fit``
        (hybrid mode), then gated.  A gate only ever removes coincidences, so
        if survivors fall short, the raw batch keeps growing — pulling in one
        more raw coincidence and re-gating — until survivors clear
        ``min_fit`` or the raw batch reaches ``RAW_CAP_MULTIPLIER * min_fit``,
        at which point the window is dropped as contaminated (see
        :data:`RAW_CAP_MULTIPLIER`).  Without a gate, gating is a no-op and
        this reduces to the original fixed-size batching.  Defaults to
        :data:`MIN_FIT`.
    min_anchor_planes:
        Minimum telescope planes with an unambiguous (single-candidate) hit for
        a cluster to survive the ``no_anchor_plane`` gate.  ``0`` disables the
        gate.  Defaults to ``1`` (matches :class:`~monrad.pose.PoseFitter`).
    max_rigidity_resid_mm:
        Pre-fit geometric gate (see :func:`~monrad.pose.filter_rigidity`),
        applied to a window's coincidences *before* ``fit_probe_pose``.  Probe
        hits and track projections are the same physical points related by a
        rigid transform, which preserves pairwise distances; a cross-particle
        telescope track that time-matched an unrelated probe hit violates this
        invariant and is dropped.  Pose-free — needs only ``z_ref`` (the
        previous accepted window's ``z_p``; for the first window, an ungated
        ``fit_probe_pose`` bootstrap on the window's own coincidences —
        ``mean(z_corr)`` is NOT used, since the rigidity residual scales with
        ``|z_ref - z_p|`` and the telescope-stack mean can sit over 1000 mm
        from a probe far off-stack, e.g. the testLab setup).  Can drop a
        window down to 0 survivors when none of its coincidences are genuine,
        in which case the raw batch keeps growing until it hits
        ``RAW_CAP_MULTIPLIER * min_fit`` and is dropped — this is intentional;
        the gate does not preserve a floor.  During cold start (``prev_pose``
        still ``None``), the bootstrap fit is a throwaway anchor, not the
        committed pose, so it's cached across growth steps and only
        recomputed every :data:`COLD_START_REFIT_STRIDE` raw coincidences
        rather than on every single one.  ``None`` (the default) disables the
        gate.
    max_off_probe_mm:
        Pre-fit geometric gate (see :func:`~monrad.pose.filter_off_probe`),
        applied after the rigidity gate.  Extrapolates each track to the
        *previous accepted window's* pose and drops it if it lands more than
        this far outside the probe's physical footprint
        (``n_probe_ch * 10`` mm on a side).  Skipped on the first window (no
        reference pose yet).  Like the rigidity gate, can drop a window to 0
        survivors.  ``None`` (the default) disables the gate.
    max_pose_jump_mm, max_pose_jump_deg:
        Post-fit continuity gate, checked *after* ``fit_probe_pose`` runs on
        a candidate batch of gate-survivors, unlike the two pre-fit gates
        above which filter individual coincidences before the fit ever sees
        them.  A contaminated batch can be internally consistent enough
        (against itself, via rigidity, and against the previous pose's
        footprint) that ``fit_probe_pose``'s own Mahalanobis cut locks onto a
        spurious cluster instead of the genuine one -- see
        docs/handoffs/2026-07-09-post-fix-monitor-run-transition-window.md.
        Reject the candidate pose if it has moved more than ``max_pose_jump_mm``
        (Euclidean distance of the fit corner ``(t_x, t_y, z_p)``) or rotated
        more than ``max_pose_jump_deg`` from the previous accepted window's
        pose -- checked independently (see :func:`_pose_jump`) because a
        rotation can partly cancel a corner translation when viewed at the
        probe centre.  Unlike the pre-fit gates, a rejection here is
        terminal: the whole raw batch is discarded immediately (logged the
        same way as the other drop paths) and the next window starts fresh
        from the following coincidence, rather than growing the same batch
        and retrying -- diluting a contaminated batch by regrowing it is what
        ``min_fit`` is for; this gate exists to catch the case where that
        wasn't done.  Skipped on the first window (no reference pose yet).
        Either ``None`` (the default) disables that half of the gate; both
        ``None`` disables it entirely.
    fibers_per_ribbon:
        The probe's fiber×ribbon combine factor (DESIGN.md §2.4) — number of
        fiber positions actually wired per ribbon channel.  Defaults to 10
        (the raw hardware width); pass the probe's actual value if it wires
        fewer.
    """
    tel_dir = Path(tel_dir)
    prb_dir = Path(prb_dir)
    z_tel = np.asarray(z_tel, dtype=float)
    validate_probe_footprint(n_probe_ch, fibers_per_ribbon)

    tel = load_detector(tel_dir)
    prb = load_detector(prb_dir)

    alignment, _ = fit_alignment(
        tel, z_tel, tot_thresh=tot_thresh, tot_weights=tot_weights
    )
    z_corr = alignment.corrected_z_tel(z_tel)

    window_ns = None if window_s is None else int(window_s * 1e9)
    acc = _WindowAccumulator(
        z_corr=z_corr,
        alignment=alignment,
        n_probe_ch=n_probe_ch,
        fibers_per_ribbon=fibers_per_ribbon,
        window_ns=window_ns,
        min_fit=min_fit,
        max_rigidity_resid_mm=max_rigidity_resid_mm,
        max_off_probe_mm=max_off_probe_mm,
        max_pose_jump_mm=max_pose_jump_mm,
        max_pose_jump_deg=max_pose_jump_deg,
    )

    stream = stream_coincidences(
        tel,
        prb,
        z_tel=z_tel,
        alignment=alignment,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
        min_anchor_planes=min_anchor_planes,
        fibers_per_ribbon=fibers_per_ribbon,
    )
    for co in stream:
        acc.push(co)
    acc.finalize()

    results = acc.results

    # Whole-run residual-RMS distribution — a diagnostic of window quality.
    if results:
        rms_vals = np.array([r.resid_rms for r in results])
        print(
            f"Residual RMS over {len(results)} window(s): "
            f"min={rms_vals.min():.1f}  median={np.median(rms_vals):.1f}  "
            f"max={rms_vals.max():.1f} mm"
        )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(results, out_dir / "pose_timeseries.csv")
        if make_plots and results:
            _plot_timeseries(results, out_dir / "pose_timeseries.png")
        print(f"Wrote {len(results)} window(s) to {out_dir}")

    return results


# ── CSV output ────────────────────────────────────────────────────────────────


def _write_csv(results: list[WindowResult], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "utc_start",
                "utc_end",
                "n_inliers",
                "t_x",
                "sigma_tx",
                "t_y",
                "sigma_ty",
                "z_p",
                "sigma_zp",
                "theta",
                "sigma_theta",
                "resid_rms",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.utc_start.isoformat(),
                    r.utc_end.isoformat(),
                    r.n_inliers,
                    f"{r.t_x:.6g}",
                    f"{r.sigma_tx:.6g}",
                    f"{r.t_y:.6g}",
                    f"{r.sigma_ty:.6g}",
                    f"{r.z_p:.6g}",
                    f"{r.sigma_zp:.6g}",
                    f"{r.theta:.6g}",
                    f"{r.sigma_theta:.6g}",
                    f"{r.resid_rms:.6g}",
                ]
            )


# ── Plot ──────────────────────────────────────────────────────────────────────


def _plot_timeseries(results: list[WindowResult], path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    times = [r.utc_start for r in results]
    vals = [
        np.array([r.t_x for r in results]),
        np.array([r.t_y for r in results]),
        np.array([r.z_p for r in results]),
    ]
    sigs = [
        np.array([r.sigma_tx for r in results]),
        np.array([r.sigma_ty for r in results]),
        np.array([r.sigma_zp for r in results]),
    ]
    ylabels = ["t_x  [mm]", "t_y  [mm]", "z_p  [mm]"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, lbl, vs, ss in zip(axs, ylabels, vals, sigs):
        ax.plot(times, vs, "o-", ms=5)
        ax.fill_between(times, vs - ss, vs + ss, alpha=0.3, label="±1σ")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    axs[0].set_title("Probe position vs time (centre-referenced ±1σ bands)")
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = MacroArgumentParser(
        prog="monrad-monitor",
        description="Probe pose monitoring over an acquisition.",
        epilog="Flags can be collected in a macro file and loaded with "
        "'@path/to/file.args' (one flag per line, '#' comments allowed); "
        "e.g. 'monrad-monitor @run.args --out other/'. Flags given on the "
        "command line after the @file override lines from the file.",
    )
    p.add_argument(
        "--telescope",
        type=Path,
        required=True,
        metavar="DIR",
        help="Telescope acquisition directory.",
    )
    p.add_argument(
        "--probe",
        type=Path,
        required=True,
        metavar="DIR",
        help="Probe acquisition directory.",
    )
    p.add_argument(
        "--z-tel",
        nargs="+",
        type=float,
        required=True,
        metavar="Z",
        help="Telescope plane z-positions (mm).",
    )
    p.add_argument(
        "--min-fit",
        type=int,
        default=MIN_FIT,
        metavar="N",
        help="Minimum coincidences, after any geometric gates, fed to a pose "
        "fit.  The raw batch grows past this count (and past --window-s, if "
        "given) whenever a gate strips it below the floor, up to "
        f"{RAW_CAP_MULTIPLIER}x --min-fit raw coincidences before the window "
        f"is dropped as contaminated (default: {MIN_FIT}).",
    )
    p.add_argument(
        "--min-anchor-planes",
        type=int,
        default=1,
        metavar="N",
        help="Minimum telescope planes with an unambiguous hit for a cluster to "
        "pass the no_anchor_plane gate.  0 disables the gate (default: 1).",
    )
    p.add_argument(
        "--max-rigidity-resid-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Pre-fit geometric gate (mm).  Drop coincidences whose track-vs-"
        "probe pairwise distances are inconsistent with a rigid transform — "
        "catches cross-particle wide-angle tracks that time-matched an "
        "unrelated probe hit, before they reach the pose fit.  Pose-free "
        "(z_ref = previous accepted window's z_p, or mean(z_corr) for the "
        "first window); tune from the whole-run pairwise-residual "
        "distribution. Off by default.",
    )
    p.add_argument(
        "--max-off-probe-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Pre-fit geometric gate (mm), applied after --max-rigidity-resid-"
        "mm.  Extrapolate each track to the previous accepted window's pose "
        "and drop it if it lands more than this far outside the probe's "
        "physical footprint.  Skipped on the first window (no reference pose "
        "yet). Off by default.",
    )
    p.add_argument(
        "--max-pose-jump-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Post-fit continuity gate (mm), checked after fit_probe_pose runs "
        "(unlike --max-rigidity-resid-mm/--max-off-probe-mm, which filter "
        "coincidences before the fit).  Reject a candidate window's fitted "
        "pose if its (t_x, t_y, z_p) corner has moved more than this far from "
        "the previous accepted window's pose -- the probe moves slowly, so a "
        "bigger jump means the window's gate-survivors are still internally "
        "coherent enough to fool the Mahalanobis cut.  On rejection the raw "
        "batch keeps growing (diluting the contamination) until it passes or "
        "hits the raw cap and is dropped.  Skipped on the first window (no "
        "reference pose yet).  Checked independently of --max-pose-jump-deg. "
        "Off by default.",
    )
    p.add_argument(
        "--max-pose-jump-deg",
        type=float,
        default=None,
        metavar="DEG",
        help="Post-fit continuity gate (degrees), companion to "
        "--max-pose-jump-mm: reject a candidate window if theta has rotated "
        "more than this many degrees from the previous accepted window's "
        "pose.  Checked independently because a rotation can partly cancel a "
        "corner translation when viewed at the probe centre.  Off by "
        "default.",
    )
    p.add_argument(
        "--window-s",
        type=float,
        default=None,
        metavar="SECS",
        help="Minimum window duration in seconds.  Omit for count-based batches "
        "of --min-fit coincidences.  When given, each window spans at least this "
        "long AND holds at least --min-fit coincidences (whichever is longer).",
    )
    p.add_argument(
        "--n-probe-ch",
        type=int,
        default=30,
        metavar="N",
        help="Probe channel count for centre-covariance propagation (default: 30).",
    )
    p.add_argument(
        "--fibers-per-ribbon",
        type=int,
        default=10,
        metavar="N",
        help="Probe fiber x ribbon combine factor (DESIGN.md section 2.4) -- "
        "number of fiber positions wired per ribbon channel (default: 10).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./pipeline_out/monitor"),
        help="Output directory (default: ./pipeline_out/monitor).",
    )
    p.add_argument("--tot-thresh", type=int, default=1)
    p.add_argument("--tot-weights", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib output.")
    return p


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, set[str]]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.min_fit < _MIN_COINCS:
        parser.error(
            f"--min-fit must be >= {_MIN_COINCS} (fit_probe_pose's hard "
            f"minimum); got {args.min_fit}"
        )
    if not 1 <= args.fibers_per_ribbon <= 10:
        parser.error(
            "--fibers-per-ribbon must be in 1..10 (a probe can wire at most "
            f"the 10 raw fiber positions); got {args.fibers_per_ribbon}"
        )

    # Which flags did the user actually type, vs leave at their default?
    # Re-parse the same argv with every default suppressed: a dest only
    # survives into the resulting namespace if the user supplied it.
    probe = _build_parser()
    for action in probe._actions:
        action.default = argparse.SUPPRESS
    explicit = set(vars(probe.parse_args(argv)).keys())
    return args, explicit


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args, explicit = _parse_args(argv)

    def _tag(name: str) -> str:
        return "(user-specified)" if name in explicit else "(default)"

    logger.info("=== Run configuration ===")
    logger.info("  Telescope data:      %s  %s", args.telescope, _tag("telescope"))
    logger.info("  Probe data:          %s  %s", args.probe, _tag("probe"))
    logger.info("  Output dir:          %s  %s", args.out, _tag("out"))
    logger.info(
        "  Telescope plane z (mm): %s  %s",
        "  ".join(f"{z:g}" for z in args.z_tel),
        _tag("z_tel"),
    )
    logger.info("  min_fit:             %s  %s", args.min_fit, _tag("min_fit"))
    logger.info(
        "  min_anchor_planes:   %s  %s",
        args.min_anchor_planes,
        _tag("min_anchor_planes"),
    )
    logger.info(
        "  max_rigidity_resid_mm: %s  %s",
        args.max_rigidity_resid_mm,
        _tag("max_rigidity_resid_mm"),
    )
    logger.info(
        "  max_off_probe_mm:    %s  %s", args.max_off_probe_mm, _tag("max_off_probe_mm")
    )
    logger.info(
        "  max_pose_jump_mm:    %s  %s", args.max_pose_jump_mm, _tag("max_pose_jump_mm")
    )
    logger.info(
        "  max_pose_jump_deg:   %s  %s",
        args.max_pose_jump_deg,
        _tag("max_pose_jump_deg"),
    )
    logger.info("  window_s:            %s  %s", args.window_s, _tag("window_s"))
    logger.info("  n_probe_ch:          %s  %s", args.n_probe_ch, _tag("n_probe_ch"))
    logger.info(
        "  fibers_per_ribbon:   %s  %s",
        args.fibers_per_ribbon,
        _tag("fibers_per_ribbon"),
    )
    logger.info("  tot_thresh:          %s  %s", args.tot_thresh, _tag("tot_thresh"))
    logger.info("  tot_weights:         %s  %s", args.tot_weights, _tag("tot_weights"))
    logger.info("  no_plots:            %s  %s", args.no_plots, _tag("no_plots"))

    results = monitor_probe(
        args.telescope,
        args.probe,
        window_s=args.window_s,
        z_tel=np.array(args.z_tel),
        n_probe_ch=args.n_probe_ch,
        out_dir=args.out,
        min_fit=args.min_fit,
        min_anchor_planes=args.min_anchor_planes,
        max_rigidity_resid_mm=args.max_rigidity_resid_mm,
        max_off_probe_mm=args.max_off_probe_mm,
        max_pose_jump_mm=args.max_pose_jump_mm,
        max_pose_jump_deg=args.max_pose_jump_deg,
        tot_thresh=args.tot_thresh,
        tot_weights=args.tot_weights,
        fibers_per_ribbon=args.fibers_per_ribbon,
        make_plots=not args.no_plots,
    )
    print(f"Fitted {len(results)} window(s).")
    for r in results:
        print(
            f"  {r.utc_start.strftime('%H:%M:%S')} – {r.utc_end.strftime('%H:%M:%S')}"
            f"  n_inliers={r.n_inliers}"
            f"  t_x={r.t_x:.1f}±{r.sigma_tx:.2f}"
            f"  t_y={r.t_y:.1f}±{r.sigma_ty:.2f}"
            f"  z_p={r.z_p:.1f}±{r.sigma_zp:.2f} mm"
            f"  rms={r.resid_rms:.1f}"
        )


if __name__ == "__main__":
    main()
