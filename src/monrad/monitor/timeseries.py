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
spans ``window_s``), the configured geometric gates (``max_rigidity_resid_mm``,
``max_off_probe_mm``) run against it.  A gate only ever removes coincidences,
so a raw batch sized to exactly ``min_fit`` can drop below the floor after
gating; when that happens the batch keeps growing — pulling in more raw
coincidences and re-gating — until the *survivor* count clears ``min_fit``.
If the raw batch grows past ``RAW_CAP_MULTIPLIER * min_fit`` coincidences
without enough survivors, the window is abandoned as contaminated and
dropped; the trailing remainder at end-of-stream is dropped the same way.

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
    centre_cov_2x2,
    fit_alignment,
    load_detector,
    stream_coincidences,
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
    tot_thresh: int = 1,
    tot_weights: bool = False,
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
    """
    tel_dir = Path(tel_dir)
    prb_dir = Path(prb_dir)
    z_tel = np.asarray(z_tel, dtype=float)

    tel = load_detector(tel_dir)
    prb = load_detector(prb_dir)

    alignment, _ = fit_alignment(
        tel, z_tel, tot_thresh=tot_thresh, tot_weights=tot_weights
    )
    z_corr = alignment.corrected_z_tel(z_tel)
    probe_size_mm = n_probe_ch * 10.0

    results: list[WindowResult] = []
    prev_pose: PoseResult | None = None
    cold_start_z_ref: float | None = None
    cold_start_n = 0

    def _run_gates(
        coincs: list[Coincidence], utc_start: datetime, utc_end: datetime
    ) -> list[Coincidence]:
        """Apply the configured geometric gates once; return the survivors.

        Read-only w.r.t. ``prev_pose`` — callers decide whether/when to
        actually commit a fit from the result, so this can be called
        repeatedly on a growing raw batch without side effects.
        """
        nonlocal cold_start_z_ref, cold_start_n
        working = coincs
        if max_rigidity_resid_mm is not None:
            if prev_pose is not None:
                z_ref = prev_pose.z_p
            else:
                # Cold start: no previous accepted pose to anchor z_ref. The
                # rigidity residual scales with |z_ref - z_p|, so a probe far
                # from the telescope stack's mean z (~1500 mm in the testLab
                # setup: mean(z_corr)=-670 vs true z_p=+840) makes mean(z_corr)
                # useless and would flag every genuine coincidence, dropping
                # window 0 entirely and permanently starving prev_pose (it
                # never updates from None, so every later window inherits the
                # same bad z_ref). Bootstrap a quick ungated fit on this
                # window's own coincidences instead — its z_p is a much better
                # anchor even though the window may itself be contaminated.
                # It's a throwaway anchor, not the committed pose, so the
                # (expensive, four-step) bootstrap fit is cached and only
                # recomputed every COLD_START_REFIT_STRIDE raw coincidences
                # rather than on every single growth step (see the module
                # docstring for the cost this avoids).
                if (
                    cold_start_z_ref is None
                    or len(coincs) - cold_start_n >= COLD_START_REFIT_STRIDE
                ):
                    cold_start_z_ref = fit_probe_pose(working, z_corr, alignment).z_p
                    cold_start_n = len(coincs)
                z_ref = cold_start_z_ref
            working, dropped_rigid = filter_rigidity(
                working, z_ref, max_rigidity_resid_mm
            )
            if dropped_rigid:
                logger.info(
                    "Window %s–%s: rigidity gate dropped %d/%d coincidence(s) "
                    "(z_ref=%.1f mm).",
                    utc_start.isoformat(),
                    utc_end.isoformat(),
                    len(dropped_rigid),
                    len(coincs),
                    z_ref,
                )

        if max_off_probe_mm is not None and prev_pose is not None:
            n_before = len(working)
            working, dropped_fp = filter_off_probe(
                working, prev_pose, probe_size_mm, max_off_probe_mm
            )
            if dropped_fp:
                logger.info(
                    "Window %s–%s: footprint gate dropped %d/%d coincidence(s).",
                    utc_start.isoformat(),
                    utc_end.isoformat(),
                    len(dropped_fp),
                    n_before,
                )
        return working

    def _fit_and_record(
        working: list[Coincidence], utc_start: datetime, utc_end: datetime
    ) -> None:
        nonlocal prev_pose
        pose = fit_probe_pose(working, z_corr, alignment)

        # Absolute-mm residual RMS over ALL coincidences fed to the fit — the
        # honest window-quality signal.  The inlier-only residuals the fit
        # reports look clean even for a contaminated window, because the
        # Mahalanobis cut rejects the wild tracks; counting the rejected tracks
        # back in is what exposes the contamination (see _window_resid_rms).
        rms = _window_resid_rms(pose)

        cov_c = centre_cov_2x2(pose.cov, pose.theta, n_probe_ch)
        results.append(
            WindowResult(
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
        )
        prev_pose = pose

    def _utc(t_ns: float) -> datetime:
        return datetime.fromtimestamp(t_ns / 1e9, tz=timezone.utc)

    stream = stream_coincidences(
        tel,
        prb,
        z_tel=z_tel,
        alignment=alignment,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
        min_anchor_planes=min_anchor_planes,
    )

    # A geometric gate only ever removes coincidences, so a raw batch sized to
    # exactly min_fit can never yield >= min_fit survivors unless the gate
    # happens to drop nothing. Grow the raw batch past its nominal size,
    # re-gating on every new coincidence, until the *survivor* count clears
    # min_fit or the raw batch reaches RAW_CAP_MULTIPLIER * min_fit, at which
    # point the window is abandoned as contaminated. Without a gate, _run_gates
    # is a no-op and this reduces to the original fixed-size batching (first
    # check point == min_fit raw coincidences, always passes, same as before).
    window_ns = None if window_s is None else int(window_s * 1e9)
    raw_cap = min_fit * RAW_CAP_MULTIPLIER
    batch: list[Coincidence] = []
    win_start_ns: int | None = None
    for co in stream:
        if win_start_ns is None:
            win_start_ns = co.t_ns
        batch.append(co)

        spanned = window_ns is None or (co.t_ns - win_start_ns >= window_ns)
        if not spanned or len(batch) < min_fit:
            continue

        utc_start, utc_end = _utc(win_start_ns), _utc(co.t_ns)
        working = _run_gates(batch, utc_start, utc_end)
        if len(working) >= min_fit:
            _fit_and_record(working, utc_start, utc_end)
            batch = []
            win_start_ns = None
            cold_start_z_ref = None
            cold_start_n = 0
        elif len(batch) >= raw_cap:
            logger.warning(
                "Dropping window %s–%s: only %d/%d raw coincidence(s) survive "
                "the geometric gates after reaching the %dx raw cap "
                "(< min_fit=%d).",
                utc_start.isoformat(),
                utc_end.isoformat(),
                len(working),
                len(batch),
                RAW_CAP_MULTIPLIER,
                min_fit,
            )
            batch = []
            win_start_ns = None
            cold_start_z_ref = None
            cold_start_n = 0
        # else: not enough survivors yet — keep growing the raw batch and
        # re-gate on the next coincidence.

    if batch:
        utc_start, utc_end = _utc(win_start_ns), _utc(batch[-1].t_ns)
        logger.warning(
            "Dropping trailing window %s–%s: stream ended with only %d raw "
            "coincidence(s), never reaching min_fit=%d survivors.",
            utc_start.isoformat(),
            utc_end.isoformat(),
            len(batch),
            min_fit,
        )

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
    p = argparse.ArgumentParser(
        prog="monrad-monitor",
        description="Probe pose monitoring over an acquisition.",
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
    logger.info("  window_s:            %s  %s", args.window_s, _tag("window_s"))
    logger.info("  n_probe_ch:          %s  %s", args.n_probe_ch, _tag("n_probe_ch"))
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
        tot_thresh=args.tot_thresh,
        tot_weights=args.tot_weights,
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
