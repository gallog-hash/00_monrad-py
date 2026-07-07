"""Probe pose monitoring over an acquisition (monitoring Step 2).

Console script ``monrad-monitor``.  Streams an acquisition and emits one
probe pose per batch, tracking the probe's position over time with
per-batch uncertainty.

Two batching modes, both driven by ``min_fit`` (the minimum decoded
coincidences fed to :func:`~monrad.pose.fit_probe_pose`):

* **Count-based (default, ``window_s`` omitted).**  Buffer decoded
  :class:`~monrad.pose.Coincidence` objects until exactly ``min_fit`` are
  collected, fit once, emit a :class:`WindowResult`, then reset.  Each point
  is an independent fit over ``min_fit`` coincidences; the trailing remainder
  (``< min_fit``) is dropped.  The batch timestamps come from the first and
  last coincidence in the batch.
* **Hybrid (``window_s`` given).**  Each window grows until it spans at least
  ``window_s`` seconds *and* holds at least ``min_fit`` coincidences, whichever
  bound takes longer; a sparse window stretches past ``window_s`` to reach the
  count.  Batch timestamps are the first and last coincidence in the window.

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..pose import (
    Coincidence,
    PoseFitter,
    PoseResult,
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
    pose: PoseResult = field(repr=False)


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
    max_resid_rms_mm: float | None = None,
    max_abs_resid_mm: float | None = None,
    max_rigidity_resid_mm: float | None = None,
    max_off_probe_mm: float | None = None,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    make_plots: bool = True,
) -> list[WindowResult]:
    """Stream an acquisition and fit the probe pose in successive batches.

    Emits one :class:`WindowResult` per batch of at least ``min_fit`` decoded
    coincidences.  Only the open batch is buffered — RAM usage is bounded to
    roughly ``min_fit`` coincidences.

    Parameters
    ----------
    tel_dir, prb_dir:
        Acquisition directories for the telescope and probe.
    window_s:
        Window duration in seconds.  When ``None`` (the default), batches are
        count-based: each batch holds exactly ``min_fit`` coincidences and its
        timestamps come from the first and last coincidence.  When given, the
        mode is hybrid: each window grows until it spans at least ``window_s``
        seconds *and* holds at least ``min_fit`` coincidences (whichever bound
        takes longer), so a sparse window stretches past ``window_s`` to reach
        the count.
    z_tel:
        Telescope plane z-positions (mm).
    n_probe_ch:
        Probe channel count; used to propagate the pose covariance to the
        physical probe centre (see :func:`~monrad.monitor.io.centre_cov_2x2`).
    out_dir:
        If given, write ``pose_timeseries.csv`` and (when ``make_plots``)
        ``pose_timeseries.png`` here.
    min_fit:
        Minimum decoded coincidences fed to a pose fit.  In count-based mode it
        is the batch size; in time-window mode it is the per-window floor.
        Defaults to :data:`MIN_FIT`.
    min_anchor_planes:
        Minimum telescope planes with an unambiguous (single-candidate) hit for
        a cluster to survive the ``no_anchor_plane`` gate.  ``0`` disables the
        gate.  Defaults to ``1`` (matches :class:`~monrad.pose.PoseFitter`).
    max_resid_rms_mm:
        Window quality gate.  When given, a window whose ``resid_rms`` (the
        combined absolute-mm residual RMS over *all* coincidences fed to the
        fit; see :func:`_window_resid_rms`) exceeds this value is flagged,
        logged, and **dropped** (not emitted).  This catches windows
        contaminated by an excess of wide-angle "wild" telescope tracks.  Note
        the RMS is dominated by the ever-present wild-track baseline the fit
        correctly ignores, so its absolute scale is large and setup-dependent
        (~150 mm in the testLab data, with a contaminated window near ~280 mm) —
        tune the threshold per setup from the whole-run RMS distribution printed
        at the end of a run, not from a universal constant.  ``None`` (the
        default) disables the gate; ``resid_rms`` is recorded on every emitted
        window regardless.
    max_abs_resid_mm:
        Opt-in absolute-mm residual cut passed through to
        :func:`~monrad.pose.fit_probe_pose` (see there).  Unlike
        ``max_resid_rms_mm`` — which *drops the whole window* — this rejects the
        individual wide-angle "wild" coincidences *inside* the fit and refits on
        the survivors, recovering the good core rather than discarding the
        window.  The recovery path when the probe is far and good coincidences
        are scarce.  ``None`` (the default) ⇒ Mahalanobis-only, no change.
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
        window down to 0 survivors when none of its coincidences are genuine
        (the ``min_fit`` check below then skips it) — this is intentional;
        the gate does not preserve a floor.  ``None`` (the default) disables
        the gate.
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

    def _emit(
        coincs: list[Coincidence], utc_start: datetime, utc_end: datetime
    ) -> None:
        nonlocal prev_pose
        if len(coincs) < min_fit:
            return

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
                z_ref = fit_probe_pose(working, z_corr, alignment).z_p
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

        if len(working) < min_fit:
            logger.warning(
                "Dropping window %s–%s: only %d coincidence(s) survive the "
                "geometric gates (< min_fit=%d).",
                utc_start.isoformat(),
                utc_end.isoformat(),
                len(working),
                min_fit,
            )
            return

        pose = fit_probe_pose(
            working, z_corr, alignment, max_abs_resid_mm=max_abs_resid_mm
        )

        # Absolute-mm residual RMS over ALL coincidences fed to the fit — the
        # honest window-quality signal.  The inlier-only residuals the fit
        # reports look clean even for a contaminated window, because the
        # Mahalanobis cut rejects the wild tracks; counting the rejected tracks
        # back in is what exposes the contamination (see _window_resid_rms).
        rms = _window_resid_rms(pose)

        if max_resid_rms_mm is not None and rms > max_resid_rms_mm:
            logger.warning(
                "Dropping window %s–%s: residual RMS %.1f mm > %.1f mm "
                "(n_inliers=%d) — likely wide-angle track contamination.",
                utc_start.isoformat(),
                utc_end.isoformat(),
                rms,
                max_resid_rms_mm,
                pose.n_inliers,
            )
            return

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
                pose=pose,
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

    if window_s is None:
        # Count-based: a fresh fit every ``min_fit`` coincidences.  The trailing
        # remainder (< min_fit) is dropped by construction.
        batch: list[Coincidence] = []
        for co in stream:
            batch.append(co)
            if len(batch) >= min_fit:
                _emit(batch, _utc(batch[0].t_ns), _utc(batch[-1].t_ns))
                batch = []
    else:
        # Hybrid: each window spans at least ``window_s`` seconds AND holds at
        # least ``min_fit`` coincidences — whichever bound takes longer.  The
        # window opens at its first coincidence and closes on the first one that
        # satisfies both bounds; the trailing remainder is dropped.
        window_ns = int(window_s * 1e9)
        win_coincs: list[Coincidence] = []
        win_start_ns: int | None = None
        for co in stream:
            if win_start_ns is None:
                win_start_ns = co.t_ns
            win_coincs.append(co)
            spanned = co.t_ns - win_start_ns >= window_ns
            if spanned and len(win_coincs) >= min_fit:
                _emit(win_coincs, _utc(win_start_ns), _utc(co.t_ns))
                win_coincs = []
                win_start_ns = None

    # Whole-run residual-RMS distribution — helps set --max-resid-rms per setup.
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        help="Minimum coincidences fed to a pose fit.  Without --window-s this "
        "is the count-based batch size; with --window-s each window must reach "
        f"this count (stretching past --window-s if needed) (default: {MIN_FIT}).",
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
        "--max-resid-rms",
        type=float,
        default=None,
        metavar="MM",
        help="Window quality gate (mm).  Drop (and log) any window whose "
        "all-coincidence residual RMS (the resid_rms column) exceeds this — "
        "catches an excess of wide-angle 'wild' track contamination.  The RMS "
        "scale is large and setup-dependent (typical windows ~150 mm in the "
        "testLab data, a contaminated one ~280 mm); set the threshold from the "
        "whole-run RMS distribution printed at the end of a run, not a fixed "
        "value.  Off by default.",
    )
    p.add_argument(
        "--max-abs-resid",
        type=float,
        default=None,
        metavar="MM",
        help="Opt-in absolute-mm residual cut applied INSIDE each window's pose "
        "fit (on top of the Mahalanobis cut): reject any coincidence whose "
        "combined residual magnitude exceeds this and refit on the survivors. "
        "Unlike --max-resid-rms (which drops the whole window), this recovers "
        "the good core of a window contaminated by wide-angle 'wild' tracks — "
        "the far-probe recovery path. Tune per setup. Off by default.",
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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    results = monitor_probe(
        args.telescope,
        args.probe,
        window_s=args.window_s,
        z_tel=np.array(args.z_tel),
        n_probe_ch=args.n_probe_ch,
        out_dir=args.out,
        min_fit=args.min_fit,
        min_anchor_planes=args.min_anchor_planes,
        max_resid_rms_mm=args.max_resid_rms,
        max_abs_resid_mm=args.max_abs_resid,
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
