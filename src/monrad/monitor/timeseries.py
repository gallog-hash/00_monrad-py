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
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..pose import Coincidence, PoseFitter, PoseResult, fit_probe_pose
from .io import (
    centre_cov_2x2,
    fit_alignment,
    load_detector,
    stream_coincidences,
)

# Default minimum decoded coincidences fed to a pose fit — the same floor the
# streaming PoseFitter applies, so monitored fits never diverge from it.  Used
# both as the count-based batch size and as the time-window floor.
MIN_FIT = PoseFitter.MIN_FIT


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

    results: list[WindowResult] = []

    def _emit(
        coincs: list[Coincidence], utc_start: datetime, utc_end: datetime
    ) -> None:
        if len(coincs) < min_fit:
            return
        pose = fit_probe_pose(coincs, z_corr, alignment)
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
                pose=pose,
            )
        )

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
        )


if __name__ == "__main__":
    main()
