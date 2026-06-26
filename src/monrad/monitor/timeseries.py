"""Time-windowed probe monitoring (monitoring Step 2).

Console script ``monrad-monitor``.  Streams an acquisition and emits one
probe pose per time window, tracking the probe's position over time with
per-window uncertainty.

The streaming loop buckets decoded :class:`~monrad.pose.Coincidence` objects
by ``t_ns // window_ns``; on window close it calls
:func:`~monrad.pose.fit_probe_pose` on that window's coincidences and records
a :class:`WindowResult`.  Only the open window is ever buffered in RAM.

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

from ..coincidence import coincidence_stream
from ..pose import Coincidence, PoseFitter, PoseResult, fit_probe_pose
from ..timing import reconstruct_stream
from .io import centre_cov_2x2, fit_alignment, load_detector

# Minimum decoded coincidences in a window to attempt a pose fit.
# Mirrors PoseFitter.MIN_FIT so windowed fits use the same floor.
MIN_FIT = 30


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
    window_s: float,
    z_tel: np.ndarray,
    n_probe_ch: int = 30,
    out_dir: Path | None = None,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    make_plots: bool = True,
) -> list[WindowResult]:
    """Stream an acquisition and fit the probe pose in successive time windows.

    Yields one :class:`WindowResult` per window that accumulates at least
    :data:`MIN_FIT` decoded coincidences.  The final (possibly shorter)
    window is included if it meets the minimum.  Only the open window is
    buffered — RAM usage is bounded to roughly ``MIN_FIT`` coincidences.

    Parameters
    ----------
    tel_dir, prb_dir:
        Acquisition directories for the telescope and probe.
    window_s:
        Window duration in seconds.  Choose based on the coincidence rate and
        the N_required table from ``monrad-resolution``.
    z_tel:
        Telescope plane z-positions (mm).
    n_probe_ch:
        Probe channel count; used to propagate the pose covariance to the
        physical probe centre (see :func:`~monrad.monitor.io.centre_cov_2x2`).
    out_dir:
        If given, write ``pose_timeseries.csv`` and (when ``make_plots``)
        ``pose_timeseries.png`` here.
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

    fitter = PoseFitter(
        tel_z=z_tel,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel.pos_paths,
        prb_pos_paths=prb.pos_paths,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
    )

    tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
    prb_stream = reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0)

    window_ns = int(window_s * 1e9)
    current_win: int | None = None
    win_coincs: list[Coincidence] = []
    results: list[WindowResult] = []

    def _close_window(win_idx: int, coincs: list[Coincidence]) -> None:
        if len(coincs) < MIN_FIT:
            return
        pose = fit_probe_pose(coincs, z_corr, alignment)
        cov_c = centre_cov_2x2(pose.cov, pose.theta, n_probe_ch)
        utc_s = datetime.fromtimestamp(win_idx * window_ns / 1e9, tz=timezone.utc)
        utc_e = datetime.fromtimestamp((win_idx + 1) * window_ns / 1e9, tz=timezone.utc)
        results.append(
            WindowResult(
                utc_start=utc_s,
                utc_end=utc_e,
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

    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        co = fitter.decode_cluster(cluster)
        if co is None:
            continue
        win = co.t_ns // window_ns
        if current_win is None:
            current_win = win
        if win != current_win:
            _close_window(current_win, win_coincs)
            win_coincs = []
            current_win = win
        win_coincs.append(co)

    if current_win is not None:
        _close_window(current_win, win_coincs)

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
        description="Time-windowed probe pose monitoring.",
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
        "--window-s",
        type=float,
        required=True,
        metavar="SECS",
        help="Window duration in seconds.",
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
