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

from ..alignment import AlignmentCorrection
from ..pose import Coincidence, PoseFitter, PoseResult, fit_probe_pose
from .io import (
    DetectorFiles,
    centre_cov_2x2,
    fit_alignment,
    load_detector,
    stream_coincidences,
)
from .resolution import interp_sigma_eff, n_required

# Minimum decoded coincidences in a window to attempt a pose fit — the same
# floor the streaming PoseFitter applies, so windowed fits never diverge from it.
MIN_FIT = PoseFitter.MIN_FIT

# Default location of the synthetic resolution study's σ_eff table (Step 1
# output), used to size adaptive windows.  Repo-relative: src/monrad/monitor →
# parents[3] is the repo root.
_DEFAULT_RESOLUTION_CSV = (
    Path(__file__).resolve().parents[3] / "reports" / "resolution" / "n_required.csv"
)


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


def _size_window(
    tel: DetectorFiles,
    prb: DetectorFiles,
    *,
    z_tel: np.ndarray,
    z_corr: np.ndarray,
    alignment: AlignmentCorrection,
    target_zp: float,
    resolution_csv: Path,
    inspect_coinc: int,
    tot_thresh: int,
    tot_weights: bool,
) -> tuple[float, dict]:
    """Derive a monitoring window from a target z_p resolution.

    Buffers up to ``inspect_coinc`` accepted coincidences from a short prefix of
    the stream, measures the accepted-coincidence rate over their time span, and
    fits an approximate probe pose.  The fitted ``z_p`` selects ``σ_eff,z`` from
    the synthetic resolution study (:func:`interp_sigma_eff`), which inverts via
    the ``σ_eff/√N`` law (:func:`n_required`) to the inlier budget ``N_req`` for
    the target σ.  The window is then ``N_req / rate``.

    Returns ``(window_s, diag)`` where ``diag`` carries the inspection inputs for
    reporting.  Raises when fewer than :data:`MIN_FIT` coincidences are
    collected (too sparse to size a window — pass ``window_s`` explicitly).
    """
    buffer: list[Coincidence] = []
    for co in stream_coincidences(
        tel,
        prb,
        z_tel=z_tel,
        alignment=alignment,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
    ):
        buffer.append(co)
        if len(buffer) >= inspect_coinc:
            break

    n_collected = len(buffer)
    if n_collected < MIN_FIT:
        raise RuntimeError(
            f"not enough coincidences to size a window (collected {n_collected}, "
            f"need ≥ {MIN_FIT}) — pass --window-s explicitly"
        )
    span_s = (buffer[-1].t_ns - buffer[0].t_ns) / 1e9
    if span_s <= 0:
        raise RuntimeError(
            "inspection coincidences span zero time — pass --window-s explicitly"
        )
    coinc_rate = n_collected / span_s

    pose = fit_probe_pose(buffer, z_corr, alignment)
    sigma_eff_z = interp_sigma_eff(resolution_csv, pose.z_p)
    n_req = n_required(sigma_eff_z, target_zp)
    window_s = n_req / coinc_rate

    diag = {
        "window_s": window_s,
        "target_zp": target_zp,
        "z_p_approx": pose.z_p,
        "coinc_rate": coinc_rate,
        "n_collected": n_collected,
        "span_s": span_s,
        "sigma_eff_z": sigma_eff_z,
        "N_req": n_req,
        "pose_n_inliers": pose.n_inliers,
    }
    return window_s, diag


def monitor_probe(
    tel_dir: Path,
    prb_dir: Path,
    *,
    window_s: float | None = None,
    z_tel: np.ndarray,
    n_probe_ch: int = 30,
    out_dir: Path | None = None,
    target_zp: float = 0.5,
    resolution_csv: Path = _DEFAULT_RESOLUTION_CSV,
    inspect_coinc: int = 300,
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
        Window duration in seconds.  When ``None`` (the default), the window is
        sized adaptively from ``target_zp`` via :func:`_size_window`: inspect a
        short prefix of the stream, measure the coincidence rate and an
        approximate ``z_p``, and pick the window that reaches the target z_p
        resolution.  Pass an explicit value to override (manual mode).
    z_tel:
        Telescope plane z-positions (mm).
    n_probe_ch:
        Probe channel count; used to propagate the pose covariance to the
        physical probe centre (see :func:`~monrad.monitor.io.centre_cov_2x2`).
    out_dir:
        If given, write ``pose_timeseries.csv`` and (when ``make_plots``)
        ``pose_timeseries.png`` here, plus ``window_meta.txt`` in adaptive mode.
    target_zp:
        Target z_p σ (mm) used to size the window in adaptive mode; ignored when
        ``window_s`` is given.
    resolution_csv:
        ``n_required.csv`` from the synthetic resolution study, providing
        ``σ_eff,z(z_p)`` for the window sizing.
    inspect_coinc:
        Number of accepted coincidences to buffer from the stream prefix when
        sizing the window adaptively.
    """
    tel_dir = Path(tel_dir)
    prb_dir = Path(prb_dir)
    z_tel = np.asarray(z_tel, dtype=float)

    # Adaptive mode needs the σ(N) study to size the window; fail fast with a
    # clear message before the (slow) alignment pass rather than with a bare
    # FileNotFoundError from interp_sigma_eff deep in _size_window.
    if window_s is None and not Path(resolution_csv).exists():
        raise FileNotFoundError(
            f"resolution study not found at {resolution_csv}. Run `monrad-resolution` "
            f"to generate it, or pass --window-s to size the window manually."
        )

    tel = load_detector(tel_dir)
    prb = load_detector(prb_dir)

    alignment, _ = fit_alignment(
        tel, z_tel, tot_thresh=tot_thresh, tot_weights=tot_weights
    )
    z_corr = alignment.corrected_z_tel(z_tel)

    window_diag: dict | None = None
    if window_s is None:
        window_s, window_diag = _size_window(
            tel,
            prb,
            z_tel=z_tel,
            z_corr=z_corr,
            alignment=alignment,
            target_zp=target_zp,
            resolution_csv=resolution_csv,
            inspect_coinc=inspect_coinc,
            tot_thresh=tot_thresh,
            tot_weights=tot_weights,
        )
        print(
            f"Adaptive window: {window_s:.1f} s  (target σ_zp={target_zp:g} mm)\n"
            f"  inspected {window_diag['n_collected']} coincidences over "
            f"{window_diag['span_s']:.1f} s → rate "
            f"{window_diag['coinc_rate']:.4g}/s\n"
            f"  z_p≈{window_diag['z_p_approx']:.1f} mm (pose n_inliers="
            f"{window_diag['pose_n_inliers']}) → σ_eff,z="
            f"{window_diag['sigma_eff_z']:.3f} mm → N_req="
            f"{window_diag['N_req']:.0f}"
        )

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

    for co in stream_coincidences(
        tel,
        prb,
        z_tel=z_tel,
        alignment=alignment,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
    ):
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
        if window_diag is not None:
            (out_dir / "window_meta.txt").write_text(
                f"window_s={window_diag['window_s']:.1f}  "
                f"target_zp={window_diag['target_zp']:g} mm  "
                f"z_p_approx={window_diag['z_p_approx']:.1f} mm  "
                f"sigma_eff_z={window_diag['sigma_eff_z']:.3f} mm  "
                f"N_req={window_diag['N_req']:.0f}  "
                f"coinc_rate={window_diag['coinc_rate']:.4g}/s  "
                f"n_inspect={window_diag['n_collected']}  "
                f"span_s={window_diag['span_s']:.1f}  "
                f"pose_n_inliers={window_diag['pose_n_inliers']}\n"
            )
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
        default=None,
        metavar="SECS",
        help="Window duration in seconds.  Omit to size adaptively from "
        "--target-zp (the coincidence rate sets the window).",
    )
    p.add_argument(
        "--target-zp",
        type=float,
        default=0.5,
        metavar="MM",
        help="Required z_p σ in mm; sizes the adaptive window.  Ignored when "
        "--window-s is given (default: 0.5).",
    )
    p.add_argument(
        "--resolution-csv",
        type=Path,
        default=_DEFAULT_RESOLUTION_CSV,
        metavar="CSV",
        help="Resolution-study n_required.csv supplying σ_eff,z(z_p) for adaptive "
        f"window sizing (default: {_DEFAULT_RESOLUTION_CSV}).",
    )
    p.add_argument(
        "--inspect-coinc",
        type=int,
        default=300,
        metavar="N",
        help="Coincidences to buffer from the stream prefix when sizing the "
        "adaptive window (default: 300).",
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
        target_zp=args.target_zp,
        resolution_csv=args.resolution_csv,
        inspect_coinc=args.inspect_coinc,
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
