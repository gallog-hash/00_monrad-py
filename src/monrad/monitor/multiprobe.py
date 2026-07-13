"""Multi-probe pose monitoring over one telescope acquisition (monitoring Step 3).

Console script ``monrad-multiprobe``.  Extends monitoring Step 2
(``monrad-monitor``, :mod:`monrad.monitor.timeseries`) to N probes sharing one
telescope acquisition: one shared coincidence-cluster stream
(:func:`~monrad.monitor.io.build_cluster_stream`), decoded once per probe via
that probe's own :class:`~monrad.pose.PoseFitter` (``PoseFitter._decode_cluster``'s
documented multi-probe contract — a cluster only yields a
:class:`~monrad.pose.Coincidence` for the probe(s) it is actually consistent
with), and one independent
:class:`~monrad.monitor.timeseries._WindowAccumulator` per probe, so each
probe's windowing/gating/continuity state (raw batch, cold-start anchor,
previous accepted pose) never mixes with another probe's.

Gate thresholds (``max_rigidity_resid_mm``, ``max_off_probe_mm``,
``max_pose_jump_mm``/``max_pose_jump_deg``, ``min_fit``, ``window_s``,
``min_anchor_planes``) apply identically to every probe.  ``n_probe_ch`` is
the one per-probe override: physical probe footprint size commonly differs
between probes in a multi-probe deployment, and it feeds directly into the
off-probe gate and the centre-covariance propagation.  Pass one value to
apply it to every probe, or one value per probe (matching ``--probe``'s
order 1:1).

Known, deferred inefficiency: the combinatorial telescope track search
inside ``PoseFitter._decode_cluster`` depends only on a cluster's telescope
entry, not on which probe is asking, so with N probes every cluster pays
that search N times over.  Not fixed here (would need a per-cluster
memoization keyed off the telescope ``PosRef``, shared across fitters) —
harmless for the small N a real deployment is expected to have; flagged as
a follow-up alongside the Step 0b deferred-optimization list.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ..pose import PoseFitter, _MIN_COINCS
from .io import (
    build_cluster_stream,
    fit_alignment,
    load_detector,
    validate_probe_footprint,
)
from .timeseries import (
    MIN_FIT,
    WindowResult,
    _plot_timeseries,
    _WindowAccumulator,
    _write_csv,
)

logger = logging.getLogger(__name__)


def monitor_probes(
    tel_dir: Path,
    prb_dirs: list[Path],
    *,
    window_s: float | None = None,
    z_tel: np.ndarray,
    n_probe_ch: list[int] | None = None,
    out_dir: Path | None = None,
    min_fit: int = MIN_FIT,
    min_anchor_planes: int = 1,
    max_rigidity_resid_mm: float | None = None,
    max_off_probe_mm: float | None = None,
    max_pose_jump_mm: float | None = None,
    max_pose_jump_deg: float | None = None,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    fibers_per_ribbon: list[int] | None = None,
    make_plots: bool = True,
) -> list[list[WindowResult]]:
    """Stream one acquisition and fit N probes' poses independently.

    One inner list per probe, same order as ``prb_dirs``.  See the module
    docstring for the shared-stream / per-probe-accumulator design and
    :func:`~monrad.monitor.timeseries.monitor_probe`'s docstring for the
    windowing/gating contract each probe's accumulator applies
    independently.

    Parameters
    ----------
    tel_dir:
        Telescope acquisition directory (shared by every probe).
    prb_dirs:
        Probe acquisition directories, at least one.
    n_probe_ch:
        Probe channel count(s).  Either one value (broadcast to every
        probe) or one value per ``prb_dirs`` entry, in the same order.
        ``None`` defaults to ``[30]`` (broadcast).
    fibers_per_ribbon:
        Probe fiber×ribbon combine factor(s) (DESIGN.md §2.4).  Same
        broadcast-or-one-per-probe contract as ``n_probe_ch``.  ``None``
        defaults to ``[10]`` (broadcast).
    Other parameters mirror :func:`~monrad.monitor.timeseries.monitor_probe`
    and apply identically to every probe's independent accumulator.
    """
    tel_dir = Path(tel_dir)
    prb_dirs = [Path(d) for d in prb_dirs]
    if not prb_dirs:
        raise ValueError("monitor_probes requires at least one probe directory")
    z_tel = np.asarray(z_tel, dtype=float)

    if n_probe_ch is None:
        n_probe_ch = [30]
    if len(n_probe_ch) == 1:
        n_probe_ch = n_probe_ch * len(prb_dirs)
    if len(n_probe_ch) != len(prb_dirs):
        raise ValueError(
            f"n_probe_ch must have length 1 or {len(prb_dirs)} (one per probe), "
            f"got {len(n_probe_ch)}"
        )

    if fibers_per_ribbon is None:
        fibers_per_ribbon = [10]
    if len(fibers_per_ribbon) == 1:
        fibers_per_ribbon = fibers_per_ribbon * len(prb_dirs)
    if len(fibers_per_ribbon) != len(prb_dirs):
        raise ValueError(
            f"fibers_per_ribbon must have length 1 or {len(prb_dirs)} (one per "
            f"probe), got {len(fibers_per_ribbon)}"
        )

    for k in range(len(prb_dirs)):
        try:
            validate_probe_footprint(n_probe_ch[k], fibers_per_ribbon[k])
        except ValueError as e:
            raise ValueError(f"probe {k + 1} ({prb_dirs[k]}): {e}") from e

    tel = load_detector(tel_dir)
    probes = [load_detector(d) for d in prb_dirs]

    alignment, _ = fit_alignment(
        tel, z_tel, tot_thresh=tot_thresh, tot_weights=tot_weights
    )
    z_corr = alignment.corrected_z_tel(z_tel)
    window_ns = None if window_s is None else int(window_s * 1e9)

    fitters = [
        PoseFitter(
            tel_z=z_tel,
            alignment=alignment,
            tel_id=0,
            prb_id=k + 1,
            tel_pos_paths=tel.pos_paths,
            prb_pos_paths=probes[k].pos_paths,
            tot_thresh=tot_thresh,
            tot_weights=tot_weights,
            min_anchor_planes=min_anchor_planes,
            prb_fibers_per_ribbon=fibers_per_ribbon[k],
        )
        for k in range(len(probes))
    ]
    accumulators = [
        _WindowAccumulator(
            z_corr=z_corr,
            alignment=alignment,
            n_probe_ch=n_probe_ch[k],
            fibers_per_ribbon=fibers_per_ribbon[k],
            window_ns=window_ns,
            min_fit=min_fit,
            max_rigidity_resid_mm=max_rigidity_resid_mm,
            max_off_probe_mm=max_off_probe_mm,
            max_pose_jump_mm=max_pose_jump_mm,
            max_pose_jump_deg=max_pose_jump_deg,
            label=f"probe{k + 1}",
        )
        for k in range(len(probes))
    ]

    for cluster in build_cluster_stream(tel, probes):
        for fitter, acc in zip(fitters, accumulators):
            co = fitter.decode_cluster(cluster)
            if co is not None:
                acc.push(co)
    for acc in accumulators:
        acc.finalize()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[list[WindowResult]] = []
    for k, acc in enumerate(accumulators):
        results = acc.results
        all_results.append(results)
        if results:
            rms_vals = np.array([r.resid_rms for r in results])
            print(
                f"Probe {k + 1}: residual RMS over {len(results)} window(s): "
                f"min={rms_vals.min():.1f}  median={np.median(rms_vals):.1f}  "
                f"max={rms_vals.max():.1f} mm"
            )
        if out_dir is not None:
            _write_csv(results, out_dir / f"pose_timeseries_probe{k + 1}.csv")
            if make_plots and results:
                _plot_timeseries(results, out_dir / f"pose_timeseries_probe{k + 1}.png")
            print(f"Probe {k + 1}: wrote {len(results)} window(s) to {out_dir}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="monrad-multiprobe",
        description="Multi-probe pose monitoring over one telescope acquisition.",
    )
    p.add_argument(
        "--telescope",
        type=Path,
        required=True,
        metavar="DIR",
        help="Telescope acquisition directory (shared by every probe).",
    )
    p.add_argument(
        "--probe",
        dest="probe",
        type=Path,
        action="append",
        required=True,
        metavar="DIR",
        help="Probe acquisition directory.  Repeatable — pass once per probe.",
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
        "--n-probe-ch",
        nargs="+",
        type=int,
        default=[30],
        metavar="N",
        help="Probe channel count(s), for the off-probe footprint gate and "
        "centre-covariance propagation.  Pass one value to apply it to every "
        "probe, or one value per --probe (same order).  Probes commonly differ "
        "in channel count in a multi-probe setup (default: 30, all probes).",
    )
    p.add_argument(
        "--fibers-per-ribbon",
        nargs="+",
        type=int,
        default=[10],
        metavar="N",
        help="Probe fiber x ribbon combine factor(s) (DESIGN.md section 2.4) -- "
        "number of fiber positions wired per ribbon channel.  Pass one value to "
        "apply it to every probe, or one value per --probe (same order) "
        "(default: 10, all probes).",
    )
    p.add_argument(
        "--min-fit",
        type=int,
        default=MIN_FIT,
        metavar="N",
        help="Minimum coincidences, after any geometric gates, fed to each "
        f"probe's pose fit (default: {MIN_FIT}).  Same semantics as "
        "monrad-monitor's --min-fit, applied independently per probe.",
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
        help="Pre-fit geometric gate (mm), applied independently per probe.  See "
        "monrad-monitor's --max-rigidity-resid-mm.  Off by default.",
    )
    p.add_argument(
        "--max-off-probe-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Pre-fit geometric gate (mm), applied independently per probe.  See "
        "monrad-monitor's --max-off-probe-mm.  Off by default.",
    )
    p.add_argument(
        "--max-pose-jump-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Post-fit continuity gate (mm), applied independently per probe.  "
        "See monrad-monitor's --max-pose-jump-mm.  Off by default.",
    )
    p.add_argument(
        "--max-pose-jump-deg",
        type=float,
        default=None,
        metavar="DEG",
        help="Post-fit continuity gate (degrees), companion to "
        "--max-pose-jump-mm.  Off by default.",
    )
    p.add_argument(
        "--window-s",
        type=float,
        default=None,
        metavar="SECS",
        help="Minimum window duration in seconds.  Omit for count-based batches "
        "of --min-fit coincidences.  Same semantics as monrad-monitor's "
        "--window-s, applied independently per probe.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./pipeline_out/multiprobe"),
        help="Output directory (default: ./pipeline_out/multiprobe).",
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
    n_probes = len(args.probe)
    if len(args.n_probe_ch) not in (1, n_probes):
        parser.error(
            f"--n-probe-ch must have length 1 or {n_probes} (matching the "
            f"number of --probe flags); got {len(args.n_probe_ch)}"
        )
    if len(args.fibers_per_ribbon) not in (1, n_probes):
        parser.error(
            f"--fibers-per-ribbon must have length 1 or {n_probes} (matching "
            f"the number of --probe flags); got {len(args.fibers_per_ribbon)}"
        )
    for n in args.fibers_per_ribbon:
        if not 1 <= n <= 10:
            parser.error(
                "--fibers-per-ribbon values must be in 1..10 (a probe can "
                f"wire at most the 10 raw fiber positions); got {n}"
            )

    # Which flags did the user actually type, vs leave at their default?
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
    logger.info("  n_probe_ch:          %s  %s", args.n_probe_ch, _tag("n_probe_ch"))
    logger.info(
        "  fibers_per_ribbon:   %s  %s",
        args.fibers_per_ribbon,
        _tag("fibers_per_ribbon"),
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
    logger.info("  tot_thresh:          %s  %s", args.tot_thresh, _tag("tot_thresh"))
    logger.info("  tot_weights:         %s  %s", args.tot_weights, _tag("tot_weights"))
    logger.info("  no_plots:            %s  %s", args.no_plots, _tag("no_plots"))

    all_results = monitor_probes(
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
    for k, results in enumerate(all_results):
        print(f"Probe {k + 1}: fitted {len(results)} window(s).")
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
