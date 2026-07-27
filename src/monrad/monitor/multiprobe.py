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

The combinatorial telescope track search inside ``PoseFitter`` depends only
on a cluster's telescope entry, not on which probe is asking (finding 9).
Every fitter built here shares the same ``tel_id``/``tel_z``/``alignment``/
``tot_thresh``/``tot_weights``/``min_anchor_planes``/``tel_pos_paths``, so
the search is run once per cluster
(:meth:`~monrad.pose.PoseFitter.decode_telescope_track`, called on
``fitters[0]``) and shared with every fitter's
:meth:`~monrad.pose.PoseFitter.decode_from_telescope_track` instead of each
fitter re-running it.
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from ..alignment import load_alignment
from ..coincidence import WINDOW_NS_DEFAULT
from ..pose import PoseFitter
from ..reconstruction import MAX_PER_PLANE_DEFAULT
from .cli_args import (
    MacroArgumentParser,
    add_alignment_arg,
    add_chi2_track_args,
    add_coincidence_window_ns_arg,
    add_mahal_cut_arg,
    add_max_off_probe_mm_arg,
    add_max_per_plane_arg,
    add_max_pose_jump_deg_arg,
    add_max_pose_jump_mm_arg,
    add_max_rigidity_resid_mm_arg,
    add_min_anchor_planes_arg,
    add_min_fit_arg,
    add_no_plots_arg,
    add_out_arg,
    add_telescope_arg,
    add_tot_thresh_arg,
    add_tot_weights_arg,
    add_window_s_arg,
    add_z_tel_arg,
    validate_chi2_track_args,
    validate_coincidence_window_ns,
    validate_fibers_per_ribbon,
    validate_mahal_cut,
    validate_max_per_plane,
    validate_min_fit,
)
from .io import (
    _cluster_tel_time,
    build_cluster_stream,
    fit_alignment,
    load_alignment_schedule,
    load_detector,
    static_alignment_label,
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


def _broadcast_per_probe(
    name: str, values: list[int] | None, default: int, n_probes: int
) -> list[int]:
    """Broadcast a singleton per-probe override to ``n_probes``, or pass
    through a value already given one-per-probe.

    ``values=None`` falls back to ``[default]``.  Any other length raises
    ``ValueError`` -- callers that only want to validate (e.g. ``_parse_args``,
    which must not mutate an already-broadcast-later CLI list) can call this
    for its raise and discard the return value.
    """
    if values is None:
        values = [default]
    if len(values) == 1:
        return list(values) * n_probes
    if len(values) == n_probes:
        return list(values)
    raise ValueError(
        f"{name} must have length 1 or {n_probes} (one per probe), got {len(values)}"
    )


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
    alignment_path: Path | None = None,
    make_plots: bool = True,
    chi2_track: float | None = None,
    max_cluster_width: int | None = None,
    mahal_cut: float | None = None,
    max_per_plane: int = MAX_PER_PLANE_DEFAULT,
    coincidence_window_ns: int = WINDOW_NS_DEFAULT,
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
    alignment_path:
        Optional path to a telescope alignment correction saved by
        ``monrad-align``.  When given, load it and skip the in-run alignment
        fit (the saved ``z_tel`` must match this run's).  ``None`` (the
        default) fits the alignment from this acquisition.
    chi2_track, max_cluster_width, mahal_cut, max_per_plane:
        Shared cuts (not per-probe) forwarded straight to every probe's
        :class:`~monrad.pose.PoseFitter`, exactly as in
        :func:`~monrad.monitor.timeseries.monitor_probe`. ``None`` keeps
        that fitter's own defaults (4.0 / off / ``fit_probe_pose``'s d > 4).
        ``mahal_cut`` additionally reaches each probe's ``_WindowAccumulator``
        fits.
    coincidence_window_ns:
        The stage-2 hardware coincidence window (DESIGN.md §5), shared by the
        single merged cluster stream.  Distinct from ``window_s``, the
        *monitoring* window each probe's accumulator fits a pose over.
    Other parameters mirror :func:`~monrad.monitor.timeseries.monitor_probe`
    and apply identically to every probe's independent accumulator.
    """
    tel_dir = Path(tel_dir)
    prb_dirs = [Path(d) for d in prb_dirs]
    if not prb_dirs:
        raise ValueError("monitor_probes requires at least one probe directory")
    z_tel = np.asarray(z_tel, dtype=float)

    n_probe_ch = _broadcast_per_probe("n_probe_ch", n_probe_ch, 30, len(prb_dirs))
    fibers_per_ribbon = _broadcast_per_probe(
        "fibers_per_ribbon", fibers_per_ribbon, 10, len(prb_dirs)
    )

    for k in range(len(prb_dirs)):
        try:
            validate_probe_footprint(n_probe_ch[k], fibers_per_ribbon[k])
        except ValueError as e:
            raise ValueError(f"probe {k + 1} ({prb_dirs[k]}): {e}") from e

    tel = load_detector(tel_dir)
    probes = [load_detector(d) for d in prb_dirs]

    schedule = None
    if alignment_path is not None and alignment_path.is_dir():
        schedule = load_alignment_schedule(alignment_path, expect_z_tel=z_tel)
        alignment = schedule.corrections[0]
        logger.info(
            "Loaded %d-window time-varying alignment from %s",
            len(schedule.corrections),
            alignment_path,
        )
    elif alignment_path is not None:
        alignment = load_alignment(alignment_path, expect_z_tel=z_tel)
        logger.info("Loaded telescope alignment from %s", alignment_path)
    else:
        alignment, _ = fit_alignment(
            tel, z_tel, tot_thresh=tot_thresh, tot_weights=tot_weights
        )
    # z_corr feeds only the z_p start guess in fit_probe_pose; under a schedule
    # the per-coincidence correction is switched below, so the first window's
    # value is a fine seed.
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
            chi2_track=chi2_track,
            max_cluster_width=max_cluster_width,
            mahal_cut=mahal_cut,
            max_per_plane=max_per_plane,
        )
        for k in range(len(probes))
    ]
    # The shared-telescope-search path below assumes every fitter's
    # telescope-side configuration is identical (finding 9) -- true by
    # construction above, but re-asserted here since PoseFitter.alignment is
    # mutable per-instance (update_alignment) and nothing stops a future
    # change here from diverging it per probe.
    assert all(f.tel_id == fitters[0].tel_id for f in fitters)
    assert all(f.alignment is fitters[0].alignment for f in fitters)
    assert all(f.tel_pos_paths == fitters[0].tel_pos_paths for f in fitters)
    assert all(np.array_equal(f.tel_z, fitters[0].tel_z) for f in fitters)
    assert all(f.tot_thresh == fitters[0].tot_thresh for f in fitters)
    assert all(f.tot_weights == fitters[0].tot_weights for f in fitters)
    assert all(f.min_anchor_planes == fitters[0].min_anchor_planes for f in fitters)
    assert all(f.chi2_track == fitters[0].chi2_track for f in fitters)
    assert all(f.max_cluster_width == fitters[0].max_cluster_width for f in fitters)
    # max_per_plane is telescope-side (it caps reconstruct_plane_candidates in
    # the shared search), so it must match too.  mahal_cut is probe-side --
    # it only reaches each probe's own pose fit -- and is deliberately not
    # asserted here.
    assert all(f.max_per_plane == fitters[0].max_per_plane for f in fitters)
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
            mahal_cut=mahal_cut,
            label=f"probe{k + 1}",
        )
        for k in range(len(probes))
    ]

    label = static_alignment_label(alignment_path)
    for cluster in build_cluster_stream(tel, probes, window_ns=coincidence_window_ns):
        if schedule is not None:
            t_ns = _cluster_tel_time(cluster, tel_id=0)
            if t_ns is not None:
                corr = schedule.at(t_ns)
                if corr is not fitters[0].alignment:
                    # Switch all fitters to the *same* object so the shared-
                    # search identity invariant (asserted above) still holds;
                    # only fitters[0].alignment is read by the shared decode.
                    for f in fitters:
                        f.update_alignment(corr)
                label = schedule.label_at(t_ns)
        tel_result = fitters[0].decode_telescope_track(cluster)
        for fitter, acc in zip(fitters, accumulators):
            co = fitter.decode_from_telescope_track(cluster, tel_result)
            if co is not None:
                acc.push(co._replace(alignment_label=label))
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
    p = MacroArgumentParser(
        prog="monrad-multiprobe",
        description="Multi-probe pose monitoring over one telescope acquisition.",
        epilog="Flags can be collected in a macro file and loaded with "
        "'@path/to/file.args' (one flag per line, '#' comments allowed, "
        "repeatable flags like --probe on their own lines); e.g. "
        "'monrad-multiprobe @run.args --out other/'. Single-value flags "
        "given on the command line after the @file override the file's "
        "value. --probe is append-only, though: it accumulates across the "
        "file and the command line rather than overriding, so a CLI "
        "--probe on top of a macro file adds a probe instead of replacing "
        "the file's list. --n-probe-ch/--fibers-per-ribbon take the whole "
        "space-separated list at once, so repeating them (file vs CLI) "
        "replaces rather than appends.",
    )
    add_telescope_arg(p, help_suffix=" (shared by every probe).")
    p.add_argument(
        "--probe",
        dest="probe",
        type=Path,
        action="append",
        required=True,
        metavar="DIR",
        help="Probe acquisition directory.  Repeatable — pass once per probe.",
    )
    add_z_tel_arg(p)
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
    add_min_fit_arg(
        p,
        help_suffix=(
            "  Applied independently per probe, same semantics as "
            "monrad-monitor's --min-fit."
        ),
    )
    add_min_anchor_planes_arg(p)
    add_max_rigidity_resid_mm_arg(p, help_suffix=" Applied independently per probe.")
    add_max_off_probe_mm_arg(p, help_suffix=" Applied independently per probe.")
    add_max_pose_jump_mm_arg(p)
    add_max_pose_jump_deg_arg(p)
    add_window_s_arg(
        p,
        help_suffix=(
            "  Same semantics as monrad-monitor's --window-s, applied "
            "independently per probe."
        ),
    )
    add_alignment_arg(p)
    add_out_arg(p, default=Path("./pipeline_out/multiprobe"))
    add_chi2_track_args(p, shared_across_probes=True)
    add_mahal_cut_arg(p, shared_across_probes=True)
    add_max_per_plane_arg(p, shared_across_probes=True)
    add_coincidence_window_ns_arg(p)
    add_tot_thresh_arg(p)
    add_tot_weights_arg(p)
    add_no_plots_arg(p)
    return p


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, set[str]]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    n_probes = len(args.probe)
    try:
        validate_min_fit(args.min_fit)
        _broadcast_per_probe("--n-probe-ch", args.n_probe_ch, 30, n_probes)
        _broadcast_per_probe(
            "--fibers-per-ribbon", args.fibers_per_ribbon, 10, n_probes
        )
        validate_fibers_per_ribbon(args.fibers_per_ribbon)
        validate_chi2_track_args(args)
        validate_mahal_cut(args.mahal_cut)
        validate_max_per_plane(args.max_per_plane)
        validate_coincidence_window_ns(args.coincidence_window_ns)
    except ValueError as exc:
        parser.error(str(exc))

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
    logger.info(
        "  Alignment source:    %s  %s",
        args.alignment if args.alignment is not None else "(fit from this run)",
        _tag("alignment"),
    )
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
    logger.info("  chi2_track:          %s  %s", args.chi2_track, _tag("chi2_track"))
    logger.info(
        "  max_cluster_width:   %s  %s",
        args.max_cluster_width,
        _tag("max_cluster_width"),
    )
    logger.info("  mahal_cut:           %s  %s", args.mahal_cut, _tag("mahal_cut"))
    logger.info(
        "  max_per_plane:       %s  %s", args.max_per_plane, _tag("max_per_plane")
    )
    logger.info(
        "  coincidence_window_ns: %s  %s",
        args.coincidence_window_ns,
        _tag("coincidence_window_ns"),
    )
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
        alignment_path=args.alignment,
        make_plots=not args.no_plots,
        chi2_track=args.chi2_track,
        max_cluster_width=args.max_cluster_width,
        mahal_cut=args.mahal_cut,
        max_per_plane=(
            MAX_PER_PLANE_DEFAULT if args.max_per_plane is None else args.max_per_plane
        ),
        coincidence_window_ns=(
            WINDOW_NS_DEFAULT
            if args.coincidence_window_ns is None
            else args.coincidence_window_ns
        ),
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
