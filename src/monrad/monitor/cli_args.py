"""Argparse builders shared across the pose-fitting drivers and
``scripts/run_pipeline.py``, keeping each flag's type/default/metavar/help in
one place.

Flags whose shape genuinely differs per driver (``--probe``, ``--n-probe-ch``,
``--fibers-per-ribbon``, run_pipeline's fixed-3 ``--z-tel``) are *not* covered
here -- each driver keeps its own ``add_argument`` line for those.
"""

import argparse
import shlex
from pathlib import Path

from ..coincidence import WINDOW_NS_DEFAULT
from ..pose import _MAHAL_CUT, _MIN_COINCS, PoseFitter
from ..reconstruction import MAX_PER_PLANE_DEFAULT

MIN_FIT = PoseFitter.MIN_FIT


class MacroArgumentParser(argparse.ArgumentParser):
    """``ArgumentParser`` with ``@file`` macro-file expansion built in.

    Any command-line token starting with ``@`` (argparse's
    ``fromfile_prefix_chars``) is read as a macro file: one flag per line
    (``--min-fit 50``), ``#`` comments and blank lines ignored, values
    tokenized with :func:`shlex.split` so quoted paths with spaces survive.
    This overrides argparse's own default ``convert_arg_line_to_args``, which
    treats each line as a single argument -- unusable for ``--flag value``
    pairs. A macro file can be combined with, or overridden by, ordinary CLI
    flags (later flags win), e.g. ``monrad-monitor @run.args --out other/``.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("fromfile_prefix_chars", "@")
        super().__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        line = arg_line.split("#", 1)[0].strip()
        if not line:
            return []
        return shlex.split(line)


def add_telescope_arg(p: argparse.ArgumentParser, *, help_suffix: str = "") -> None:
    p.add_argument(
        "--telescope",
        type=Path,
        required=True,
        metavar="DIR",
        help="Telescope acquisition directory." + help_suffix,
    )


def add_out_arg(p: argparse.ArgumentParser, *, default: Path) -> None:
    p.add_argument(
        "--out",
        type=Path,
        default=default,
        metavar="DIR",
        help=f"Output directory (default: {default}).",
    )


def add_tot_thresh_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tot-thresh",
        type=int,
        default=1,
        metavar="N",
        help="Minimum number of the 16 rows in which a bit must fire "
        "to be kept in the OR mask (default: 1 = plain OR). "
        "Values 2-4 filter single-row cross-talk spikes.",
    )


def add_tot_weights_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--tot-weights",
        action="store_true",
        default=False,
        help="Weight cluster centroids by per-bit TOT counts "
        "(ribbon_count x fiber_count). No effect on golden hits. Off by "
        "default.",
    )


def add_z_tel_arg(p: argparse.ArgumentParser, *, help_suffix: str = "") -> None:
    p.add_argument(
        "--z-tel",
        nargs="+",
        type=float,
        required=True,
        metavar="Z",
        help="Telescope plane z-positions (mm)." + help_suffix,
    )


def add_no_plots_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib output.")


def add_alignment_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--alignment",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a telescope alignment correction saved by monrad-align, "
        "OR a directory of alignment_<label>.json files for time-varying "
        "correction (the active window is switched as the stream crosses "
        "boundaries). When given, load it and skip the in-run alignment fit "
        "(every saved --z-tel must match this run's). Off by default (fit from "
        "this acquisition).",
    )


def add_min_anchor_planes_arg(
    p: argparse.ArgumentParser, *, choices: bool = False
) -> None:
    p.add_argument(
        "--min-anchor-planes",
        type=int,
        default=1,
        metavar="N",
        choices=range(0, 4) if choices else None,
        help="Minimum telescope planes that must decode to a single resolved "
        "candidate (an 'anchor') before the combinatorial track search runs, "
        "0-3 (default: 1). 1 keeps the original gate; 0 also searches "
        "all-ambiguous clusters (more tracks, much heavier compute, pile-up "
        "can fabricate tracks); 3 demands every plane already resolved.",
    )


def add_max_rigidity_resid_mm_arg(
    p: argparse.ArgumentParser, *, help_suffix: str = ""
) -> None:
    p.add_argument(
        "--max-rigidity-resid-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Pre-fit geometric gate (mm), applied once to the whole run's "
        "accepted coincidences before the final pose fit: drops coincidences "
        "whose track-vs-probe pairwise distances (DESIGN.md docs/handoffs/"
        "2026-07-07-off-probe-track-gate-strategy.md) are inconsistent with a "
        "rigid transform -- catches combinatorial telescope-track picks that "
        "look fine over the telescope's own short baseline but are wrong once "
        "extrapolated out to z_p. Off by default." + help_suffix,
    )


def add_max_off_probe_mm_arg(
    p: argparse.ArgumentParser, *, help_suffix: str = ""
) -> None:
    p.add_argument(
        "--max-off-probe-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Pre-fit geometric gate (mm), applied after --max-rigidity-resid-"
        "mm: drops coincidences whose track projects more than this far "
        "outside the probe's footprint. Off by default." + help_suffix,
    )


def add_min_fit_arg(p: argparse.ArgumentParser, *, help_suffix: str = "") -> None:
    p.add_argument(
        "--min-fit",
        type=int,
        default=MIN_FIT,
        metavar="N",
        help=f"Minimum coincidences, after any geometric gates, fed to a pose "
        f"fit (default: {MIN_FIT})." + help_suffix,
    )


def add_window_s_arg(p: argparse.ArgumentParser, *, help_suffix: str = "") -> None:
    p.add_argument(
        "--window-s",
        type=float,
        default=None,
        metavar="SECS",
        help="Minimum window duration in seconds. Omit for count-based batches "
        "of --min-fit coincidences." + help_suffix,
    )


def add_max_pose_jump_mm_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-pose-jump-mm",
        type=float,
        default=None,
        metavar="MM",
        help="Post-fit continuity gate (mm): reject a candidate window's "
        "fitted pose if its (t_x, t_y, z_p) corner has moved more than this "
        "far from the previous accepted window's pose. Off by default.",
    )


def add_max_pose_jump_deg_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--max-pose-jump-deg",
        type=float,
        default=None,
        metavar="DEG",
        help="Post-fit continuity gate (degrees), companion to "
        "--max-pose-jump-mm: reject a candidate window if theta has rotated "
        "more than this many degrees from the previous accepted window's "
        "pose. Off by default.",
    )


def add_chi2_track_args(
    p: argparse.ArgumentParser, *, shared_across_probes: bool = False
) -> None:
    """Add ``--chi2-track``/``--max-cluster-width`` to a driver's parser.

    Shared by ``scripts/run_pipeline.py`` and the ``monrad-monitor``/
    ``monrad-multiprobe`` drivers so the two flags' type/default/help stay in
    lockstep. ``shared_across_probes`` selects multiprobe's help wording,
    where a single override applies to every probe's fitter.
    """
    p.add_argument(
        "--chi2-track",
        type=float,
        default=None,
        metavar="X",
        help=(
            "Telescope line-fit chi-squared threshold override, shared "
            "across every probe (default: PoseFitter's built-in 4.0)."
            if shared_across_probes
            else "Telescope line-fit chi-squared threshold override (default: "
            "PoseFitter's built-in 4.0)."
        ),
    )
    p.add_argument(
        "--max-cluster-width",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap on the per-axis merged-channel width a hit's centroid may be "
            "built from, shared across every probe; a wider hit is treated as "
            "unresolved. Off by default."
            if shared_across_probes
            else "Cap on the per-axis merged-channel width a hit's centroid may "
            "be built from; a wider hit is treated as unresolved. Off by "
            "default."
        ),
    )


def add_mahal_cut_arg(
    p: argparse.ArgumentParser, *, shared_across_probes: bool = False
) -> None:
    p.add_argument(
        "--mahal-cut",
        type=float,
        default=None,
        metavar="D",
        help=(
            "Mahalanobis outlier-cut distance for the probe pose fit"
            + (", shared across every probe" if shared_across_probes else "")
            + f" (default: fit_probe_pose's built-in {_MAHAL_CUT}). Independent "
            "of --chi2-track: that cut measures the telescope line fit, this one "
            "the probe-vs-track residual, so a coherent wrong-fold pick that "
            "lies on the fitted line is invisible to --chi2-track but cut here."
        ),
    )


def add_max_per_plane_arg(
    p: argparse.ArgumentParser, *, shared_across_probes: bool = False
) -> None:
    p.add_argument(
        "--max-per-plane",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap on the telescope candidate positions enumerated per plane "
            "before the combinatorial track search"
            + (", shared across every probe" if shared_across_probes else "")
            + f" (default: {MAX_PER_PLANE_DEFAULT}). The default covers the "
            "single-particle mirror fold only; on real pile-up it binds and "
            "discards candidates, so raising it changes which triple wins at a "
            "cost of up to N^3 line fits per cluster."
        ),
    )


def add_coincidence_window_ns_arg(
    p: argparse.ArgumentParser, *, help_suffix: str = ""
) -> None:
    p.add_argument(
        "--coincidence-window-ns",
        type=int,
        default=None,
        metavar="NS",
        help=f"Stage-2 coincidence window in nanoseconds (default: "
        f"{WINDOW_NS_DEFAULT}). This is the hardware coincidence window, NOT "
        f"--window-s, which is the monitoring window a pose is fitted over."
        + help_suffix,
    )


def validate_mahal_cut(mahal_cut: float | None) -> None:
    """Raise ``ValueError`` if ``--mahal-cut`` is not > 0 (``None`` allowed)."""
    if mahal_cut is not None and not mahal_cut > 0:
        raise ValueError(f"--mahal-cut must be > 0; got {mahal_cut}")


def validate_max_per_plane(max_per_plane: int | None) -> None:
    """Raise ``ValueError`` if ``--max-per-plane`` is not >= 1 (``None`` allowed)."""
    if max_per_plane is not None and max_per_plane < 1:
        raise ValueError(f"--max-per-plane must be >= 1; got {max_per_plane}")


def validate_coincidence_window_ns(window_ns: int | None) -> None:
    """Raise ``ValueError`` if ``--coincidence-window-ns`` is not > 0 (``None`` ok)."""
    if window_ns is not None and window_ns <= 0:
        raise ValueError(f"--coincidence-window-ns must be > 0; got {window_ns}")


def validate_chi2_track_args(args: argparse.Namespace) -> None:
    """Raise ``ValueError`` if ``--chi2-track``/``--max-cluster-width`` are out of range.

    Mirrors :func:`validate_probe_footprint`'s contract: a pure check the CLI
    boundary wraps in ``try/except ValueError -> parser.error``.
    """
    if args.chi2_track is not None and not args.chi2_track > 0:
        raise ValueError(f"--chi2-track must be > 0; got {args.chi2_track}")
    if args.max_cluster_width is not None and args.max_cluster_width < 1:
        raise ValueError(
            f"--max-cluster-width must be >= 1; got {args.max_cluster_width}"
        )


def validate_fibers_per_ribbon(values: list[int]) -> None:
    """Raise ``ValueError`` if any ``--fibers-per-ribbon`` value is outside 1..10.

    A probe can wire at most the 10 raw fiber positions per ribbon channel
    (DESIGN.md section 2.4). Dedups the identical scalar check in
    run_pipeline/timeseries and the per-value loop in multiprobe -- pass
    ``[value]`` for a scalar caller.
    """
    for n in values:
        if not 1 <= n <= 10:
            raise ValueError(
                f"--fibers-per-ribbon must be in 1..10 (a probe can wire at "
                f"most the 10 raw fiber positions); got {n}"
            )


def validate_min_fit(min_fit: int) -> None:
    """Raise ``ValueError`` if ``--min-fit`` is below ``fit_probe_pose``'s floor."""
    if min_fit < _MIN_COINCS:
        raise ValueError(
            f"--min-fit must be >= {_MIN_COINCS} (fit_probe_pose's hard "
            f"minimum); got {min_fit}"
        )
