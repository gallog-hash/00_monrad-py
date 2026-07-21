"""Telescope alignment calibration + hardware-drift monitor.

Console script ``monrad-align``.  The telescope stack is rigidly mounted, so
its internal alignment (DESIGN.md §7) is a stable calibration refit on a
fixed cadence (one day by default) rather than something worth refitting on
every monitoring run.  By default this tool processes the **entire dataset**
in the given telescope directory:

1. splits it into fixed-length, midnight-anchored windows of
   ``--interval-hours`` (default 24 -- one per calendar day; pass e.g. ``6``
   for four refits a day),
2. for each window, fits **one** :class:`~monrad.alignment.AlignmentCorrection`
   over *all* events in that window's first ``--n-files`` file pairs (a true
   whole-subset fit), and
3. writes it to ``alignment_<label>.json`` for the monitoring drivers to load
   with ``--alignment`` instead of refitting.

Pass ``--date`` to restrict this to one day (or, under a sub-day
``--interval-hours``, one specific window).

Window selection (:func:`~monrad.monitor.io.select_alignment_windows`) and
the whole-subset fit (:func:`~monrad.monitor.io.fit_daily_alignment`) live in
:mod:`monrad.monitor.io`.  The single-day building block
(:func:`~monrad.monitor.io.select_day_files`) is reused, unchanged, by the
monitoring drivers' no-``--alignment`` fallback (``monrad.monitor.io.fit_alignment``),
so a monitoring run's auto-fit alignment is identical to a precomputed
``monrad-align`` correction for the same telescope directory (issue #18).

The fit doubles as a **hardware-drift monitor**: ``fit_telescope_alignment``
already raises ``needs_correction`` when any plane's offset / rotation / z /
tilt exceeds a mechanical threshold.  Each day's per-plane parameters are
appended to ``alignment_history.csv`` and plotted against date in
``alignment_history.png`` (with the thresholds drawn in), so telescope drift
is visible over time.  ``needs_correction`` is logged loudly but the tool
still exits 0 -- it is a monitor, not a gate.
"""

import argparse
import csv
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from ..alignment import AlignmentCorrection, PlaneCorrection, save_alignment
from ..timing import _utc_to_ns

# Mechanical significance thresholds fit_telescope_alignment uses to set
# needs_correction; imported so the drift plot and the per-parameter warning
# report against the same limits the fit flags on.
from ..alignment.accumulator import (
    _OFFSET_THRESH,
    _ROTATION_THRESH,
    _TILT_THRESH,
    _Z_THRESH,
)
from .cli_args import (
    MacroArgumentParser,
    add_no_plots_arg,
    add_out_arg,
    add_telescope_arg,
    add_tot_thresh_arg,
    add_tot_weights_arg,
    add_z_tel_arg,
)
from .io import (
    DAILY_ALIGNMENT_N_FILES,
    DetectorFiles,
    _parse_file_ts,
    _parse_window_label,
    fit_daily_alignment,
    group_by_day,
    load_detector,
    select_alignment_windows,
    select_day_files,
)

# Re-exported: group_by_day/select_day_files/select_alignment_windows/
# fit_daily_alignment now live in .io (shared with the in-run auto-fit
# fallback, monrad.monitor.io.fit_alignment -- issue #18), but stay
# importable from here since this module is where callers/tests have always
# found them.
__all__ = [
    "compute_alignment",
    "compute_daily_alignment",
    "fit_daily_alignment",
    "group_by_day",
    "select_alignment_windows",
    "select_day_files",
]

logger = logging.getLogger(__name__)

# The six PlaneCorrection fields in NamedTuple order — column source of truth.
_PLANE_FIELDS = PlaneCorrection._fields
_N_PLANES = 3
_QUALITY_NAMES = ("GOOD", "DEGRADED", "UNTRUSTED")

# Per-field threshold used for both the warning and the plot reference lines.
# Offsets (delta_x/delta_y) and tilts share one limit each; rotation and z
# have their own.
_FIELD_THRESH = {
    "delta_x": _OFFSET_THRESH,
    "delta_y": _OFFSET_THRESH,
    "rotation_z": _ROTATION_THRESH,
    "delta_z": _Z_THRESH,
    "tilt_x": _TILT_THRESH,
    "tilt_y": _TILT_THRESH,
}


def _threshold_breaches(correction: AlignmentCorrection) -> list[str]:
    """Human-readable list of every plane parameter over its mechanical limit.

    The same conditions ``fit_telescope_alignment`` uses to set
    ``needs_correction``, but attributed to the specific plane/parameter so the
    warning names what drifted.
    """
    msgs: list[str] = []
    for k, plane in enumerate(correction.planes):
        for field, thresh in _FIELD_THRESH.items():
            val = getattr(plane, field)
            if abs(val) > thresh:
                msgs.append(f"plane {k} {field}={val:+.4g} exceeds |{thresh:g}|")
    return msgs


# ── history CSV (the drift log) ─────────────────────────────────────────────


def _history_columns() -> list[str]:
    cols = ["date", "computed_utc", "n_events", "needs_correction"]
    for k in range(_N_PLANES):
        cols += [f"p{k}_{f}" for f in _PLANE_FIELDS]
    cols += list(_QUALITY_NAMES)
    return cols


def _history_row(
    label: str,
    correction: AlignmentCorrection,
    n_events: int,
    quality: dict[str, int],
) -> dict[str, str]:
    row: dict[str, str] = {
        "date": label,
        "computed_utc": datetime.now(timezone.utc).isoformat(),
        "n_events": str(n_events),
        "needs_correction": str(correction.needs_correction),
    }
    for k, plane in enumerate(correction.planes):
        for f in _PLANE_FIELDS:
            row[f"p{k}_{f}"] = f"{getattr(plane, f):.6g}"
    for name in _QUALITY_NAMES:
        row[name] = str(quality.get(name, 0))
    return row


def update_history(path: Path, row: dict[str, str]) -> list[dict[str, str]]:
    """Append *row* to the history CSV, replacing any existing row for its date.

    Re-running a day must not duplicate it, so an existing row with the same
    ``date`` is dropped and replaced; the file is rewritten sorted by date.
    Returns the full row list (for plotting).
    """
    path = Path(path)
    cols = _history_columns()
    rows: list[dict[str, str]] = []
    if path.exists():
        with open(path, newline="") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("date") != row["date"]]
    rows.append(row)
    rows.sort(key=lambda r: r["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            # Tolerate older rows missing newly added columns.
            w.writerow({c: r.get(c, "") for c in cols})
    return rows


def _parse_history_label(label: str) -> datetime:
    """Parse a history row's ``date`` column: ``YYYYMMDD`` or ``YYYYMMDD_HHMMSS``.

    The plain day form is used for whole-day windows (the default
    ``--interval-hours 24``); sub-day windows carry the full start timestamp
    (see ``select_alignment_windows``/``_window_label`` in ``.io``).
    """
    try:
        return datetime.strptime(label, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.strptime(label, "%Y%m%d")


def _plot_history(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = [_parse_history_label(r["date"]) for r in rows]

    def series(field: str, plane: int) -> np.ndarray:
        return np.array(
            [float(r.get(f"p{plane}_{field}", "nan") or "nan") for r in rows]
        )

    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Panel 1 — per-plane translational offset magnitude.
    for k in range(_N_PLANES):
        mag = np.hypot(series("delta_x", k), series("delta_y", k))
        axs[0].plot(dates, mag, "o-", ms=4, label=f"plane {k}")
    axs[0].axhline(_OFFSET_THRESH, color="r", ls="--", lw=1, alpha=0.6, label="limit")
    axs[0].set_ylabel("|offset| [mm]")

    # Panel 2 — per-plane rotation about z.
    for k in range(_N_PLANES):
        axs[1].plot(dates, series("rotation_z", k), "o-", ms=4, label=f"plane {k}")
    for sign in (+1, -1):
        axs[1].axhline(sign * _ROTATION_THRESH, color="r", ls="--", lw=1, alpha=0.6)
    axs[1].set_ylabel("rotation_z [rad]")

    # Panel 3 — middle-plane Z offset (only the mid plane is non-zero).
    for k in range(_N_PLANES):
        axs[2].plot(dates, series("delta_z", k), "o-", ms=4, label=f"plane {k}")
    for sign in (+1, -1):
        axs[2].axhline(sign * _Z_THRESH, color="r", ls="--", lw=1, alpha=0.6)
    axs[2].set_ylabel("delta_z [mm]")

    # Panel 4 — middle-plane out-of-plane tilts.
    for k in range(_N_PLANES):
        axs[3].plot(dates, series("tilt_x", k), "o-", ms=4, label=f"tilt_x p{k}")
        axs[3].plot(dates, series("tilt_y", k), "s--", ms=4, label=f"tilt_y p{k}")
    for sign in (+1, -1):
        axs[3].axhline(sign * _TILT_THRESH, color="r", ls="--", lw=1, alpha=0.6)
    axs[3].set_ylabel("tilt [rad]")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    axs[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    axs[0].set_title("Telescope alignment drift (dashed = mechanical limit)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── orchestration ───────────────────────────────────────────────────────────


def _daq_utc_offset(det: DetectorFiles) -> timedelta:
    """The constant DAQ-file-name-clock minus GPS-UTC offset for an acquisition.

    Window labels (and file names) are in whatever clock the DAQ names files
    with -- often local time -- while events are stamped in GPS-UTC
    (``TimedEvent.t_ns``).  Measured once from the earliest file:
    ``file-name time - utc0`` (both for the acquisition start), it maps any
    file-name/window-label time to the UTC of the event a file so named would
    carry.  A window's own files cannot give this per-window (they are
    reconstructed against the acquisition-start ``utc0``, so only the earliest
    window's ``t_ns`` is absolute), so the offset is taken globally and applied
    to each window label -- see :func:`_window_utc_start_ns`.
    """
    utc0 = det.utc0 if det.utc0.tzinfo is None else det.utc0.replace(tzinfo=None)
    return _parse_file_ts(det.gps_paths[0].name) - utc0


def _window_utc_start_ns(label: str, offset: timedelta) -> int:
    """A window label's true UTC start (integer ns), shifting out the DAQ offset."""
    return _utc_to_ns(_parse_window_label(label) - offset)


def _fit_window(
    det: DetectorFiles,
    label: str,
    gps: list[Path],
    pos: list[Path],
    z_tel: np.ndarray,
    **fit_kwargs,
) -> tuple[AlignmentCorrection, int, dict[str, int]]:
    """Fit + log one window's alignment.  Shared by every caller below."""
    logger.info(
        "Fitting alignment for %s from %d file(s): %s",
        label,
        len(pos),
        ", ".join(p.name for p in pos),
    )
    correction, n_events, quality = fit_daily_alignment(
        gps, pos, det.utc0, det.f0, z_tel, **fit_kwargs
    )
    logger.info(
        "Fit over %d alignment-usable event(s) (golden/cluster hits); "
        "timing quality over all streamed events: %s",
        n_events,
        "  ".join(f"{k}={quality[k]}" for k in _QUALITY_NAMES),
    )
    for k, plane in enumerate(correction.planes):
        logger.info(
            "  plane %d: dx=%+.3f dy=%+.3f mm  rot_z=%+.2e rad  "
            "dz=%+.3f mm  tilt_x=%+.2e tilt_y=%+.2e rad",
            k,
            plane.delta_x,
            plane.delta_y,
            plane.rotation_z,
            plane.delta_z,
            plane.tilt_x,
            plane.tilt_y,
        )
    if correction.needs_correction:
        breaches = _threshold_breaches(correction)
        logger.warning(
            "HARDWARE DRIFT: %s alignment exceeds mechanical limits — %s. "
            "Inspect the telescope stack; the fitted correction is still saved.",
            label,
            "; ".join(breaches),
        )
    else:
        logger.info("Alignment within mechanical limits (needs_correction=False).")
    return correction, n_events, quality


def _write_window_artifact(
    out_dir: Path,
    label: str,
    correction: AlignmentCorrection,
    pos: list[Path],
    z_tel: np.ndarray,
    n_events: int,
    quality: dict[str, int],
    *,
    utc_start_ns: int | None = None,
    interval_hours: float = 24.0,
) -> list[dict[str, str]]:
    """Write ``alignment_<label>.json`` and append the drift-history row.

    Returns the full (cumulative) history row list, so a multi-window caller
    can defer plotting until after its last window and still plot everything.

    ``utc_start_ns`` (the window's true UTC start, from the first fitted event)
    and ``interval_hours`` are recorded as the window's real-clock UTC bounds
    (``utc_start_ns``/``utc_end_ns = start + interval``) so a time-varying
    consumer keys on UTC, not the file-name ``label`` (which is DAQ-local).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"alignment_{label}.json"
    utc_end_ns = (
        None
        if utc_start_ns is None
        else utc_start_ns + int(interval_hours * 3600 * 1_000_000_000)
    )
    save_alignment(
        correction,
        json_path,
        date=label,
        z_tel=z_tel,
        files=[p.name for p in pos],
        n_events=n_events,
        quality=quality,
        utc_start_ns=utc_start_ns,
        utc_end_ns=utc_end_ns,
    )
    rows = update_history(
        out_dir / "alignment_history.csv",
        _history_row(label, correction, n_events, quality),
    )
    print(f"Wrote {json_path} and updated alignment_history.csv ({len(rows)} row(s))")
    return rows


def compute_daily_alignment(
    tel_dir: Path,
    z_tel: Sequence[float] | np.ndarray,
    *,
    date: str | None = None,
    n_files: int = DAILY_ALIGNMENT_N_FILES,
    out_dir: Path | None = None,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    make_plots: bool = True,
) -> AlignmentCorrection:
    """Fit a single day's telescope alignment and record it as a reusable artifact.

    Writes ``alignment_<date>.json`` (the reusable correction), appends the fit
    to ``alignment_history.csv`` (idempotent per date) and regenerates
    ``alignment_history.png`` under ``out_dir``.  Returns the correction.

    This is the single-day building block; :func:`compute_alignment` (what
    ``monrad-align`` calls by default) loops it -- as one window per
    ``--interval-hours`` -- over the whole telescope directory.
    """
    tel_dir = Path(tel_dir)
    z_tel = np.asarray(z_tel, dtype=float)
    det = load_detector(tel_dir)
    day, gps, pos = select_day_files(det, date, n_files)

    correction, n_events, quality = _fit_window(
        det,
        f"day {day}",
        gps,
        pos,
        z_tel,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
    )

    if out_dir is not None:
        rows = _write_window_artifact(
            out_dir,
            day,
            correction,
            pos,
            z_tel,
            n_events,
            quality,
            utc_start_ns=_window_utc_start_ns(day, _daq_utc_offset(det)),
            interval_hours=24.0,
        )
        if make_plots and rows:
            _plot_history(rows, Path(out_dir) / "alignment_history.png")

    return correction


def compute_alignment(
    tel_dir: Path,
    z_tel: Sequence[float] | np.ndarray,
    *,
    date: str | None = None,
    interval_hours: float = 24.0,
    n_files: int = DAILY_ALIGNMENT_N_FILES,
    out_dir: Path | None = None,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    make_plots: bool = True,
) -> list[AlignmentCorrection]:
    """Fit telescope alignment over every ``interval_hours`` window in the dataset.

    This is what ``monrad-align`` runs by default: with ``date=None`` it
    processes the *entire* telescope directory, one whole-subset alignment
    fit per fixed-length, midnight-anchored window (see
    :func:`~monrad.monitor.io.group_by_interval`) of ``interval_hours``
    (default 24 -- one refit per calendar day).  Pass ``date`` to restrict
    this to one day, or -- under a sub-day ``interval_hours`` -- one specific
    window (see :func:`~monrad.monitor.io.select_alignment_windows`).

    Writes one ``alignment_<label>.json`` per window and appends each to
    ``alignment_history.csv``, then regenerates ``alignment_history.png``
    once after the whole run (not once per window).  Returns the list of
    fitted corrections, oldest window first.
    """
    tel_dir = Path(tel_dir)
    z_tel = np.asarray(z_tel, dtype=float)
    det = load_detector(tel_dir)
    windows = select_alignment_windows(det, interval_hours, n_files, date=date)
    offset = _daq_utc_offset(det)

    corrections: list[AlignmentCorrection] = []
    rows: list[dict[str, str]] = []
    for i, (label, gps, pos) in enumerate(windows):
        if i > 0:
            logger.info("")  # blank line between windows' log blocks
        correction, n_events, quality = _fit_window(
            det,
            f"window {label}",
            gps,
            pos,
            z_tel,
            tot_thresh=tot_thresh,
            tot_weights=tot_weights,
        )
        corrections.append(correction)
        if out_dir is not None:
            rows = _write_window_artifact(
                out_dir,
                label,
                correction,
                pos,
                z_tel,
                n_events,
                quality,
                utc_start_ns=_window_utc_start_ns(label, offset),
                interval_hours=interval_hours,
            )

    if out_dir is not None and make_plots and rows:
        _plot_history(rows, Path(out_dir) / "alignment_history.png")

    return corrections


# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = MacroArgumentParser(
        prog="monrad-align",
        description="Telescope alignment calibration + hardware-drift monitor. "
        "By default processes the entire dataset in --telescope, one refit "
        "per --interval-hours window.",
        epilog="Flags can be collected in a macro file and loaded with "
        "'@path/to/file.args' (one flag per line, '#' comments allowed); "
        "e.g. 'monrad-align @align.args --date 20230418'. Flags given on the "
        "command line after the @file override lines from the file.",
    )
    add_telescope_arg(p, help_suffix=" (may span many days).")
    add_z_tel_arg(
        p,
        help_suffix=(
            "  Recorded with the saved correction; monitoring runs must "
            "reuse it with the same --z-tel."
        ),
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYYMMDD",
        help="Restrict calibration to one day (or, under a sub-day "
        "--interval-hours, a full YYYYMMDD_HHMMSS to restrict to one "
        "window).  Default: process every window across the whole "
        "--telescope directory.",
    )
    p.add_argument(
        "--interval-hours",
        type=float,
        default=24.0,
        metavar="HOURS",
        help="Length of each alignment-refit window, in hours (default: 24 "
        "-- one refit per calendar day). E.g. 6 for four refits a day. "
        "Windows are anchored to 00:00 of the earliest day present.",
    )
    p.add_argument(
        "--n-files",
        type=int,
        default=DAILY_ALIGNMENT_N_FILES,
        metavar="N",
        help=f"Number of each window's first file pairs to fit over "
        f"(default: {DAILY_ALIGNMENT_N_FILES}).",
    )
    add_out_arg(p, default=Path("./pipeline_out/alignment"))
    add_tot_thresh_arg(p)
    add_tot_weights_arg(p)
    add_no_plots_arg(p)
    return p


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, set[str]]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.n_files < 1:
        parser.error(f"--n-files must be >= 1; got {args.n_files}")
    if args.interval_hours <= 0:
        parser.error(f"--interval-hours must be > 0; got {args.interval_hours}")

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
    logger.info("  Output dir:          %s  %s", args.out, _tag("out"))
    logger.info(
        "  Telescope plane z (mm): %s  %s",
        "  ".join(f"{z:g}" for z in args.z_tel),
        _tag("z_tel"),
    )
    logger.info("  date:                %s  %s", args.date, _tag("date"))
    logger.info(
        "  interval_hours:      %s  %s", args.interval_hours, _tag("interval_hours")
    )
    logger.info("  n_files:             %s  %s", args.n_files, _tag("n_files"))
    logger.info("  tot_thresh:          %s  %s", args.tot_thresh, _tag("tot_thresh"))
    logger.info("  tot_weights:         %s  %s", args.tot_weights, _tag("tot_weights"))
    logger.info("  no_plots:            %s  %s", args.no_plots, _tag("no_plots"))
    logger.info("")

    corrections = compute_alignment(
        args.telescope,
        np.array(args.z_tel),
        date=args.date,
        interval_hours=args.interval_hours,
        n_files=args.n_files,
        out_dir=args.out,
        tot_thresh=args.tot_thresh,
        tot_weights=args.tot_weights,
        make_plots=not args.no_plots,
    )
    n_flagged = sum(c.needs_correction for c in corrections)
    logger.info("")
    print(
        f"Processed {len(corrections)} window(s); {n_flagged} flagged "
        "needs_correction=True" + (" (see WARNING(s) above)" if n_flagged else "")
    )


if __name__ == "__main__":
    main()
