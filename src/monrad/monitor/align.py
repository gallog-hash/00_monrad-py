"""Daily telescope alignment calibration + hardware-drift monitor.

Console script ``monrad-align``.  The telescope stack is rigidly mounted, so
its internal alignment (DESIGN.md §7) is a stable, once-a-day calibration
rather than something worth refitting on every monitoring run.  This tool:

1. picks the **first ``--n-files`` telescope file pairs of one day** (the
   earliest day in the directory, or ``--date YYYYMMDD``),
2. fits **one** :class:`~monrad.alignment.AlignmentCorrection` over *all*
   events in those files (a true whole-subset fit -- unlike the in-run
   ``fit_alignment`` path, which returns only a trailing chunk), and
3. writes it to ``alignment_<date>.json`` for the monitoring drivers to load
   with ``--alignment`` instead of refitting.

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
from collections import Counter, OrderedDict
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..alignment import (
    AlignmentAccumulator,
    AlignmentCorrection,
    PlaneCorrection,
    save_alignment,
)

# Mechanical significance thresholds fit_telescope_alignment uses to set
# needs_correction; imported so the drift plot and the per-parameter warning
# report against the same limits the fit flags on.
from ..alignment.accumulator import (
    _OFFSET_THRESH,
    _ROTATION_THRESH,
    _TILT_THRESH,
    _Z_THRESH,
)
from ..reconstruction import decode_position
from ..timing import reconstruct_stream
from .io import DetectorFiles, MacroArgumentParser, load_detector

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


def group_by_day(det: DetectorFiles) -> "OrderedDict[str, list[tuple[Path, Path]]]":
    """Group a detector's file pairs by ``YYYYMMDD`` day prefix, day-ascending.

    Files are named ``YYYYMMDD_HHMMSS[_GPS].bin``.  The returned mapping is
    ordered by day, and each day's list is ordered by filename (=
    acquisition time), so ``select_day_files`` can take the earliest day and
    its first-N files by iteration order regardless of the input order.
    """
    grouped: dict[str, list[tuple[Path, Path]]] = {}
    for gps, pos in zip(det.gps_paths, det.pos_paths):
        grouped.setdefault(gps.name[:8], []).append((gps, pos))
    days: "OrderedDict[str, list[tuple[Path, Path]]]" = OrderedDict()
    for day in sorted(grouped):
        days[day] = sorted(grouped[day], key=lambda pair: pair[0].name)
    return days


def select_day_files(
    det: DetectorFiles, date: str | None, n_files: int
) -> tuple[str, list[Path], list[Path]]:
    """Choose a day and return its first ``n_files`` (gps, pos) paths, split.

    ``date`` selects the day (``YYYYMMDD``); ``None`` picks the earliest day
    present.  Raises ``ValueError`` if the requested day is absent.
    """
    days = group_by_day(det)
    if not days:
        raise ValueError("no dated file pairs found")
    if date is None:
        day = next(iter(days))
    elif date in days:
        day = date
    else:
        raise ValueError(
            f"no files for date {date!r}; available days: {', '.join(days)}"
        )
    pairs = days[day][:n_files]
    gps = [p[0] for p in pairs]
    pos = [p[1] for p in pairs]
    return day, gps, pos


def fit_daily_alignment(
    gps_paths: list[Path],
    pos_paths: list[Path],
    utc0: datetime,
    f0: int,
    z_tel: np.ndarray,
    *,
    tot_thresh: int = 1,
    tot_weights: bool = False,
) -> tuple[AlignmentCorrection, int, dict[str, int]]:
    """Fit one alignment correction over *all* events in the given files.

    Mirrors :func:`monrad.monitor.io.fit_alignment`'s stage-1→stage-4 wiring,
    but drives the :class:`AlignmentAccumulator` with an effectively infinite
    ``flush_every`` so ``add`` never mid-flushes: the single trailing
    ``flush`` then fits over every buffered event, not just the last chunk.

    Returns ``(correction, n_events, quality_by_name)``.
    """
    accum = AlignmentAccumulator(flush_every=1 << 62, z_tel=z_tel)
    quality: Counter = Counter()
    for ev, ref in reconstruct_stream(gps_paths, pos_paths, utc0, f0):
        quality[ev.quality.name] += 1
        hits = decode_position(
            ref, pos_paths, n_cols=3, tot_thresh=tot_thresh, tot_weights=tot_weights
        )
        accum.add(hits)
    n_events = accum.n_buffered  # read before flush() clears the buffer
    correction = accum.flush()
    quality_by_name = {name: quality.get(name, 0) for name in _QUALITY_NAMES}
    return correction, n_events, quality_by_name


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
    day: str,
    correction: AlignmentCorrection,
    n_events: int,
    quality: dict[str, int],
) -> dict[str, str]:
    row: dict[str, str] = {
        "date": day,
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


def _plot_history(rows: list[dict[str, str]], path: Path) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = [datetime.strptime(r["date"], "%Y%m%d") for r in rows]

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


def compute_daily_alignment(
    tel_dir: Path,
    z_tel: Sequence[float] | np.ndarray,
    *,
    date: str | None = None,
    n_files: int = 3,
    out_dir: Path | None = None,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    make_plots: bool = True,
) -> AlignmentCorrection:
    """Fit a day's telescope alignment and record it as a reusable artifact.

    Writes ``alignment_<date>.json`` (the reusable correction), appends the fit
    to ``alignment_history.csv`` (idempotent per date) and regenerates
    ``alignment_history.png`` under ``out_dir``.  Returns the correction.
    """
    tel_dir = Path(tel_dir)
    z_tel = np.asarray(z_tel, dtype=float)
    det = load_detector(tel_dir)
    day, gps, pos = select_day_files(det, date, n_files)

    logger.info(
        "Fitting alignment for day %s from %d file(s): %s",
        day,
        len(pos),
        ", ".join(p.name for p in pos),
    )
    correction, n_events, quality = fit_daily_alignment(
        gps,
        pos,
        det.utc0,
        det.f0,
        z_tel,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
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
            "HARDWARE DRIFT: day %s alignment exceeds mechanical limits — %s. "
            "Inspect the telescope stack; the fitted correction is still saved.",
            day,
            "; ".join(breaches),
        )
    else:
        logger.info("Alignment within mechanical limits (needs_correction=False).")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"alignment_{day}.json"
        save_alignment(
            correction,
            json_path,
            date=day,
            z_tel=z_tel,
            files=[p.name for p in pos],
            n_events=n_events,
            quality=quality,
        )
        rows = update_history(
            out_dir / "alignment_history.csv",
            _history_row(day, correction, n_events, quality),
        )
        if make_plots and rows:
            _plot_history(rows, out_dir / "alignment_history.png")
        print(
            f"Wrote {json_path} and updated alignment_history.csv ({len(rows)} day(s))"
        )

    return correction


# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = MacroArgumentParser(
        prog="monrad-align",
        description="Daily telescope alignment calibration + hardware-drift monitor.",
        epilog="Flags can be collected in a macro file and loaded with "
        "'@path/to/file.args' (one flag per line, '#' comments allowed); "
        "e.g. 'monrad-align @align.args --date 20230418'. Flags given on the "
        "command line after the @file override lines from the file.",
    )
    p.add_argument(
        "--telescope",
        type=Path,
        required=True,
        metavar="DIR",
        help="Telescope acquisition directory (may span many days).",
    )
    p.add_argument(
        "--z-tel",
        nargs="+",
        type=float,
        required=True,
        metavar="Z",
        help="Telescope plane z-positions (mm).  Recorded with the saved "
        "correction; monitoring runs must reuse it with the same --z-tel.",
    )
    p.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYYMMDD",
        help="Which day to calibrate.  Default: the earliest day present.",
    )
    p.add_argument(
        "--n-files",
        type=int,
        default=3,
        metavar="N",
        help="Number of that day's first file pairs to fit over (default: 3).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("./pipeline_out/alignment"),
        help="Output directory (default: ./pipeline_out/alignment).",
    )
    p.add_argument("--tot-thresh", type=int, default=1)
    p.add_argument("--tot-weights", action="store_true")
    p.add_argument("--no-plots", action="store_true", help="Skip matplotlib output.")
    return p


def _parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, set[str]]:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.n_files < 1:
        parser.error(f"--n-files must be >= 1; got {args.n_files}")

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
    logger.info("  n_files:             %s  %s", args.n_files, _tag("n_files"))
    logger.info("  tot_thresh:          %s  %s", args.tot_thresh, _tag("tot_thresh"))
    logger.info("  tot_weights:         %s  %s", args.tot_weights, _tag("tot_weights"))
    logger.info("  no_plots:            %s  %s", args.no_plots, _tag("no_plots"))

    correction = compute_daily_alignment(
        args.telescope,
        np.array(args.z_tel),
        date=args.date,
        n_files=args.n_files,
        out_dir=args.out,
        tot_thresh=args.tot_thresh,
        tot_weights=args.tot_weights,
        make_plots=not args.no_plots,
    )
    print(
        f"needs_correction={correction.needs_correction}"
        + (" (see WARNING above)" if correction.needs_correction else "")
    )


if __name__ == "__main__":
    main()
