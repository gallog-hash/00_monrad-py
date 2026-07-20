"""Shared detector-setup helpers for the monitoring drivers.

The pose-fitting drivers (``resolution``, ``timeseries``, ``multiprobe``) all
need the same two pieces of setup that ``scripts/run_pipeline.py`` performs by
hand: locate a detector's file pairs + header, and run the telescope alignment
pass.  Extracting them here keeps the drivers (and the script) from each
carrying their own copy.
"""

import argparse
import json
import math
import shlex
from collections import Counter, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..alignment import AlignmentAccumulator, AlignmentCorrection, load_alignment
from ..coincidence import coincidence_stream
from ..decoders.position import POS_HALF_BITS
from ..pose import Coincidence, PoseFitter
from ..reconstruction import decode_position
from ..synthetic.generate import STRIP_MM
from ..timing import (
    PosRef,
    TimedEvent,
    _utc_to_ns,
    find_file_pairs,
    load_header_params,
    reconstruct_stream,
)


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


# monrad-align's own default: the earliest day's first 3 file pairs (see
# monrad.monitor.align).  Shared here so fit_alignment's no-``--alignment``
# fallback picks the identical subset -- keeping the two paths in lockstep is
# the whole point of issue #18; a second hardcoded "3" would just reopen it.
DAILY_ALIGNMENT_N_FILES = 3


class DetectorFiles(NamedTuple):
    """One detector's decoded header + matched ``*_GPS.bin`` / ``*.bin`` pairs."""

    utc0: datetime
    f0: int
    gps_paths: list[Path]
    pos_paths: list[Path]


def load_detector(d: Path) -> DetectorFiles:
    """Locate a detector directory's header and ``*_GPS.bin`` / ``*.bin`` pairs.

    Wraps :func:`monrad.timing.load_header_params` and
    :func:`monrad.timing.find_file_pairs`.  Raises ``FileNotFoundError`` when
    the directory carries no ``*_header*.txt`` or no matching file pairs — the
    library counterpart of the ``sys.exit`` guards in ``run_pipeline.py``.
    """
    d = Path(d)
    headers = list(d.glob("*_header*.txt"))
    if not headers:
        raise FileNotFoundError(f"no *_header.txt found in {d}")
    utc0, f0 = load_header_params(headers[0])
    gps_paths, pos_paths = find_file_pairs(d)
    if not gps_paths:
        raise FileNotFoundError(f"no *_GPS.bin / *.bin pairs found in {d}")
    return DetectorFiles(utc0, f0, gps_paths, pos_paths)


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


def _parse_file_ts(name: str) -> datetime:
    """Parse the ``YYYYMMDD_HHMMSS`` acquisition timestamp off a file name."""
    return datetime.strptime(name[:15], "%Y%m%d_%H%M%S")


def _window_label(start: datetime, interval_hours: float) -> str:
    """Format a window's start as a file-name / CSV-safe label.

    A window that starts at midnight and spans whole calendar day(s)
    (``interval_hours`` a multiple of 24) gets the plain ``YYYYMMDD`` label
    :func:`group_by_day` already uses -- so the default ``interval_hours=24``
    reproduces the existing ``alignment_<date>.json`` naming exactly.  Any
    finer window gets the full ``YYYYMMDD_HHMMSS`` start timestamp (the same
    style acquisition files already use) so same-day windows stay distinct.
    """
    if (
        interval_hours >= 24
        and interval_hours % 24 == 0
        and start.hour == 0
        and start.minute == 0
        and start.second == 0
    ):
        return start.strftime("%Y%m%d")
    return start.strftime("%Y%m%d_%H%M%S")


def _parse_window_label(label: str) -> datetime:
    """Inverse of :func:`_window_label`: a window label back to its start time.

    Accepts both forms :func:`_window_label` emits -- the whole-day ``YYYYMMDD``
    and the sub-day ``YYYYMMDD_HHMMSS`` -- returning a naive UTC datetime on the
    same clock the header ``timeR`` seeds ``utc0`` with. Mirrors
    :func:`monrad.monitor.align._parse_history_label`, kept here so the schedule
    loader (and any ``.io`` caller) need not import from ``align``.
    """
    try:
        return datetime.strptime(label, "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.strptime(label, "%Y%m%d")


def group_by_interval(
    det: DetectorFiles, interval_hours: float
) -> "OrderedDict[str, list[tuple[Path, Path]]]":
    """Group a detector's file pairs into fixed-length, midnight-anchored windows.

    Generalizes :func:`group_by_day` (equivalent to ``interval_hours=24``) to
    an arbitrary refit cadence, e.g. ``interval_hours=6`` for four alignment
    windows a day. Windows are anchored to 00:00 of the earliest day present,
    so an integer number of windows tiles each calendar day whenever ``24 %
    interval_hours == 0``. Returned mapping is ordered by window start; each
    window's file list is ordered by filename (= acquisition time).
    """
    pairs = sorted(zip(det.gps_paths, det.pos_paths), key=lambda pair: pair[0].name)
    grouped: dict[datetime, list[tuple[Path, Path]]] = {}
    if pairs:
        interval = timedelta(hours=interval_hours)
        origin = _parse_file_ts(pairs[0][0].name).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        for gps, pos in pairs:
            idx = (_parse_file_ts(gps.name) - origin) // interval
            start = origin + idx * interval
            grouped.setdefault(start, []).append((gps, pos))
    windows: "OrderedDict[str, list[tuple[Path, Path]]]" = OrderedDict()
    for start in sorted(grouped):
        windows[_window_label(start, interval_hours)] = grouped[start]
    return windows


def select_alignment_windows(
    det: DetectorFiles,
    interval_hours: float,
    n_files: int,
    date: str | None = None,
) -> list[tuple[str, list[Path], list[Path]]]:
    """Pick each refit window's first ``n_files`` (gps, pos) pairs.

    ``date=None`` (the default) selects *every* window across the whole
    telescope directory -- one alignment refit per ``interval_hours``,
    spanning the full acquisition. Pass an exact window label
    (``YYYYMMDD`` for a default day-long window, or ``YYYYMMDD_HHMMSS`` for a
    specific finer one) to refit just that one window, or a bare ``YYYYMMDD``
    day prefix under a sub-day ``interval_hours`` to refit every window
    falling on that day. Raises ``ValueError`` if nothing matches.
    """
    windows = group_by_interval(det, interval_hours)
    if not windows:
        raise ValueError("no dated file pairs found")
    if date is None:
        labels = list(windows)
    elif date in windows:
        labels = [date]
    else:
        labels = [lbl for lbl in windows if lbl.startswith(date)]
        if not labels:
            raise ValueError(
                f"no files for date {date!r}; available windows: {', '.join(windows)}"
            )
    return [
        (
            lbl,
            [g for g, _ in windows[lbl][:n_files]],
            [p for _, p in windows[lbl][:n_files]],
        )
        for lbl in labels
    ]


@dataclass
class AlignmentSchedule:
    """A time-ordered step function ``window_start_ns -> AlignmentCorrection``.

    Built from a directory of ``alignment_<label>.json`` files (one per
    ``monrad-align --interval-hours`` window) so the monitoring drivers can
    switch the active telescope correction as the coincidence stream crosses
    window boundaries, instead of applying one static correction for the whole
    run. :meth:`at` returns the correction whose window *starts* at or before a
    given telescope-event time; times before the first window clamp to the
    earliest correction rather than dropping events (the monitor windows are
    independent of the alignment windows). See ``load_alignment_schedule``.
    """

    starts_ns: np.ndarray  # int64, ascending
    corrections: list[AlignmentCorrection]
    labels: list[str]

    def at(self, t_ns: int) -> AlignmentCorrection:
        """The active correction for a telescope-event time ``t_ns``."""
        idx = int(np.searchsorted(self.starts_ns, t_ns, side="right")) - 1
        return self.corrections[max(idx, 0)]

    def label_at(self, t_ns: int) -> str:
        """The active window's label (the ``alignment_<label>.json`` stem)
        for a telescope-event time ``t_ns``.  Mirrors :meth:`at`."""
        idx = int(np.searchsorted(self.starts_ns, t_ns, side="right")) - 1
        return self.labels[max(idx, 0)]


def static_alignment_label(alignment_path: Path | None) -> str:
    """Provenance label for a fixed (non-schedule) alignment.

    Used for the monitoring CSV's ``alignment_label`` column when
    ``--alignment`` names a single file (or is omitted). Matches
    :class:`AlignmentSchedule`'s window-label convention (the file stem with
    the ``alignment_`` prefix stripped) so static and time-varying runs read
    consistently; ``"auto"`` when no ``--alignment`` was given (in-run fit).
    """
    if alignment_path is None:
        return "auto"
    return alignment_path.stem.removeprefix("alignment_")


def load_alignment_schedule(
    directory: Path, *, expect_z_tel: np.ndarray
) -> AlignmentSchedule:
    """Load every ``alignment_<label>.json`` in *directory* into a schedule.

    Each window's start time is taken from the file's ``utc_start_ns`` field --
    the true UTC (integer ns, same clock as ``Coincidence.t_ns``) of the
    window's first fitted event -- so :meth:`AlignmentSchedule.at` maps a
    coincidence to its window by real UTC time.  Files written before that
    field existed fall back to parsing the file-name ``date`` label as UTC (the
    original behavior); note this is only exact when the DAQ names files in UTC
    (see :func:`save_alignment`).  ``load_alignment``'s ``expect_z_tel`` check
    runs on *every* file: a single window fit against a different plane z-order
    aborts the whole run up front (the delta_z/tilt fit is z-order-dependent and
    cannot be mixed).
    """
    files = sorted(Path(directory).glob("alignment_*.json"))
    if not files:
        raise ValueError(f"{directory}: no alignment_*.json files found")
    rows: list[tuple[int, str, AlignmentCorrection]] = []
    for f in files:
        label = f.name[len("alignment_") : -len(".json")]
        payload = json.loads(f.read_text())
        raw_start = payload.get("utc_start_ns")
        start_ns = (
            int(raw_start)
            if raw_start is not None
            else _utc_to_ns(_parse_window_label(label))  # legacy pre-utc_start_ns
        )
        corr = load_alignment(f, expect_z_tel=expect_z_tel)
        rows.append((start_ns, label, corr))
    rows.sort(key=lambda r: r[0])
    return AlignmentSchedule(
        starts_ns=np.array([r[0] for r in rows], dtype=np.int64),
        corrections=[r[2] for r in rows],
        labels=[r[1] for r in rows],
    )


def _cluster_tel_time(
    cluster: list[tuple[int, TimedEvent, PosRef]], tel_id: int
) -> int | None:
    """The telescope event's integer-ns time from a coincidence cluster.

    Reads the same ``det_id == tel_id`` entry the decode path uses
    (``PoseFitter._decode_telescope_track``), *before* decoding, so an
    :class:`AlignmentSchedule` can pick the active correction for the cluster.
    Returns ``None`` if the cluster carries no (or more than one) telescope
    entry -- the decode will reject it anyway.
    """
    tel_times = [ev.t_ns for det_id, ev, _ in cluster if det_id == tel_id]
    if len(tel_times) != 1:
        return None
    return tel_times[0]


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

    Drives the :class:`AlignmentAccumulator` with an effectively infinite
    ``flush_every`` so ``add`` never mid-flushes: the single trailing
    ``flush`` then fits over every buffered event, not just the last chunk
    (unlike a small-``flush_every`` accumulator, whose disjoint intermediate
    fits are discarded by a caller that keeps only the final ``flush()``).

    Returns ``(correction, n_events, quality_by_name)``.  (Absolute event times
    are *not* returned here: a window's files are reconstructed against the
    acquisition-start ``utc0``, so ``t_ns`` is only correct for the earliest
    window -- the caller derives per-window UTC from a constant DAQ→UTC offset
    instead; see :func:`~monrad.monitor.align._daq_utc_offset`.)
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
    quality_by_name = {
        name: quality.get(name, 0) for name in ("GOOD", "DEGRADED", "UNTRUSTED")
    }
    return correction, n_events, quality_by_name


def validate_probe_footprint(n_probe_ch: int, fibers_per_ribbon: int) -> None:
    """Raise ``ValueError`` if ``n_probe_ch`` exceeds the channel range
    ``fibers_per_ribbon`` can actually address (``10 * fibers_per_ribbon``).

    A probe wired at combine factor N only has raw channels ``0..10*N-1``
    (DESIGN.md §2.4); anything above that aliases into an in-range-but-wrong
    channel during decode (``split_channel``/``decode_position``) instead of
    raising, silently biasing the fitted pose.  Catches the class of error
    where ``--n-probe-ch`` and ``--fibers-per-ribbon`` are inconsistent with
    each other; does not catch a wrong-but-plausible value for either.
    """
    if n_probe_ch > 10 * fibers_per_ribbon:
        raise ValueError(
            f"n_probe_ch={n_probe_ch} exceeds the maximum channel range "
            f"10 * fibers_per_ribbon={10 * fibers_per_ribbon}"
        )


def centre_jacobian(theta: float, n_probe_ch: int) -> np.ndarray:
    """The 2×4 Jacobian ``J = d(cx, cy)/d(t_x, t_y, θ, z_p)`` of the corner→centre map.

    The probe corner sits at ``(t_x, t_y)``; the centre at
    ``(t_x + half(cosθ − sinθ), t_y + half(sinθ + cosθ))``.  ``J`` has the
    leading 2×2 block equal to the identity, the θ column as the lever-arm
    derivatives, and the z_p column zero.  Single source for both the
    centre covariance (:func:`centre_cov_2x2`) and any centre/other-parameter
    cross-covariance (``J @ cov[:, k]``).
    """
    half = n_probe_ch * STRIP_MM / 2.0
    c, s = math.cos(theta), math.sin(theta)
    return np.array(
        [
            [1.0, 0.0, -half * (s + c), 0.0],
            [0.0, 1.0, half * (c - s), 0.0],
        ]
    )


def centre_cov_2x2(cov: np.ndarray, theta: float, n_probe_ch: int) -> np.ndarray:
    """Propagate a 4×4 pose covariance to the 2×2 probe-centre covariance.

    Uses :func:`centre_jacobian`.  Returns a 2×2 array: ``[0,0]`` = σ²_cx,
    ``[1,1]`` = σ²_cy.
    """
    J = centre_jacobian(theta, n_probe_ch)
    return J @ cov @ J.T


def fit_alignment(
    tel: DetectorFiles,
    z_tel: np.ndarray,
    *,
    tot_thresh: int = 1,
    tot_weights: bool = False,
) -> tuple[AlignmentCorrection, Counter]:
    """Auto-fit fallback for a monitoring run's telescope alignment.

    Used when the caller does not pass ``--alignment``.  Delegates to the
    same day-selection (:func:`select_day_files`, earliest day, first
    :data:`DAILY_ALIGNMENT_N_FILES` files) and whole-subset fit
    (:func:`fit_daily_alignment`) that ``monrad-align`` uses, so a
    monitoring run's auto-fit alignment is identical to a precomputed
    ``monrad-align`` correction for the same telescope directory.

    Previously this streamed *every* telescope event into an
    :class:`AlignmentAccumulator` and kept only its final ``flush()`` --
    since the accumulator re-fits into disjoint ``flush_every``-sized
    blocks, that silently fit over only the trailing ≤10k-event remainder
    rather than the whole acquisition (issue #18).  Returns the fitted
    correction together with a per-quality-name event histogram, so callers
    do not have to re-stream the telescope just to tally event quality.
    """
    _day, gps, pos = select_day_files(tel, date=None, n_files=DAILY_ALIGNMENT_N_FILES)
    correction, _n_events, quality = fit_daily_alignment(
        gps,
        pos,
        tel.utc0,
        tel.f0,
        z_tel,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
    )
    return correction, Counter(quality)


def build_cluster_stream(
    tel: DetectorFiles,
    probes: list[DetectorFiles],
    *,
    window_ns: int = 200,
) -> Iterator[list[tuple[int, TimedEvent, PosRef]]]:
    """Yield coincidence clusters over one telescope and N probes.

    One :func:`~monrad.timing.reconstruct_stream` per detector (telescope +
    each probe), merged by a single :func:`~monrad.coincidence.coincidence_stream`
    call.  The telescope is always ``det_id=0``; probe ``k`` (0-indexed in
    ``probes``) is ``det_id=k+1`` — the same convention
    :class:`~monrad.pose.PoseFitter`'s ``tel_id``/``prb_id`` already use, so a
    caller can build one ``PoseFitter`` per probe and call
    ``fitter.decode_cluster(cluster)`` on the same cluster for each: a cluster
    only yields a :class:`~monrad.pose.Coincidence` for the probe(s) it is
    actually consistent with (see ``PoseFitter._decode_cluster``'s multi-probe
    contract).  Shared by :mod:`~monrad.monitor.multiprobe`;
    ``window_ns`` mirrors :func:`~monrad.coincidence.coincidence_stream`'s own
    default (DESIGN.md §4/§5).
    """
    tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
    prb_streams = [
        reconstruct_stream(p.gps_paths, p.pos_paths, p.utc0, p.f0) for p in probes
    ]
    detector_ids = list(range(len(probes) + 1))
    return coincidence_stream(
        [tel_stream, *prb_streams], detector_ids=detector_ids, window_ns=window_ns
    )


def stream_coincidences(
    tel: DetectorFiles,
    prb: DetectorFiles,
    *,
    z_tel: np.ndarray,
    alignment: AlignmentCorrection,
    schedule: AlignmentSchedule | None = None,
    alignment_label: str = "",
    tot_thresh: int = 1,
    tot_weights: bool = False,
    min_anchor_planes: int = 1,
    fibers_per_ribbon: int = POS_HALF_BITS,
    chi2_track: float | None = None,
    max_cluster_width: int | None = None,
) -> Iterator[Coincidence]:
    """Yield decoded probe–telescope coincidences for one acquisition.

    Wires the stage-5 :class:`~monrad.pose.PoseFitter` decode path
    (combinatorial telescope track finder + alignment correction + track/probe
    quality cuts) to two stage-1 :func:`~monrad.timing.reconstruct_stream`
    generators and emits one :class:`~monrad.pose.Coincidence` per surviving
    cluster.  Shared by the ``resolution`` and ``timeseries`` drivers so the
    fitter wiring and the ``tel_id``/``prb_id`` convention live in one place.

    fibers_per_ribbon : the probe's fiber×ribbon combine factor (DESIGN.md
    §2.4), passed through as ``PoseFitter``'s ``prb_fibers_per_ribbon``.
    Telescope decode is unaffected.

    chi2_track, max_cluster_width : passed straight through to
    ``PoseFitter``; ``None`` keeps that fitter's own defaults (4.0 / off).

    schedule : optional :class:`AlignmentSchedule`. When given, the fitter's
    alignment is switched (via ``update_alignment``) before each cluster
    decodes, to the correction active at that cluster's telescope-event time --
    time-varying alignment. ``None`` (the default) keeps ``alignment`` fixed for
    the whole stream, exactly as before.

    alignment_label : the label recorded on each yielded ``Coincidence`` when
    ``schedule`` is ``None`` (a fixed, whole-run alignment) -- e.g. the static
    alignment file's name, for provenance in the monitoring CSV output. When
    ``schedule`` is given, each cluster's label instead comes from
    ``schedule.label_at()`` and this parameter is unused.
    """
    fitter = PoseFitter(
        tel_z=z_tel,
        alignment=alignment,
        tel_id=0,
        prb_id=1,
        tel_pos_paths=tel.pos_paths,
        prb_pos_paths=prb.pos_paths,
        tot_thresh=tot_thresh,
        tot_weights=tot_weights,
        min_anchor_planes=min_anchor_planes,
        prb_fibers_per_ribbon=fibers_per_ribbon,
        chi2_track=chi2_track,
        max_cluster_width=max_cluster_width,
    )
    label = alignment_label
    tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
    prb_stream = reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0)
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        if schedule is not None:
            t_ns = _cluster_tel_time(cluster, tel_id=0)
            if t_ns is not None:
                corr = schedule.at(t_ns)
                if corr is not fitter.alignment:
                    fitter.update_alignment(corr)
                label = schedule.label_at(t_ns)
        co = fitter.decode_cluster(cluster)
        if co is not None:
            yield co._replace(alignment_label=label)
