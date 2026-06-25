"""Shared detector-setup helpers for the monitoring drivers.

The pose-fitting drivers (``resolution``, ``timeseries``, ``multiprobe``) all
need the same two pieces of setup that ``scripts/run_pipeline.py`` performs by
hand: locate a detector's file pairs + header, and run the telescope alignment
pass.  Extracting them here keeps the drivers (and the script) from each
carrying their own copy.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..alignment import AlignmentAccumulator, AlignmentCorrection
from ..reconstruction import decode_position
from ..timing import (
    find_file_pairs,
    load_header_params,
    reconstruct_stream,
)


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


def fit_alignment(
    tel: DetectorFiles,
    z_tel: np.ndarray,
    *,
    tot_thresh: int = 1,
    tot_weights: bool = False,
) -> tuple[AlignmentCorrection, Counter]:
    """Run the stage-4 telescope alignment pass over one telescope acquisition.

    Mirrors ``run_pipeline.py``'s pass 1a: stream every telescope event,
    decode its three-plane hits, and feed them to an
    :class:`AlignmentAccumulator`.  Returns the fitted correction together with
    the per-event :class:`monrad.timing.Quality` histogram (the same count the
    script prints for stage 1), so callers do not have to re-stream the
    telescope just to tally event quality.
    """
    accum = AlignmentAccumulator(z_tel=z_tel)
    quality: Counter = Counter()
    for ev, ref in reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0):
        quality[ev.quality] += 1
        hits = decode_position(
            ref, tel.pos_paths, n_cols=3, tot_thresh=tot_thresh, tot_weights=tot_weights
        )
        accum.add(hits)
    return accum.flush(), quality
