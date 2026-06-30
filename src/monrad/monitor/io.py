"""Shared detector-setup helpers for the monitoring drivers.

The pose-fitting drivers (``resolution``, ``timeseries``, ``multiprobe``) all
need the same two pieces of setup that ``scripts/run_pipeline.py`` performs by
hand: locate a detector's file pairs + header, and run the telescope alignment
pass.  Extracting them here keeps the drivers (and the script) from each
carrying their own copy.
"""

import math
from collections import Counter
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ..alignment import AlignmentAccumulator, AlignmentCorrection
from ..coincidence import coincidence_stream
from ..pose import Coincidence, PoseFitter
from ..reconstruction import decode_position
from ..synthetic.generate import STRIP_MM
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


def stream_coincidences(
    tel: DetectorFiles,
    prb: DetectorFiles,
    *,
    z_tel: np.ndarray,
    alignment: AlignmentCorrection,
    tot_thresh: int = 1,
    tot_weights: bool = False,
) -> Iterator[Coincidence]:
    """Yield decoded probe–telescope coincidences for one acquisition.

    Wires the stage-5 :class:`~monrad.pose.PoseFitter` decode path
    (combinatorial telescope track finder + alignment correction + track/probe
    quality cuts) to two stage-1 :func:`~monrad.timing.reconstruct_stream`
    generators and emits one :class:`~monrad.pose.Coincidence` per surviving
    cluster.  Shared by the ``resolution`` and ``timeseries`` drivers so the
    fitter wiring and the ``tel_id``/``prb_id`` convention live in one place.
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
    )
    tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
    prb_stream = reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0)
    for cluster in coincidence_stream([tel_stream, prb_stream], detector_ids=[0, 1]):
        co = fitter.decode_cluster(cluster)
        if co is not None:
            yield co
