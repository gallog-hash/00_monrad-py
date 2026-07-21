"""Scan the geometric-cut family (``chi2_track`` and friends) against real data.

Answers "where should the stage-3/5 geometric cuts sit?" with measured curves
rather than a single asserted number.  Stages 1-2 are provably identical to the
MATLAB reference (2201/2201 raw coincidences), so the entire yield gap lives in
the cuts this script scans:

  pre-fit   ``max_cluster_width`` -> candidate enumeration (stage 3)
            ``min_anchor_planes`` -> before the combinatorial search
            ``chi2_track``        -> after it
            probe hit quality     -> after that
  post-fit  ``max_rigidity_resid_mm``, ``max_off_probe_mm``,
            the Mahalanobis cut inside ``fit_probe_pose``

Two tiers, because a dense grid of full decodes is unaffordable:

**Tier A (decode)** streams stage 1->2->3 once per
``(tot_thresh, tot_weights, max_cluster_width)`` config at the *loosest*
``chi2_track``/``min_anchor_planes``, caching one record per stage-2 cluster to
an ``.npz``.  Those three parameters are the only ones that change candidate
enumeration.

**Tier B (replay)** reproduces any *tighter* ``(chi2_track,
min_anchor_planes)`` and any post-fit gate combination exactly from that cache,
offline.  This is sound because ``chi2_track`` is applied strictly *after* the
combinatorial search (``PoseFitter._decode_telescope_track``) and so never
influences which triple wins, ``min_anchor_planes`` is a pure pass/fail on the
cached ``cand_counts``, and the post-fit gates act on plain ``Coincidence``
tuples.  ``tests/test_scan_geometric_cuts.py`` pins that equivalence against a
live ``PoseFitter`` run -- the whole design rests on it.

Nothing here re-implements the search or the gates: the decode pass drives the
real :class:`~monrad.pose.PoseFitter` and the replay re-applies the real
:func:`~monrad.pose.filter_rigidity` / :func:`~monrad.pose.filter_off_probe` /
:func:`~monrad.pose.fit_probe_pose`.

Usage
-----
    uv run python scripts/scan_geometric_cuts.py \\
        --telescope data/0_testLab_20210723/Base \\
        --probe data/0_testLab_20210723/Probe_0 \\
        --z-tel 0 -1340 -670 --n-probe-ch 40 --tot-weights \\
        --alignment reports/cut_scan_testLab_20210723/alignment \\
        --label clean --start 20210724_000000 --end 20210724_060000 \\
        --out reports/cut_scan_testLab_20210723 --stage all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))
sys.path.insert(0, str(_HERE))  # so `import scan_plots` works from any cwd

from monrad.alignment import AlignmentCorrection, load_alignment  # noqa: E402
from monrad.coincidence import coincidence_stream  # noqa: E402
from monrad.decoders.position import POS_HALF_BITS  # noqa: E402
from monrad.monitor import cli_args as ca  # noqa: E402
from monrad.monitor.io import (  # noqa: E402
    AlignmentSchedule,
    DetectorFiles,
    _cluster_tel_time,
    _parse_file_ts,
    fit_alignment,
    load_alignment_schedule,
    load_detector,
    static_alignment_label,
    validate_probe_footprint,
)
from monrad.pose import (  # noqa: E402
    GATE_ORDER,
    Coincidence,
    PoseFitter,
    PoseResult,
    filter_off_probe,
    filter_rigidity,
    fit_probe_pose,
)
from monrad.pose import optimize as pose_optimize  # noqa: E402
from monrad.pose.optimize import tel_align_arrays  # noqa: E402
from monrad.reconstruction import GOOD_QUALITIES  # noqa: E402
from monrad.timing import reconstruct_stream  # noqa: E402

STRIP_MM = 10.0  # mm per channel strip -- DESIGN.md §6.5

# Sentinel for "no cut" in the cache: the decode pass runs at chi2_track=inf so
# every cluster that clears the anchor gate reaches the probe-quality step and
# is therefore replayable at any finite threshold.
DECODE_CHI2 = math.inf

# Default grids (the plan's B1/B2/B3 stages).
CHI2_GRID = (2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 25.0, 37.0, 50.0, 100.0)
ANCHOR_GRID = (1, 2, 3)
MAX_RESID_MM_GRID = (4.71, 7.0, 10.0, 14.29, 20.0)
RIGIDITY_GRID = (None, 50.0, 100.0, 200.0)
OFF_PROBE_GRID = (None, 50.0, 100.0, 200.0)
MAHAL_GRID = (3.0, 4.0, 6.0)


# ── Tier A: the per-cluster cache ─────────────────────────────────────────


@dataclass
class ClusterCache:
    """One record per stage-2 coincidence cluster, from a single decode pass.

    Every array is indexed by cluster, in stream order.  Fields a cluster never
    reached are ``nan`` (floats), ``-1`` (``cand_counts``) or ``""`` (strings) --
    e.g. an ``ambiguous_cluster`` has no ``cand_counts``, and a
    ``zero_candidate_plane`` has no ``chi2``.

    ``resid`` is the winning triple's per-plane, per-axis line-fit residual in
    mm, computed in the alignment-corrected frame at decode time (the same
    frame ``_fit_triple`` fits in).  Caching it here is what lets the replay
    compare the pipeline's sigma-adaptive chi2 cut against a MATLAB-style
    absolute-mm cut without re-decoding.
    """

    meta: dict
    t_ns: np.ndarray  # (N,)   int64
    reason: np.ndarray  # (N,)   U24 -- GATE_ORDER entry or "accepted"
    cand_counts: np.ndarray  # (N,3) int16, -1 = not reached
    chi2: np.ndarray  # (N,)   f8, nan = not reached
    prb_quality: np.ndarray  # (N,)   U16, "" = not reached
    tel_quality: np.ndarray  # (N,3) U8, "" = not reached
    cand_xy: np.ndarray  # (N,3,2) f8 winning triple raw (x_mm, y_mm)
    cand_sigma: np.ndarray  # (N,3,2) f8 winning triple (sigma_x, sigma_y)
    resid: np.ndarray  # (N,3,2) f8 per-plane fit residual (mm)
    line: np.ndarray  # (N,4) f8 (a_x, b_x, a_y, b_y)
    cov_ab: np.ndarray  # (N,2,3) f8 (cov_ab_x, cov_ab_y)
    probe: np.ndarray  # (N,4) f8 (u, v, sigma_prb_x, sigma_prb_y)
    alignment_label: np.ndarray  # (N,) U32

    def __len__(self) -> int:
        return int(self.t_ns.shape[0])

    @property
    def cluster_width(self) -> np.ndarray:
        """(N,3,2) merged-channel width of each winning candidate's axis.

        Inverse of ``PlaneCandidate``'s ``sigma = STRIP_MM * width / sqrt(12)``.
        Rounded because the forward map is exact only up to float error.
        """
        return np.rint(self.cand_sigma * math.sqrt(12.0) / STRIP_MM)

    @staticmethod
    def _row_max(a: np.ndarray) -> np.ndarray:
        """Per-row max of a (N, k) array, nan for rows that are entirely nan.

        ``np.nanmax`` warns (and returns -inf) on an all-nan slice, which every
        cluster rejected before the line fit produces -- so fill instead.
        """
        flat = a.reshape(a.shape[0], -1)
        all_nan = np.all(np.isnan(flat), axis=1)
        filled = np.where(np.isnan(flat), -np.inf, flat)
        return np.where(all_nan, np.nan, np.max(filled, axis=1))

    @property
    def max_abs_resid(self) -> np.ndarray:
        """(N,) largest per-plane, per-axis absolute residual (mm), nan-safe."""
        return self._row_max(np.abs(self.resid))

    @property
    def max_cluster_width(self) -> np.ndarray:
        """(N,) widest axis of the winning triple, in merged channels."""
        return self._row_max(self.cluster_width)

    def subset(self, mask: np.ndarray) -> "ClusterCache":
        """A cache holding only the rows selected by a boolean/index array.

        Used to evaluate the same grid point over time bins (see
        :func:`bin_series`) without re-decoding: every gate the replay applies
        is per-cluster, so a row subset replays exactly as the full cache does.
        """
        return ClusterCache(
            meta=dict(self.meta),
            **{
                name: getattr(self, name)[mask]
                for name in (
                    "t_ns",
                    "reason",
                    "cand_counts",
                    "chi2",
                    "prb_quality",
                    "tel_quality",
                    "cand_xy",
                    "cand_sigma",
                    "resid",
                    "line",
                    "cov_ab",
                    "probe",
                    "alignment_label",
                )
            },
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            meta=np.array(json.dumps(self.meta)),
            **{
                name: getattr(self, name)
                for name in (
                    "t_ns",
                    "reason",
                    "cand_counts",
                    "chi2",
                    "prb_quality",
                    "tel_quality",
                    "cand_xy",
                    "cand_sigma",
                    "resid",
                    "line",
                    "cov_ab",
                    "probe",
                    "alignment_label",
                )
            },
        )

    @classmethod
    def load(cls, path: Path) -> "ClusterCache":
        with np.load(path, allow_pickle=False) as z:
            fields = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        return cls(meta=meta, **fields)


class _CacheBuilder:
    """Row accumulator for :class:`ClusterCache` (append during the stream)."""

    def __init__(self, meta: dict) -> None:
        self.meta = meta
        self.rows: list[dict] = []

    def add(
        self,
        *,
        t_ns: int,
        reason: str,
        cand_counts: tuple[int, int, int] | None,
        chi2: float | None,
        prb_quality: str | None,
        tel_quality: tuple[str, str, str] | None,
        best_cands,
        line: tuple[float, float, float, float] | None,
        cov_ab: tuple[tuple[float, ...], tuple[float, ...]] | None,
        resid: np.ndarray | None,
        probe: tuple[float, float, float, float] | None,
        alignment_label: str,
    ) -> None:
        nan3x2 = np.full((3, 2), np.nan)
        self.rows.append(
            {
                "t_ns": t_ns,
                "reason": reason,
                "cand_counts": cand_counts if cand_counts is not None else (-1, -1, -1),
                "chi2": np.nan if chi2 is None else chi2,
                "prb_quality": prb_quality or "",
                "tel_quality": tel_quality if tel_quality is not None else ("", "", ""),
                "cand_xy": (
                    nan3x2
                    if best_cands is None
                    else np.array([[c.x_mm, c.y_mm] for c in best_cands])
                ),
                "cand_sigma": (
                    nan3x2
                    if best_cands is None
                    else np.array([[c.sigma_x, c.sigma_y] for c in best_cands])
                ),
                "resid": nan3x2 if resid is None else resid,
                "line": line if line is not None else (np.nan,) * 4,
                "cov_ab": (
                    np.full((2, 3), np.nan) if cov_ab is None else np.array(cov_ab)
                ),
                "probe": probe if probe is not None else (np.nan,) * 4,
                "alignment_label": alignment_label,
            }
        )

    def finalize(self) -> ClusterCache:
        n = len(self.rows)

        def stack(key, dtype, shape=()):
            if n == 0:
                return np.empty((0, *shape), dtype=dtype)
            return np.array([r[key] for r in self.rows], dtype=dtype)

        return ClusterCache(
            meta=self.meta,
            t_ns=stack("t_ns", np.int64),
            reason=stack("reason", "U24"),
            cand_counts=stack("cand_counts", np.int16, (3,)),
            chi2=stack("chi2", np.float64),
            prb_quality=stack("prb_quality", "U16"),
            tel_quality=stack("tel_quality", "U8", (3,)),
            cand_xy=stack("cand_xy", np.float64, (3, 2)),
            cand_sigma=stack("cand_sigma", np.float64, (3, 2)),
            resid=stack("resid", np.float64, (3, 2)),
            line=stack("line", np.float64, (4,)),
            cov_ab=stack("cov_ab", np.float64, (2, 3)),
            probe=stack("probe", np.float64, (4,)),
            alignment_label=stack("alignment_label", "U32"),
        )


def _triple_residuals(
    best_cands,
    line: tuple[float, float, float, float],
    alignment: AlignmentCorrection,
    z_tel: np.ndarray,
) -> np.ndarray:
    """(3, 2) per-plane (x, y) residual of the winning triple, in mm.

    Recomputes exactly the frame :func:`~monrad.pose.optimize._fit_triple`
    fits in: raw candidate positions shifted by the fitted per-plane
    ``delta``, each axis evaluated at its own tilt-corrected plane z.  The
    residual is ``measured - predicted``, so its scale is directly comparable
    to MATLAB's absolute ``ALIGNDIST`` cut.
    """
    a_x, b_x, a_y, b_y = line
    ta = tel_align_arrays(alignment)
    z_corr = alignment.corrected_z_tel(z_tel)
    x = np.array([c.x_mm for c in best_cands]) - ta.delta_x
    y = np.array([c.y_mm for c in best_cands]) - ta.delta_y
    z_x = z_corr + ta.tilt_y * x
    z_y = z_corr + ta.tilt_x * y
    return np.column_stack([x - (a_x + b_x * z_x), y - (a_y + b_y * z_y)])


def decode_pass(
    tel: DetectorFiles,
    prb: DetectorFiles,
    *,
    z_tel: np.ndarray,
    alignment: AlignmentCorrection,
    schedule: AlignmentSchedule | None = None,
    alignment_label: str = "",
    tot_thresh: int = 1,
    tot_weights: bool = False,
    max_cluster_width: int | None = None,
    fibers_per_ribbon: int = POS_HALF_BITS,
    min_anchor_planes: int = 0,
    window_ns: int = 200,
    log_every: int = 0,
) -> ClusterCache:
    """Stream stages 1->2->3 once and cache one record per stage-2 cluster.

    Runs the real :class:`~monrad.pose.PoseFitter` decode path at
    ``chi2_track=inf`` so *every* cluster clearing the anchor gate also reaches
    the probe-quality step -- that is what makes the whole ``chi2_track`` axis
    replayable offline.

    ``min_anchor_planes`` is the *floor* the cache supports on replay: 0 caches
    everything (and searches every all-ambiguous cluster, which is expensive --
    up to 16^3 triples), 1 matches the pipeline default and is much cheaper.
    :func:`replay` refuses to go below whatever is baked in here.
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
        chi2_track=DECODE_CHI2,
        max_cluster_width=max_cluster_width,
    )
    reports: list = []
    fitter.on_decode = reports.append

    builder = _CacheBuilder(
        meta={
            "telescope": str(tel.pos_paths[0].parent),
            "probe": str(prb.pos_paths[0].parent),
            "n_tel_files": len(tel.pos_paths),
            "n_prb_files": len(prb.pos_paths),
            "z_tel": [float(v) for v in z_tel],
            "z_corr": [float(v) for v in alignment.corrected_z_tel(z_tel)],
            "tot_thresh": tot_thresh,
            "tot_weights": bool(tot_weights),
            "max_cluster_width": max_cluster_width,
            "fibers_per_ribbon": fibers_per_ribbon,
            "min_anchor_planes": min_anchor_planes,
            "chi2_track": None,  # inf -- JSON has no infinity literal
            "window_ns": window_ns,
            "alignment_label": alignment_label,
            "time_varying_alignment": schedule is not None,
        }
    )

    tel_stream = reconstruct_stream(tel.gps_paths, tel.pos_paths, tel.utc0, tel.f0)
    prb_stream = reconstruct_stream(prb.gps_paths, prb.pos_paths, prb.utc0, prb.f0)
    label = alignment_label
    t0 = time.monotonic()
    for i, cluster in enumerate(
        coincidence_stream(
            [tel_stream, prb_stream], detector_ids=[0, 1], window_ns=window_ns
        )
    ):
        if schedule is not None:
            t_cluster = _cluster_tel_time(cluster, tel_id=0)
            if t_cluster is not None:
                corr = schedule.at(t_cluster)
                if corr is not fitter.alignment:
                    fitter.update_alignment(corr)
                label = schedule.label_at(t_cluster)

        reports.clear()
        tel_result = fitter.decode_telescope_track(cluster)
        co = fitter.decode_from_telescope_track(cluster, tel_result)
        report = reports[-1]

        line = cov = resid = probe = None
        if tel_result.best_cands is not None:
            line = (tel_result.a_x, tel_result.b_x, tel_result.a_y, tel_result.b_y)
            cov = (tel_result.cov_ab_x, tel_result.cov_ab_y)
            resid = _triple_residuals(
                tel_result.best_cands, line, fitter.alignment, z_tel
            )
        if co is not None:
            probe = (co.u, co.v, co.sigma_prb_x, co.sigma_prb_y)

        builder.add(
            t_ns=tel_result.t_ns,
            reason=report.reason,
            cand_counts=tel_result.cand_counts,
            chi2=tel_result.chi2,
            prb_quality=report.prb_quality,
            tel_quality=tel_result.tel_quality,
            best_cands=tel_result.best_cands,
            line=line,
            cov_ab=cov,
            resid=resid,
            probe=probe,
            alignment_label=label,
        )
        if log_every and (i + 1) % log_every == 0:
            rate = (i + 1) / max(time.monotonic() - t0, 1e-9)
            print(f"    {i + 1:>8} clusters  ({rate:.0f}/s)", flush=True)

    return builder.finalize()


# ── Tier B: offline replay ────────────────────────────────────────────────


@dataclass
class ReplayResult:
    """Outcome of replaying one ``(chi2_track, min_anchor_planes)`` point."""

    counts: Counter  # gate reason -> n, plus "accepted"
    coincs: list[Coincidence]
    accepted_idx: np.ndarray  # cache row indices of the accepted coincidences

    @property
    def n_raw(self) -> int:
        return int(sum(self.counts.values()))


def _coincidence(cache: ClusterCache, i: int) -> Coincidence:
    """Rebuild the exact ``Coincidence`` the live fitter would have emitted."""
    a_x, b_x, a_y, b_y = cache.line[i]
    u, v, sig_x, sig_y = cache.probe[i]
    cx, cy = cache.cov_ab[i, 0], cache.cov_ab[i, 1]
    q = cache.tel_quality[i]
    return Coincidence(
        a_x=float(a_x),
        b_x=float(b_x),
        a_y=float(a_y),
        b_y=float(b_y),
        cov_ab_x=(float(cx[0]), float(cx[1]), float(cx[2])),
        cov_ab_y=(float(cy[0]), float(cy[1]), float(cy[2])),
        u=float(u),
        v=float(v),
        sigma_prb_x=float(sig_x),
        sigma_prb_y=float(sig_y),
        tel_quality=(str(q[0]), str(q[1]), str(q[2])),
        t_ns=int(cache.t_ns[i]),
        alignment_label=str(cache.alignment_label[i]),
    )


def replay(
    cache: ClusterCache,
    *,
    chi2_track: float = 4.0,
    min_anchor_planes: int = 1,
    max_resid_mm: float | None = None,
) -> ReplayResult:
    """Reproduce a tighter decode configuration offline from *cache*.

    Applies the live gate precedence of ``PoseFitter._decode_cluster`` verbatim:
    ``ambiguous_cluster`` -> ``zero_candidate_plane`` -> ``no_anchor_plane`` ->
    ``chi2_track_cut`` -> ``probe_quality`` -> accepted.

    ``max_resid_mm``, when given, replaces the sigma-adaptive chi2 test with a
    MATLAB-``ALIGNDIST``-style absolute cut on the winning triple's largest
    per-plane residual.  It reuses the ``chi2_track_cut`` funnel slot, since it
    sits at exactly the same point in the precedence -- only the cut's *shape*
    differs.  ``chi2_track`` is then ignored.

    Raises ``ValueError`` when asked for a configuration looser than the one
    the cache was decoded at (which would need candidates the pass never
    enumerated).
    """
    decoded_anchor = int(cache.meta["min_anchor_planes"])
    if min_anchor_planes < decoded_anchor:
        raise ValueError(
            f"cache was decoded at min_anchor_planes={decoded_anchor}; cannot "
            f"replay the looser {min_anchor_planes} (those clusters were never "
            f"searched). Re-run the decode stage with the lower value."
        )
    decoded_chi2 = cache.meta.get("chi2_track")
    decoded_chi2 = math.inf if decoded_chi2 is None else float(decoded_chi2)
    if max_resid_mm is None and chi2_track > decoded_chi2:
        raise ValueError(
            f"cache was decoded at chi2_track={decoded_chi2}; cannot replay the "
            f"looser {chi2_track}."
        )

    counts: Counter = Counter()
    coincs: list[Coincidence] = []
    idx: list[int] = []

    reason_arr = cache.reason
    cand = cache.cand_counts
    chi2 = cache.chi2
    score = cache.max_abs_resid if max_resid_mm is not None else None

    for i in range(len(cache)):
        r = str(reason_arr[i])
        if r in ("ambiguous_cluster", "zero_candidate_plane"):
            counts[r] += 1
            continue
        n_anchor = int(np.count_nonzero(cand[i] == 1))
        if n_anchor < min_anchor_planes:
            counts["no_anchor_plane"] += 1
            continue
        if r == "no_anchor_plane":
            # Only reachable when the cache itself was decoded with an anchor
            # gate; the guard above already rejected looser replays, so this
            # row genuinely fails at this threshold too.
            counts["no_anchor_plane"] += 1
            continue
        if score is not None:
            passed = bool(score[i] <= max_resid_mm)
        else:
            # Mirrors `best_chi2 >= chi2_track` (and the best_fit-is-None path,
            # which caches chi2 as nan): anything not strictly below is cut.
            passed = bool(chi2[i] < chi2_track)
        if not passed:
            counts["chi2_track_cut"] += 1
            continue
        if str(cache.prb_quality[i]) not in GOOD_QUALITIES:
            counts["probe_quality"] += 1
            continue
        counts["accepted"] += 1
        coincs.append(_coincidence(cache, i))
        idx.append(i)

    return ReplayResult(
        counts=counts, coincs=coincs, accepted_idx=np.array(idx, dtype=np.int64)
    )


# ── Figures of merit ──────────────────────────────────────────────────────


def _resid_rms(pose: PoseResult) -> float:
    """Absolute-mm residual RMS over *all* coincidences fed to the fit.

    Same quantity as ``monrad.monitor.timeseries._window_resid_rms``: inlier-only
    residuals look clean even in a contaminated window, because the Mahalanobis
    cut already removed the wild tracks -- counting the rejected ones back in is
    what exposes contamination.  Reimplemented here (rather than imported from a
    private name) so the scan is not coupled to that module's internals.
    """
    coincs = pose.inliers + pose.outliers
    if not coincs:
        return float("nan")
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    sq = 0.0
    for co in coincs:
        rx = (pose.t_x + co.u * c - co.v * s) - (co.a_x + co.b_x * pose.z_p)
        ry = (pose.t_y + co.u * s + co.v * c) - (co.a_y + co.b_y * pose.z_p)
        sq += rx * rx + ry * ry
    return math.sqrt(sq / len(coincs))


def probe_frame_coords(
    coincs: list[Coincidence], pose: PoseResult
) -> tuple[np.ndarray, np.ndarray]:
    """Project each track to ``pose.z_p`` and map it into probe-frame (u, v).

    Same algebra as :func:`~monrad.pose.filter_off_probe`, exposed so the purity
    estimate and the footprint figure share one definition of "where on the
    probe did this track land".
    """
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    u = np.empty(len(coincs))
    v = np.empty(len(coincs))
    for i, co in enumerate(coincs):
        dx = (co.a_x + co.b_x * pose.z_p) - pose.t_x
        dy = (co.a_y + co.b_y * pose.z_p) - pose.t_y
        u[i] = c * dx + s * dy
        v[i] = -s * dx + c * dy
    return u, v


def on_probe_purity(
    coincs: list[Coincidence], pose: PoseResult, probe_size_mm: float
) -> tuple[float, float, int]:
    """Estimate the accidental pedestal under the probe footprint.

    Tracks that genuinely passed through the probe land inside
    ``[0, probe_size_mm]^2``; time-coincident but geometrically unrelated ones
    scatter roughly flat over a much wider area.  Counting the flat population
    in the surrounding annulus (the ``2L x 2L`` box centred on the footprint,
    minus the footprint itself -- three times the footprint's area) gives a
    pedestal density that is extrapolated *into* the footprint and subtracted.

    Truth-free, and deliberately independent of any cut being scanned, so it
    can referee "did that extra yield buy signal or junk?".  It is only
    meaningful on a coincidence set that has *not* already been passed through
    ``--max-off-probe-mm``, which flattens the annulus by construction.

    Returns ``(signal_count, purity, n_inside)``.  ``purity`` is
    ``signal / n_inside``, nan when nothing landed inside.
    """
    if not coincs:
        return float("nan"), float("nan"), 0
    u, v = probe_frame_coords(coincs, pose)
    L = probe_size_mm
    inside = (u >= 0) & (u <= L) & (v >= 0) & (v <= L)
    box = (u >= -0.5 * L) & (u <= 1.5 * L) & (v >= -0.5 * L) & (v <= 1.5 * L)
    n_in = int(np.count_nonzero(inside))
    n_ring = int(np.count_nonzero(box & ~inside))
    if n_in == 0:
        return 0.0, float("nan"), 0
    # Ring area is 4L^2 - L^2 = 3L^2, i.e. 3x the footprint.
    signal = n_in - n_ring / 3.0
    return signal, signal / n_in, n_in


def _sigmas(pose: PoseResult) -> tuple[float, float, float, float]:
    """(sigma_tx, sigma_ty, sigma_theta, sigma_zp) from the pose covariance."""
    d = np.diag(pose.cov)
    with np.errstate(invalid="ignore"):
        s = np.sqrt(np.where(d >= 0, d, np.nan))
    return float(s[0]), float(s[1]), float(s[2]), float(s[3])


def _half_spread(pose: PoseResult) -> float:
    """``|Δz_p| / σ_zp`` between the even/odd stratified halves.

    ``PoseResult.half_params`` refits each parity half at the fitted theta;
    a consistent fit keeps the two within ~1 sigma.  Zero rows mean a half had
    fewer than 3 coincidences, in which case this is nan.
    """
    hp = pose.half_params
    if not np.any(hp[0]) or not np.any(hp[1]):
        return float("nan")
    sigma_zp = _sigmas(pose)[3]
    if not math.isfinite(sigma_zp) or sigma_zp <= 0:
        return float("nan")
    return abs(hp[0][3] - hp[1][3]) / sigma_zp


def evaluate_with_pose(
    cache: ClusterCache,
    *,
    chi2_track: float = 4.0,
    min_anchor_planes: int = 1,
    max_resid_mm: float | None = None,
    max_rigidity_resid_mm: float | None = None,
    max_off_probe_mm: float | None = None,
    mahal_cut: float = 4.0,
    probe_size_mm: float = 400.0,
    min_fit: int = PoseFitter.MIN_FIT,
    alignment: AlignmentCorrection,
) -> tuple[dict, PoseResult | None]:
    """Replay one full grid point; return its metrics row *and* the fitted pose.

    The pose itself is only needed by the footprint figure; :func:`evaluate`
    is the row-only wrapper the grid driver uses.

    Applies the post-fit gates in the same order, with the same bootstrap
    anchors and the same "too few survivors -> keep the unfiltered set"
    fallbacks, as ``scripts/run_pipeline.py``: rigidity (anchored on an
    unfiltered fit's ``z_p``), then off-probe (anchored on a fit of the
    rigidity-gated set), then the pose fit itself.

    ``mahal_cut`` has no CLI flag in the pipeline (``optimize._MAHAL_CUT`` is a
    module global read inside ``fit_probe_pose``), so it is patched around the
    fits here rather than by editing source.
    """
    rep = replay(
        cache,
        chi2_track=chi2_track,
        min_anchor_planes=min_anchor_planes,
        max_resid_mm=max_resid_mm,
    )
    z_corr = np.array(cache.meta["z_corr"])
    row: dict = {
        "chi2_track": chi2_track,
        "min_anchor_planes": min_anchor_planes,
        "max_resid_mm": max_resid_mm,
        "max_rigidity_resid_mm": max_rigidity_resid_mm,
        "max_off_probe_mm": max_off_probe_mm,
        "mahal_cut": mahal_cut,
        "tot_thresh": cache.meta["tot_thresh"],
        "tot_weights": cache.meta["tot_weights"],
        "max_cluster_width": cache.meta["max_cluster_width"],
        "n_raw": rep.n_raw,
        **{f"gate_{g}": int(rep.counts[g]) for g in GATE_ORDER},
        "n_accepted": int(rep.counts["accepted"]),
    }

    coincs = rep.coincs
    original = list(coincs)
    n_rigidity_dropped = 0
    n_off_probe_dropped = 0
    rigidity_fallback = off_probe_fallback = False

    prev_mahal = pose_optimize._MAHAL_CUT
    pose_optimize._MAHAL_CUT = float(mahal_cut)
    try:
        if max_rigidity_resid_mm is not None and len(coincs) >= 3:
            z_ref = fit_probe_pose(coincs, z_corr, alignment).z_p
            kept, dropped = filter_rigidity(coincs, z_ref, max_rigidity_resid_mm)
            if len(kept) >= 3:
                n_rigidity_dropped = len(dropped)
                coincs = kept
            else:
                rigidity_fallback = True
        if max_off_probe_mm is not None and len(coincs) >= 3:
            ref_pose = fit_probe_pose(coincs, z_corr, alignment)
            kept, dropped = filter_off_probe(
                coincs, ref_pose, probe_size_mm, max_off_probe_mm
            )
            if len(kept) >= 3:
                n_off_probe_dropped = len(dropped)
                coincs = kept
            else:
                off_probe_fallback = True

        pose = (
            fit_probe_pose(coincs, z_corr, alignment)
            if len(coincs) >= min_fit
            else None
        )
        # Purity is judged on the *ungated* accepted set: --max-off-probe-mm
        # empties the pedestal annulus by construction, which would report a
        # spurious ~100% purity for exactly the configurations it is applied to.
        purity_pose = pose
        if pose is not None and max_off_probe_mm is not None and len(original) >= 3:
            purity_pose = fit_probe_pose(original, z_corr, alignment)
    finally:
        pose_optimize._MAHAL_CUT = prev_mahal

    row.update(
        {
            "n_rigidity_dropped": n_rigidity_dropped,
            "rigidity_fallback": rigidity_fallback,
            "n_off_probe_dropped": n_off_probe_dropped,
            "off_probe_fallback": off_probe_fallback,
            "n_fed": len(coincs),
        }
    )
    if pose is None:
        row.update(
            {
                "n_inliers": 0,
                "fit": "skipped",
                **{
                    k: float("nan")
                    for k in (
                        "t_x",
                        "t_y",
                        "theta_deg",
                        "z_p",
                        "sigma_tx",
                        "sigma_ty",
                        "sigma_theta",
                        "sigma_zp",
                        "sigma_zp_sqrtN",
                        "resid_rms",
                        "inlier_frac",
                        "signal_count",
                        "purity",
                        "half_spread_zp",
                    )
                },
            }
        )
        return row, None

    s_tx, s_ty, s_th, s_zp = _sigmas(pose)
    assert purity_pose is not None  # tracks `pose`, which is non-None here
    signal, purity, _n_in = on_probe_purity(
        purity_pose.inliers + purity_pose.outliers, purity_pose, probe_size_mm
    )
    row.update(
        {
            "n_inliers": pose.n_inliers,
            "fit": "ok",
            "t_x": pose.t_x,
            "t_y": pose.t_y,
            "theta_deg": math.degrees(pose.theta),
            "z_p": pose.z_p,
            "sigma_tx": s_tx,
            "sigma_ty": s_ty,
            "sigma_theta": s_th,
            "sigma_zp": s_zp,
            "sigma_zp_sqrtN": s_zp * math.sqrt(pose.n_inliers),
            "resid_rms": _resid_rms(pose),
            "inlier_frac": pose.n_inliers / len(coincs),
            "signal_count": signal,
            "purity": purity,
            "half_spread_zp": _half_spread(pose),
        }
    )
    return row, pose


def evaluate(cache: ClusterCache, **kw) -> dict:
    """:func:`evaluate_with_pose` without the pose -- just the metrics row."""
    return evaluate_with_pose(cache, **kw)[0]


# ── Grid drivers ──────────────────────────────────────────────────────────


def bin_series(
    cache: ClusterCache, *, bin_s: float = 300.0, alignment: AlignmentCorrection, **kw
) -> list[dict]:
    """Evaluate one grid point independently in each fixed-length time bin.

    The ANOMALY control: on ``testLab_20210723`` the two known bad 5-minute bins
    (17:15 and 18:10 UTC) must stay clear outliers in z_p / resid_rms / inlier
    fraction under whatever cut configuration the scan ends up recommending.
    Also doubles as a stability metric on the CLEAN window, where the probe is
    static and every bin should agree.

    ``kw`` is forwarded to :func:`evaluate`; each row gains ``bin_index`` and
    ``bin_start_ns``.
    """
    if len(cache) == 0:
        return []
    bin_ns = int(bin_s * 1e9)
    t0 = int(cache.t_ns.min())
    idx = (cache.t_ns - t0) // bin_ns
    rows = []
    for b in np.unique(idx):
        row = evaluate(cache.subset(idx == b), alignment=alignment, **kw)
        row["bin_index"] = int(b)
        row["bin_start_ns"] = t0 + int(b) * bin_ns
        rows.append(row)
    return rows


def _cache_tag(
    tot_thresh: int, tot_weights: bool, max_cluster_width: int | None
) -> str:
    w = "none" if max_cluster_width is None else str(max_cluster_width)
    return f"tot{tot_thresh}_{'w' if tot_weights else 'nw'}_width{w}"


def cache_path(
    out: Path, label: str, tot_thresh, tot_weights, max_cluster_width
) -> Path:
    return (
        out
        / f"cache_{label}_{_cache_tag(tot_thresh, tot_weights, max_cluster_width)}.npz"
    )


def run_grid(
    caches: dict[str, ClusterCache],
    *,
    chi2_grid,
    anchor_grid,
    max_resid_mm_grid,
    rigidity_grid,
    off_probe_grid,
    mahal_grid,
    probe_size_mm: float,
    min_fit: int,
    alignment: AlignmentCorrection,
    top_k: int = 3,
) -> list[dict]:
    """Run the plan's staged B1 -> B2 -> B3 grid over every Tier-A cache.

    Staged rather than one full outer product: B1 sweeps
    ``chi2_track x min_anchor_planes`` with the post-fit gates held fixed, B2
    swaps the cut's *shape* for an absolute-mm one at B1's best anchor setting,
    and B3 opens up the post-fit gates only at the best few (cut, anchor)
    points.  Each row carries a ``stage`` column so the CSV stays readable.
    """
    rows: list[dict] = []

    def record(cache_key, stage, **kw):
        row = evaluate(
            caches[cache_key],
            probe_size_mm=probe_size_mm,
            min_fit=min_fit,
            alignment=alignment,
            **kw,
        )
        row["cache"] = cache_key
        row["stage"] = stage
        rows.append(row)
        return row

    def score(row):
        """Estimated signal count, the plan's recommended operating metric."""
        s = row.get("signal_count")
        return -math.inf if s is None or not math.isfinite(s) else s

    for key in caches:
        # ── B1: chi2 x anchor, post-fit gates held at the shipped defaults ──
        b1 = [
            record(
                key,
                "B1",
                chi2_track=c,
                min_anchor_planes=a,
                max_rigidity_resid_mm=100.0,
                max_off_probe_mm=None,
                mahal_cut=4.0,
            )
            for a in anchor_grid
            for c in chi2_grid
        ]
        best_anchor = max(b1, key=score)["min_anchor_planes"]

        # ── B2: cut shape -- sigma-adaptive chi2 vs absolute max-|residual| ──
        b2 = [
            record(
                key,
                "B2",
                chi2_track=math.nan,
                min_anchor_planes=best_anchor,
                max_resid_mm=m,
                max_rigidity_resid_mm=100.0,
                max_off_probe_mm=None,
                mahal_cut=4.0,
            )
            for m in max_resid_mm_grid
        ]

        # ── B3: post-fit gates at the best few (cut, anchor) points ─────────
        best_points = sorted(b1 + b2, key=score, reverse=True)[:top_k]
        for pt in best_points:
            for rig in rigidity_grid:
                for off in off_probe_grid:
                    for mah in mahal_grid:
                        if (
                            rig == 100.0
                            and off is None
                            and mah == 4.0
                            and pt["stage"] in ("B1", "B2")
                        ):
                            continue  # already measured as the B1/B2 point
                        record(
                            key,
                            "B3",
                            chi2_track=pt["chi2_track"],
                            min_anchor_planes=pt["min_anchor_planes"],
                            max_resid_mm=pt["max_resid_mm"],
                            max_rigidity_resid_mm=rig,
                            max_off_probe_mm=off,
                            mahal_cut=mah,
                        )
    return rows


def write_grid_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_funnel_csv(rows: list[dict], path: Path) -> None:
    """The gate funnel alone, one row per grid point, with running survivors."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "cache",
        "stage",
        "chi2_track",
        "max_resid_mm",
        "min_anchor_planes",
        "n_raw",
        *[f"gate_{g}" for g in GATE_ORDER],
        "n_accepted",
        "n_fed",
        "n_inliers",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ── File-window selection ─────────────────────────────────────────────────


def _parse_window_bound(text: str) -> datetime:
    """Parse a ``--start``/``--end`` bound: ``YYYYMMDD`` or ``YYYYMMDD_HHMMSS``."""
    for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable window bound {text!r} (want YYYYMMDD[_HHMMSS])")


def slice_detector(
    det: DetectorFiles, start: datetime | None, end: datetime | None
) -> DetectorFiles:
    """Restrict a detector to the file pairs whose acquisition time is in
    ``[start, end)``.

    Applied *per detector*, never by a shared filename prefix: the probe DAQ
    starts ~25 s after the telescope on this dataset, so a common prefix picks
    up one extra (or one missing) 5-minute batch on one side of the window.
    """
    pairs = sorted(zip(det.gps_paths, det.pos_paths), key=lambda p: p[0].name)
    kept = [
        (g, p)
        for g, p in pairs
        if (start is None or _parse_file_ts(g.name) >= start)
        and (end is None or _parse_file_ts(g.name) < end)
    ]
    return DetectorFiles(
        utc0=det.utc0,
        f0=det.f0,
        gps_paths=[g for g, _ in kept],
        pos_paths=[p for _, p in kept],
    )


def check_window_pairing(
    tel: DetectorFiles, prb: DetectorFiles, *, tolerance_s: float = 120.0
) -> list[str]:
    """Sanity-check a sliced window's telescope/probe file lists.

    Returns a list of human-readable problems (empty when the window is sane):
    unequal batch counts, or a per-index acquisition-time gap beyond
    ``tolerance_s`` -- either means the window edges picked up a batch on one
    detector that has no counterpart on the other.
    """
    problems: list[str] = []
    n_t, n_p = len(tel.gps_paths), len(prb.gps_paths)
    if n_t != n_p:
        problems.append(f"telescope has {n_t} file pairs but probe has {n_p}")
    for i in range(min(n_t, n_p)):
        dt = abs(
            (
                _parse_file_ts(tel.gps_paths[i].name)
                - _parse_file_ts(prb.gps_paths[i].name)
            ).total_seconds()
        )
        if dt > tolerance_s:
            problems.append(
                f"batch {i}: {tel.gps_paths[i].name} vs {prb.gps_paths[i].name} "
                f"differ by {dt:.0f} s (> {tolerance_s:.0f} s)"
            )
    return problems


# ── Mid-stream re-anchoring ───────────────────────────────────────────────
#
# Stage 1 anchors a stream's *first* PPS to the header's ``utc0`` and then
# counts PPS edges (``monrad.timing.reconstruct_stream``).  Feed it a slice
# that starts mid-acquisition and every event comes back timed as though the
# slice were the start of the run -- shifted early by however long the real
# acquisition had already been running.
#
# That shift is per detector, and the two detectors do not share it: on
# testLab_20210723 the telescope's window files begin 12:20:02 after its own
# first file and the probe's 12:20:00 after its own, so slicing skews the two
# streams by 2 s against a 200 ns coincidence window -- every coincidence is
# lost, silently, with a perfectly healthy-looking decode of zero clusters.
#
# The fix has two parts:
#   * absolute -- shift each detector forward by its own file-name elapsed
#     time.  File rotation is not PPS-locked, so this lands within a few
#     seconds of true UTC, which is ample for 5-minute time bins and for
#     picking the right multi-hour alignment window;
#   * relative -- the part that must be exact.  PPS edges are simultaneous
#     across detectors and land on whole UTC seconds, so the residual probe-vs-
#     telescope error is an integer number of seconds.  It is *measured* from
#     the data by maximising raw coincidence yield over candidate shifts,
#     rather than trusted from the file names.


def event_times(det: DetectorFiles) -> np.ndarray:
    """Every stage-1 reconstructed event time (integer ns) for one detector."""
    return np.fromiter(
        (
            ev.t_ns
            for ev, _ref in reconstruct_stream(
                det.gps_paths, det.pos_paths, det.utc0, det.f0
            )
        ),
        dtype=np.int64,
    )


def count_matches(
    t_tel: np.ndarray, t_prb: np.ndarray, shift_ns: int, window_ns: int = 200
) -> int:
    """Telescope events with a probe event within ``window_ns`` after shifting.

    The same pairing :func:`~monrad.coincidence.coincidence_stream` performs,
    reduced to a count -- cheap enough (two ``searchsorted`` calls) to evaluate
    over a whole range of candidate shifts from one decode of the event times.
    """
    shifted = t_prb + shift_ns
    lo = np.searchsorted(shifted, t_tel - window_ns, side="left")
    hi = np.searchsorted(shifted, t_tel + window_ns, side="right")
    return int(np.count_nonzero(hi > lo))


@dataclass
class ShiftCalibration:
    """Result of measuring a window's probe-vs-telescope clock shift."""

    best_shift_ns: int
    nominal_shift_ns: int
    counts: dict[int, int]  # shift (whole seconds) -> raw coincidence count

    @property
    def best_count(self) -> int:
        return self.counts[self.best_shift_ns // 10**9]

    @property
    def runner_up_count(self) -> int:
        others = [n for s, n in self.counts.items() if s != self.best_shift_ns // 10**9]
        return max(others) if others else 0

    @property
    def is_confident(self) -> bool:
        """True when the winning shift stands clearly above every other.

        A genuine alignment produces a sharp peak: neighbouring whole-second
        shifts move every event a full second away from its partner, so they
        retain only accidental matches.  A flat or near-tied scan means the
        window is not really aligned and the counts are accidentals -- better
        to refuse than to scan cuts against noise.
        """
        return self.best_count >= 10 and self.best_count > 5 * self.runner_up_count


def calibrate_shift(
    tel: DetectorFiles,
    prb: DetectorFiles,
    *,
    nominal_shift_ns: int,
    search_s: int = 5,
    window_ns: int = 200,
) -> ShiftCalibration:
    """Measure the probe clock shift that maximises raw coincidence yield.

    Scans whole-second shifts within ``search_s`` of ``nominal_shift_ns`` (the
    file-name estimate).  Whole seconds only: both detectors' PPS edges are the
    same physical GPS pulses, so any residual misanchoring is an exact integer
    number of them.
    """
    t_tel = event_times(tel)
    t_prb = event_times(prb)
    nominal_s = int(round(nominal_shift_ns / 1e9))
    counts = {
        s: count_matches(t_tel, t_prb, s * 10**9, window_ns)
        for s in range(nominal_s - search_s, nominal_s + search_s + 1)
    }
    best_s = max(counts, key=lambda s: counts[s])
    return ShiftCalibration(
        best_shift_ns=best_s * 10**9,
        nominal_shift_ns=nominal_shift_ns,
        counts=counts,
    )


def file_name_elapsed(det: DetectorFiles, full: DetectorFiles) -> timedelta:
    """How long the acquisition had been running when this slice's first file opened.

    Measured from the DAQ's own file-name clock, so it is independent of
    whatever offset that clock carries against UTC -- only the *difference*
    between two file names matters.  Good to a few seconds, since rotation is
    not PPS-locked; that residual is exactly what the measured correction in
    :func:`reanchor_window` removes.  (Observed on testLab_20210723: the
    file-name estimate put the probe 2 s from the telescope, the measured
    answer was 3 s -- trusting file names alone would have yielded zero
    coincidences.)
    """
    return _parse_file_ts(det.gps_paths[0].name) - _parse_file_ts(
        full.gps_paths[0].name
    )


def reanchor_window(
    tel: DetectorFiles,
    prb: DetectorFiles,
    *,
    full_tel: DetectorFiles,
    full_prb: DetectorFiles,
    search_s: int = 5,
    window_ns: int = 200,
    measure: bool = True,
) -> tuple[DetectorFiles, DetectorFiles, ShiftCalibration | None]:
    """Re-anchor a sliced window so its two streams share a clock (and UTC).

    Step 1 advances each detector by its *own* file-name elapsed time, which
    puts both roughly back on true UTC and removes the bulk of the skew.

    Step 2 measures what is left.  Both detectors' PPS edges are the same
    physical GPS pulses landing on whole UTC seconds, so the residual is an
    exact integer number of seconds; it is found by scanning whole-second
    probe shifts for the one that maximises raw coincidence yield.  This costs
    one extra stage-1 pass over the window and is what turns a silent
    zero-coincidence window into a measured, checkable number.

    ``measure=False`` skips step 2 and trusts the file names alone -- only
    correct if file rotation happens to be PPS-aligned on both detectors.

    Returns the re-anchored detectors and the calibration (``None`` when
    ``measure=False``).
    """
    tel_a = tel._replace(utc0=tel.utc0 + file_name_elapsed(tel, full_tel))
    prb_a = prb._replace(utc0=prb.utc0 + file_name_elapsed(prb, full_prb))
    if not measure:
        return tel_a, prb_a, None
    cal = calibrate_shift(
        tel_a, prb_a, nominal_shift_ns=0, search_s=search_s, window_ns=window_ns
    )
    prb_a = prb_a._replace(
        utc0=prb_a.utc0 + timedelta(microseconds=cal.best_shift_ns / 1000)
    )
    return tel_a, prb_a, cal


# ── CLI ───────────────────────────────────────────────────────────────────


def _float_or_none(text: str) -> float | None:
    return None if text.lower() in ("none", "off") else float(text)


def build_parser() -> argparse.ArgumentParser:
    p = ca.MacroArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ca.add_telescope_arg(p)
    p.add_argument(
        "--probe",
        type=Path,
        required=True,
        metavar="DIR",
        help="Probe acquisition directory.",
    )
    ca.add_z_tel_arg(p)
    ca.add_out_arg(p, default=Path("reports/cut_scan"))
    ca.add_alignment_arg(p)
    ca.add_tot_thresh_arg(p)
    ca.add_tot_weights_arg(p)
    ca.add_no_plots_arg(p)
    p.add_argument(
        "--n-probe-ch",
        type=int,
        default=40,
        metavar="N",
        help="Probe channels per axis; the footprint is N*10 mm (default: 40).",
    )
    p.add_argument(
        "--fibers-per-ribbon",
        type=int,
        default=POS_HALF_BITS,
        metavar="N",
        help=f"Probe fiber x ribbon combine factor (default: {POS_HALF_BITS}).",
    )
    p.add_argument(
        "--max-cluster-width",
        type=int,
        action="append",
        default=None,
        metavar="N",
        help="Cap on a hit centroid's per-axis merged-channel width. Repeatable: "
        "each value costs its own Tier-A decode pass (it changes candidate "
        "enumeration, so it cannot be replayed). Omit for no cap.",
    )
    p.add_argument(
        "--decode-anchor",
        type=int,
        default=1,
        choices=range(0, 4),
        metavar="N",
        help="min_anchor_planes the Tier-A decode runs at, and hence the "
        "loosest value the replay can reproduce (default: 1). 0 also searches "
        "all-ambiguous clusters -- up to 16^3 triples each, far slower.",
    )
    p.add_argument("--label", default="set", help="Name for this file window's cache.")
    p.add_argument("--start", default=None, metavar="YYYYMMDD[_HHMMSS]")
    p.add_argument("--end", default=None, metavar="YYYYMMDD[_HHMMSS]")
    p.add_argument(
        "--allow-unpaired-window",
        action="store_true",
        help="Proceed even if the sliced telescope/probe file lists don't pair up.",
    )
    p.add_argument(
        "--no-reanchor",
        action="store_true",
        help="Skip mid-stream re-anchoring of a --start/--end window. Stage 1 "
        "times a slice as though it were the start of the run, which skews the "
        "two detectors against each other by seconds and yields zero "
        "coincidences -- only pass this if the slice starts at the acquisition "
        "start.",
    )
    p.add_argument(
        "--trust-file-names",
        action="store_true",
        help="Re-anchor from file names alone, skipping the measured "
        "whole-second correction (and the stage-1 pass it costs). Only correct "
        "if file rotation is PPS-aligned on both detectors.",
    )
    p.add_argument(
        "--reanchor-search-s",
        type=int,
        default=5,
        metavar="N",
        help="Half-width, in whole seconds, of the clock-shift search used to "
        "re-anchor a window (default: 5).",
    )
    p.add_argument("--chi2-grid", type=float, nargs="+", default=list(CHI2_GRID))
    p.add_argument("--anchor-grid", type=int, nargs="+", default=list(ANCHOR_GRID))
    p.add_argument(
        "--max-resid-mm-grid", type=float, nargs="+", default=list(MAX_RESID_MM_GRID)
    )
    p.add_argument(
        "--rigidity-grid", type=_float_or_none, nargs="+", default=list(RIGIDITY_GRID)
    )
    p.add_argument(
        "--off-probe-grid", type=_float_or_none, nargs="+", default=list(OFF_PROBE_GRID)
    )
    p.add_argument("--mahal-grid", type=float, nargs="+", default=list(MAHAL_GRID))
    ca.add_min_fit_arg(p)
    p.add_argument(
        "--stage",
        choices=("decode", "replay", "plots", "all"),
        default="all",
        help="Which tier to run (default: all). 'replay' and 'plots' reuse the "
        "cached .npz from an earlier 'decode'.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=5000,
        metavar="N",
        help="Progress line every N clusters during decode (0 = silent).",
    )
    return p


def load_run_alignment(
    args: argparse.Namespace, tel: DetectorFiles, z_tel: np.ndarray
) -> tuple[AlignmentCorrection, AlignmentSchedule | None, str]:
    """Resolve ``--alignment`` to (correction, schedule, label).

    A directory becomes a time-varying :class:`AlignmentSchedule`; a file a
    static correction; omitting it falls back to the same in-run fit the
    monitoring drivers use -- which is *discouraged* for a cut scan, since the
    alignment would then be refit per window and cut effects would be confounded
    with alignment drift.
    """
    if args.alignment is None:
        print(
            "  ! no --alignment: fitting in-run (cut effects will be confounded "
            "with the alignment fit; prefer a fixed monrad-align JSON)"
        )
        correction, _quality = fit_alignment(
            tel, z_tel, tot_thresh=args.tot_thresh, tot_weights=args.tot_weights
        )
        return correction, None, "auto"
    if args.alignment.is_dir():
        schedule = load_alignment_schedule(args.alignment, expect_z_tel=z_tel)
        return schedule.corrections[0], schedule, "schedule"
    correction = load_alignment(args.alignment, expect_z_tel=z_tel)
    return correction, None, static_alignment_label(args.alignment)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        ca.validate_fibers_per_ribbon([args.fibers_per_ribbon])
        ca.validate_min_fit(args.min_fit)
        validate_probe_footprint(args.n_probe_ch, args.fibers_per_ribbon)
    except ValueError as exc:
        parser.error(str(exc))

    z_tel = np.array(args.z_tel, dtype=float)
    widths = args.max_cluster_width if args.max_cluster_width else [None]
    for w in widths:
        if w is not None and w < 1:
            parser.error(f"--max-cluster-width must be >= 1; got {w}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    probe_size_mm = args.n_probe_ch * STRIP_MM

    full_tel = load_detector(args.telescope)
    full_prb = load_detector(args.probe)
    tel, prb = full_tel, full_prb
    start = _parse_window_bound(args.start) if args.start else None
    end = _parse_window_bound(args.end) if args.end else None
    if start or end:
        tel = slice_detector(full_tel, start, end)
        prb = slice_detector(full_prb, start, end)
        if not tel.gps_paths or not prb.gps_paths:
            parser.error(f"window [{args.start}, {args.end}) selects no file pairs")
        problems = check_window_pairing(tel, prb)
        if problems:
            msg = "window pairing check failed:\n  " + "\n  ".join(problems)
            if not args.allow_unpaired_window:
                parser.error(msg + "\n(pass --allow-unpaired-window to proceed anyway)")
            print("  ! " + msg)
    print(
        f"  window '{args.label}': {len(tel.gps_paths)} telescope / "
        f"{len(prb.gps_paths)} probe file pairs"
    )

    # A mid-acquisition slice must be re-anchored before anything else touches
    # it -- otherwise stage 1 times both streams as though the window were the
    # start of the run, skewing them against each other by seconds and yielding
    # exactly zero coincidences with no error raised.  See `reanchor_window`.
    # Only the decode stage streams events, so only it needs (or should pay
    # the extra stage-1 pass for) re-anchoring; replay and plots work from the
    # cache, whose t_ns were already written with the corrected clock.
    if (start or end) and not args.no_reanchor and args.stage in ("decode", "all"):
        tel, prb, cal = reanchor_window(
            tel,
            prb,
            full_tel=full_tel,
            full_prb=full_prb,
            search_s=args.reanchor_search_s,
            measure=not args.trust_file_names,
        )
        if cal is not None:
            ranked = sorted(cal.counts.items(), key=lambda kv: -kv[1])[:3]
            print(
                "  clock calibration: shift "
                f"{cal.best_shift_ns / 1e9:+.0f} s -> {cal.best_count} raw "
                "coincidences  (top: "
                + ", ".join(f"{s:+d}s={n}" for s, n in ranked)
                + ")"
            )
            if not cal.is_confident:
                parser.error(
                    "window clock calibration is not conclusive -- the best "
                    "shift does not stand clearly above the others, so the "
                    "matches are consistent with accidentals. Widen "
                    "--reanchor-search-s, choose a window with more data, or "
                    "pass --no-reanchor if you know the slice is already "
                    "anchored."
                )
        print(f"  re-anchored to UTC {tel.utc0} (telescope stream start)")

    alignment, schedule, align_label = load_run_alignment(args, tel, z_tel)

    paths = {
        w: cache_path(out, args.label, args.tot_thresh, args.tot_weights, w)
        for w in widths
    }

    if args.stage in ("decode", "all"):
        for w in widths:
            print(f"  Tier A decode: max_cluster_width={w} -> {paths[w].name}")
            t0 = time.monotonic()
            cache = decode_pass(
                tel,
                prb,
                z_tel=z_tel,
                alignment=alignment,
                schedule=schedule,
                alignment_label=align_label,
                tot_thresh=args.tot_thresh,
                tot_weights=args.tot_weights,
                max_cluster_width=w,
                fibers_per_ribbon=args.fibers_per_ribbon,
                min_anchor_planes=args.decode_anchor,
                log_every=args.log_every,
            )
            cache.save(paths[w])
            print(
                f"    {len(cache)} clusters in {time.monotonic() - t0:.1f} s "
                f"-> {paths[w]}"
            )

    if args.stage in ("replay", "plots", "all"):
        caches = {}
        for w in widths:
            if not paths[w].exists():
                parser.error(f"{paths[w]} not found -- run --stage decode first")
            caches[_cache_tag(args.tot_thresh, args.tot_weights, w)] = (
                ClusterCache.load(paths[w])
            )

    if args.stage in ("replay", "all"):
        print("  Tier B replay grid ...")
        rows = run_grid(
            caches,
            chi2_grid=args.chi2_grid,
            anchor_grid=args.anchor_grid,
            max_resid_mm_grid=args.max_resid_mm_grid,
            rigidity_grid=args.rigidity_grid,
            off_probe_grid=args.off_probe_grid,
            mahal_grid=args.mahal_grid,
            probe_size_mm=probe_size_mm,
            min_fit=args.min_fit,
            alignment=alignment,
        )
        write_grid_csv(rows, out / f"scan_grid_{args.label}.csv")
        write_funnel_csv(rows, out / f"funnel_{args.label}.csv")
        print(f"    {len(rows)} grid points -> scan_grid_{args.label}.csv")

    if args.stage in ("plots", "all") and not args.no_plots:
        import scan_plots

        fig_dir = out / f"figures_{args.label}"
        grid_csv = out / f"scan_grid_{args.label}.csv"
        written = scan_plots.render_all(caches, grid_csv, fig_dir)

        # The footprint and per-bin figures need a fitted pose / a per-bin
        # replay, so they are driven here rather than from the grid CSV.
        if grid_csv.exists():
            best = max(
                scan_plots.load_grid(grid_csv),
                key=lambda r: r.get("signal_count") or -math.inf,
            )
            cfg = dict(
                chi2_track=best["chi2_track"],
                min_anchor_planes=int(best["min_anchor_planes"]),
                max_resid_mm=best["max_resid_mm"],
                max_rigidity_resid_mm=best["max_rigidity_resid_mm"],
                max_off_probe_mm=best["max_off_probe_mm"],
                mahal_cut=best["mahal_cut"],
                probe_size_mm=probe_size_mm,
                min_fit=args.min_fit,
            )
            cache = caches[best["cache"]]
            _row, pose = evaluate_with_pose(cache, alignment=alignment, **cfg)
            if pose is not None:
                written.append(
                    scan_plots.footprint(
                        pose,
                        fig_dir / f"footprint_{best['cache']}.png",
                        probe_size_mm,
                        title=f"best point: chi2={best['chi2_track']}, "
                        f"anchor>={best['min_anchor_planes']:g}",
                    )
                )
            series = {
                "best": bin_series(cache, alignment=alignment, **cfg),
                "shipped": bin_series(
                    cache,
                    alignment=alignment,
                    chi2_track=4.0,
                    min_anchor_planes=1,
                    max_rigidity_resid_mm=cfg["max_rigidity_resid_mm"],
                    max_off_probe_mm=cfg["max_off_probe_mm"],
                    probe_size_mm=probe_size_mm,
                    min_fit=args.min_fit,
                ),
            }
            written.append(
                scan_plots.anomaly_bins(series, fig_dir / "anomaly_bins.png")
            )
        print(f"    {len(written)} figures -> {fig_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
