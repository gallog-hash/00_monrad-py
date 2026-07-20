"""
Stage 3 (part) — multi-candidate enumeration and telescope disambiguation.

reconstruct_plane_candidates
    Per-plane candidate (x_mm, y_mm) lists for the Stage 5 combinatorial track
    search, instead of collapsing each plane to a single resolved Hit.
disambiguate_telescope_hits
    Two-plane linear recovery of 'unresolved' telescope planes.
"""

import math
from pathlib import Path
from typing import Literal, NamedTuple, Sequence

import numpy as np

from ..timing import PosRef
from ..decoders.position import BinDecoder, POS_HALF_BITS
from .hit import (
    Hit,
    _STRIP_MM,
    _axis_candidates_with_tot,
    _bit_counts,
    _read_block,
)


class PlaneCandidate(NamedTuple):
    x_mm: float
    y_mm: float
    sigma_x: float
    sigma_y: float
    quality: Literal["golden", "cluster"]
    tot_x: int  # ribbon_count * fiber_count summed over the X axis's bits
    tot_y: int  # same, for the Y axis


def reconstruct_plane_candidates(
    pos_ref: PosRef,
    pos_paths: list[Path],
    n_cols: int,
    max_per_plane: int = 16,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    n_fibers_per_ribbon: int = POS_HALF_BITS,
    max_cluster_width: int | None = None,
) -> list[list[PlaneCandidate]]:
    """
    Enumerate per-plane candidate (x_mm, y_mm) positions for one event,
    instead of collapsing each plane to a single resolved Hit.

    Each plane's candidate list is the Cartesian product of its X-axis and
    Y-axis candidates (`_axis_candidates_with_tot`): one candidate for a
    golden/cluster axis, the full ribbon×fiber mirror-fold cross-product for
    an ambiguous one.  An invalid plane (saturated half, or zero ribbon
    channel) yields an empty list.  The product is capped at
    `max_per_plane`, keeping the most compact candidates first (smallest
    width_x + width_y, ties broken by channel) — used by the Stage 5
    combinatorial track finder in place of a single resolved Hit per plane.

    The default cap of 16 is the worst-case mirror-fold count: each axis can
    fold into at most 2 ribbon × 2 fiber = 4 candidates, so a plane folded on
    both axes yields 4 × 4 = 16 (DESIGN.md §10).  A smaller cap can silently
    drop the true candidate when every candidate is equally compact.

    Each candidate carries `quality` ("golden" if both axes are width 1,
    else "cluster") and `tot_x`/`tot_y` (per-axis TOT score, see
    _axis_candidates_with_tot) so callers can report the signal strength and
    resolved-vs-cluster status of whichever candidate the Stage 5
    combinatorial search ultimately picks.

    tot_thresh mirrors decode_position's OR-mask threshold so the masks fed
    to candidate enumeration match the resolved decode path exactly. Per-bit
    counts are always computed (regardless of tot_thresh) to produce the TOT
    scores above; this is the same count-path decode_position's tot_weights
    uses, just unconditional here.

    tot_weights mirrors decode_position's tot_weights: when True each
    candidate centroid is TOT-weighted by its per-bit row counts (no effect
    on width-1 candidates).  Without it the telescope path would silently
    ignore the pipeline's --tot-weights flag that the probe decode honours.

    n_fibers_per_ribbon : fiber×ribbon combine factor for this detector
    (DESIGN.md §2.4).  Defaults to the raw hardware width (10); callers
    decoding a probe wired with fewer fiber positions pass its actual N.

    max_cluster_width : if set, drop any (x, y) candidate pair whose x-width
    or y-width exceeds this cap before the max_per_plane slice — an
    over-wide candidate is too ambiguous to trust even as a fallback.
    None (default) applies no cap (current behaviour).
    """
    words = _read_block(pos_paths, pos_ref, n_cols)

    planes: list[list[PlaneCandidate]] = []
    for col in range(n_cols):
        x_counts, y_counts = _bit_counts(words, col, n_cols)
        x_or = sum((1 << bit) for bit in range(20) if x_counts[bit] >= tot_thresh)
        y_or = sum((1 << bit) for bit in range(20) if y_counts[bit] >= tot_thresh)

        valid, _ = BinDecoder._is_valid(x_or, y_or)
        if not valid:
            planes.append([])
            continue

        cands_x = _axis_candidates_with_tot(
            x_or, x_counts, tot_weights, n_fibers_per_ribbon
        )
        cands_y = _axis_candidates_with_tot(
            y_or, y_counts, tot_weights, n_fibers_per_ribbon
        )
        if max_cluster_width is not None:
            cands_x = [c for c in cands_x if c[1] <= max_cluster_width]
            cands_y = [c for c in cands_y if c[1] <= max_cluster_width]

        points = [
            (wx + wy, cx, cy, wx, wy, tx, ty)
            for cx, wx, tx in cands_x
            for cy, wy, ty in cands_y
        ]
        points.sort(key=lambda p: p[:3])

        planes.append(
            [
                PlaneCandidate(
                    x_mm=(cx + 0.5) * _STRIP_MM,
                    y_mm=(cy + 0.5) * _STRIP_MM,
                    sigma_x=(_STRIP_MM * wx) / math.sqrt(12),
                    sigma_y=(_STRIP_MM * wy) / math.sqrt(12),
                    quality="golden" if wx == 1 and wy == 1 else "cluster",
                    tot_x=tx,
                    tot_y=ty,
                )
                for _, cx, cy, wx, wy, tx, ty in points[:max_per_plane]
            ]
        )

    return planes


def disambiguate_telescope_hits(
    hits: list[Hit],
    z_tel: Sequence[float] | np.ndarray,
    offsets: Sequence[tuple[float, float]] | None = None,
) -> list[Hit]:
    """
    Replace 'unresolved' hits with 'cluster' hits using a two-plane linear
    predictor, when a candidate exists within 1.5 strips (15 mm).

    For each plane k independently: if the other 2 planes both have quality
    'golden' or 'cluster', fit a straight line through them and predict the
    hit position at plane k.  If plane k is 'unresolved' with a non-empty
    candidate list and the nearest candidate is within 1.5 strips, the hit
    is replaced with quality 'cluster'.  Both X and Y axes must be resolved
    for the quality to change.

    Only applied to 3-plane inputs; returns hits unchanged otherwise.

    offsets : optional per-plane (delta_x, delta_y) alignment offsets in mm.
              When given, the two-plane prediction and the candidate-distance
              test are evaluated in the alignment-corrected frame
              (coord - delta), so a plane's recovery is not biased by the
              telescope's internal misalignment.  The returned hit always
              carries the *raw* candidate position; callers apply the same
              offset downstream.  Defaults to zero offsets (raw frame), which
              reproduces the Stage 4 behaviour exactly.
    """
    if len(hits) != 3:
        return hits

    _MATCH_TOL = 1.5 * _STRIP_MM  # acceptance window in mm
    if offsets is None:
        dx = (0.0, 0.0, 0.0)
        dy = (0.0, 0.0, 0.0)
    else:
        dx = tuple(o[0] for o in offsets)
        dy = tuple(o[1] for o in offsets)

    result = list(hits)
    for k in range(3):
        hit_k = hits[k]
        if hit_k.quality != "unresolved":
            continue

        j1, j2 = [j for j in range(3) if j != k]
        ref1, ref2 = hits[j1], hits[j2]
        if ref1.quality not in ("golden", "cluster"):
            continue
        if ref2.quality not in ("golden", "cluster"):
            continue

        t = (z_tel[k] - z_tel[j1]) / (z_tel[j2] - z_tel[j1])
        # Predict in the alignment-corrected frame (coord - delta).  With the
        # default zero offsets this is identical to the raw-frame prediction.
        rx1, rx2 = ref1.x_mm - dx[j1], ref2.x_mm - dx[j2]
        ry1, ry2 = ref1.y_mm - dy[j1], ref2.y_mm - dy[j2]
        x_pred = rx1 + t * (rx2 - rx1)
        y_pred = ry1 + t * (ry2 - ry1)

        new_x = new_y = None
        width_x = width_y = 1
        # Candidate positions are compared in the same corrected frame
        # (raw channel position - delta[k]); the stored hit keeps raw position.
        if hit_k.candidates_x:
            best_ch, best_w = min(
                hit_k.candidates_x,
                key=lambda cw: abs((cw[0] + 0.5) * _STRIP_MM - dx[k] - x_pred),
            )
            if abs((best_ch + 0.5) * _STRIP_MM - dx[k] - x_pred) <= _MATCH_TOL:
                new_x = (best_ch + 0.5) * _STRIP_MM
                width_x = best_w

        if hit_k.candidates_y:
            best_ch, best_w = min(
                hit_k.candidates_y,
                key=lambda cw: abs((cw[0] + 0.5) * _STRIP_MM - dy[k] - y_pred),
            )
            if abs((best_ch + 0.5) * _STRIP_MM - dy[k] - y_pred) <= _MATCH_TOL:
                new_y = (best_ch + 0.5) * _STRIP_MM
                width_y = best_w

        if new_x is not None and new_y is not None:
            sigma_x = (_STRIP_MM * width_x) / math.sqrt(12)
            sigma_y = (_STRIP_MM * width_y) / math.sqrt(12)
            result[k] = Hit(new_x, new_y, sigma_x, sigma_y, "cluster")

    return result
