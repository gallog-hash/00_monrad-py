"""
Stage 3 (part) — multi-candidate enumeration.

reconstruct_plane_candidates
    Per-plane candidate (x_mm, y_mm) lists for the Stage 5 combinatorial track
    search, instead of collapsing each plane to a single resolved Hit.
"""

import math
from pathlib import Path
from typing import Literal, NamedTuple

from ..timing import PosRef
from ..decoders.position import BinDecoder
from .hit import (
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

        cands_x = _axis_candidates_with_tot(x_or, x_counts, tot_weights)
        cands_y = _axis_candidates_with_tot(y_or, y_counts, tot_weights)

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
