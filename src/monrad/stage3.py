"""
Stage 3 — position decoding.

Public API
----------
decode_position(pos_ref, pos_paths, n_cols)
    -> list[Hit | None]   one entry per column (detector plane)

Hit
    NamedTuple: (x_mm, y_mm, sigma_x, sigma_y, quality)
    quality: 'golden' | 'cluster' | 'efficiency' | 'unresolved' | 'invalid'
"""

import math
import struct
from pathlib import Path
from typing import Literal, NamedTuple, Sequence

from .stage1 import PosRef
from .decoders.position import BinDecoder

_STRIP_MM = 10.0  # mm per channel strip — DESIGN.md §5.4
_N = 10  # fiber × ribbon encoding multiplier
_FULL_HALF = 0x3FF  # 10-bit all-ones — a saturated fiber/ribbon half

# Qualities that count as a usable hit for the pose fit.  'efficiency' hits are
# single-channel dropouts recovered from the two-plane track projection by
# recover_efficiency_hits().
GOOD_QUALITIES = ("golden", "cluster", "efficiency")


class Hit(NamedTuple):
    x_mm: float
    y_mm: float
    sigma_x: float
    sigma_y: float
    quality: Literal["golden", "cluster", "efficiency", "unresolved", "invalid"]
    # Candidate (centroid_ch, width) pairs for 'unresolved' hits — used by
    # disambiguate_telescope_hits().  None for all other qualities.
    candidates_x: list[tuple[float, int]] | None = None
    candidates_y: list[tuple[float, int]] | None = None
    # Raw OR'd 20-bit X / Y fields, carried on 'unresolved' and 'invalid' hits
    # so recover_efficiency_hits() can inspect which ribbon/fiber bits fired
    # when synthesizing a missing half from the track projection.
    x_or: int | None = None
    y_or: int | None = None


def _read_block(
    pos_paths: list[Path],
    pos_ref: PosRef,
    n_cols: int,
) -> list[int]:
    """
    Read the 16 × n_cols u64 words for one event, handling split blocks.

    Returns a flat list of 16 * n_cols ints in row-major order:
      [col0_row0, col1_row0, …, col(n-1)_row0, col0_row1, …]
    """
    _HDR = 8  # 4-byte n_rows + 4-byte n_cols
    _WORD = 8  # bytes per u64

    def _rows(path: Path, row_start: int, n_rows: int) -> list[int]:
        offset = _HDR + row_start * n_cols * _WORD
        with open(path, "rb") as fh:
            fh.seek(offset)
            raw = fh.read(n_rows * n_cols * _WORD)
        words = [struct.unpack_from("<Q", raw, i)[0] for i in range(0, len(raw), _WORD)]
        # Pad with zeros if the read fell short of EOF (GPS/pos count mismatch).
        # A zero word has ribbon=0, which _is_valid() rejects as 'invalid'.
        expected = n_rows * n_cols
        if len(words) < expected:
            words += [0] * (expected - len(words))
        return words

    if pos_ref.split_rows == 0:
        return _rows(pos_paths[pos_ref.file_idx], pos_ref.row_offset, 16)

    head = _rows(
        pos_paths[pos_ref.file_idx],
        pos_ref.row_offset,
        pos_ref.split_rows,
    )
    tail = _rows(
        pos_paths[pos_ref.file_idx + 1],
        0,
        16 - pos_ref.split_rows,
    )
    return head + tail


def _or_masks(
    words: list[int],
    col: int,
    n_cols: int,
    tot_thresh: int = 1,
) -> tuple[int, int]:
    """
    OR the X/Y 20-bit fields for one column across the 16-row block,
    keeping only bits that fired in >= tot_thresh of the 16 rows.

    tot_thresh=1 is a plain bitwise OR (DESIGN.md §5.1); tot_thresh>1 filters
    single-row noise spikes via a per-bit TOT count, identical to the
    count-path in decode_position().
    """
    if tot_thresh <= 1:
        x_or = 0
        y_or = 0
        for row in range(16):
            w = words[row * n_cols + col]
            y_or |= w & 0xFFFFF
            x_or |= (w >> 32) & 0xFFFFF
        return x_or, y_or

    x_counts = [0] * 20
    y_counts = [0] * 20
    for row in range(16):
        w = words[row * n_cols + col]
        y_bits = w & 0xFFFFF
        x_bits = (w >> 32) & 0xFFFFF
        for bit in range(20):
            if (x_bits >> bit) & 1:
                x_counts[bit] += 1
            if (y_bits >> bit) & 1:
                y_counts[bit] += 1
    x_or = sum((1 << bit) for bit in range(20) if x_counts[bit] >= tot_thresh)
    y_or = sum((1 << bit) for bit in range(20) if y_counts[bit] >= tot_thresh)
    return x_or, y_or


def _tot_weighted_centroid(
    candidates: list[int],
    ribbon_counts: list[int],
    fiber_counts: list[int],
) -> float:
    """
    Compute a TOT-weighted centroid over the candidate channel list.

    Each candidate ch = 10*r + f gets weight = ribbon_counts[r] * fiber_counts[f].
    Falls back to the unweighted mean if all weights are zero.
    """
    weights = [ribbon_counts[ch // _N] * fiber_counts[ch % _N] for ch in candidates]
    total = sum(weights)
    if total == 0:
        return sum(candidates) / len(candidates)
    return sum(w * ch for w, ch in zip(weights, candidates)) / total


def _axis_candidates(field_or: int) -> list[tuple[float, int]]:
    """
    Return candidate (centroid_ch, width) pairs for an axis that decoded as
    'unresolved'.  centroid_ch is in channel units; width is the number of
    combined channels in the hypothesis, used to compute sigma on selection.
    """
    fiber_half = (field_or >> _N) & 0x3FF
    ribbon_half = field_or & 0x3FF

    fcs = BinDecoder._find_clusters(fiber_half)
    rcs = BinDecoder._find_clusters(ribbon_half)

    candidates: list[tuple[float, int]] = []
    for rc in rcs:
        for fc in fcs:
            chs = [_N * r + f for r in rc for f in fc]
            candidates.append((sum(chs) / len(chs), len(chs)))
    return candidates


def _decode_axis(
    field_or: int,
    bit_counts: list[int] | None = None,
) -> tuple[float, float, Literal["golden", "cluster", "unresolved"]]:
    """
    Decode one 20-bit fiber×ribbon field (already extracted from u64).

    bits  0– 9: ribbon mask
    bits 10–19: fiber mask

    Returns (centroid_ch, sigma_mm, quality).
    Implements DESIGN.md §5.3 steps 1–5.

    bit_counts : optional 20-element list of per-bit TOT counts
                 (bit_counts[0..9] = ribbon, bit_counts[10..19] = fiber).
                 When provided, cluster centroids are TOT-weighted.
    """
    fiber_half = (field_or >> _N) & 0x3FF
    ribbon_half = field_or & 0x3FF

    fcs = BinDecoder._find_clusters(fiber_half)
    rcs = BinDecoder._find_clusters(ribbon_half)

    res = BinDecoder._reconstruct_coord(fcs, rcs, _N)
    if res is not None:
        centroid, candidates = res
        width = len(candidates)
        sigma = (_STRIP_MM * width) / math.sqrt(12)
        quality = "golden" if width == 1 else "cluster"
        if bit_counts is not None and width > 1:
            ribbon_counts = bit_counts[:_N]
            fiber_counts = bit_counts[_N:]
            centroid = _tot_weighted_centroid(candidates, ribbon_counts, fiber_counts)
        return centroid, sigma, quality

    return 0.0, 0.0, "unresolved"


def decode_position(
    pos_ref: PosRef,
    pos_paths: list[Path],
    n_cols: int,
    tot_thresh: int = 1,
    tot_weights: bool = False,
) -> list[Hit | None]:
    """
    Decode one event's position from its PosRef.

    Implements DESIGN.md §5.1–§5.4 using the fiber×ribbon logic from
    decoders/position.py (BinDecoder._find_clusters, _reconstruct_coord,
    _is_valid).  The PosRef is received directly from the stage-1 stream
    (DESIGN_UPDATE.md §4) — no pos_map lookup.

    Parameters
    ----------
    pos_ref      : location of the event's 16-row block on disk
    pos_paths    : *.bin file paths for this detector, in acquisition
                   order (indexed by pos_ref.file_idx)
    n_cols       : number of position-sensitive planes in the detector
                   (1 for a probe, 3 for the telescope)
    tot_thresh   : minimum number of the 16 rows in which a bit must fire
                   to be kept in the OR mask (1 = current behaviour; 2–4
                   filter single-row noise spikes without affecting real
                   signal hits that persist across many rows).
    tot_weights  : if True, weight cluster centroids by per-bit TOT counts
                   (each bit's weight = number of rows it fired in).  Has
                   no effect on golden hits (width=1).  Automatically
                   enables per-bit counting even when tot_thresh=1.

    Returns
    -------
    list of length n_cols.  Each element is a Hit (always present,
    with quality 'invalid' or 'unresolved' when reconstruction fails)
    so callers can use a plain membership test on quality strings
    rather than checking for None.
    """
    words = _read_block(pos_paths, pos_ref, n_cols)

    hits: list[Hit | None] = []
    for col in range(n_cols):
        if not tot_weights:
            # Fast path: OR (with threshold if requested) — DESIGN.md §5.1
            x_or, y_or = _or_masks(words, col, n_cols, tot_thresh)
            x_counts_col = None
            y_counts_col = None
        else:
            # Count-path: accumulate per-bit TOT counts across 16 rows, needed
            # for weighted centroids (_tot_weighted_centroid below).
            x_counts_col = [0] * 20
            y_counts_col = [0] * 20
            for row in range(16):
                w = words[row * n_cols + col]
                y_bits = w & 0xFFFFF
                x_bits = (w >> 32) & 0xFFFFF
                for bit in range(20):
                    if (x_bits >> bit) & 1:
                        x_counts_col[bit] += 1
                    if (y_bits >> bit) & 1:
                        y_counts_col[bit] += 1
            # Apply threshold: keep bit only if count >= tot_thresh.
            x_or = sum(
                (1 << bit) for bit in range(20) if x_counts_col[bit] >= tot_thresh
            )
            y_or = sum(
                (1 << bit) for bit in range(20) if y_counts_col[bit] >= tot_thresh
            )

        # Validity prefilter — DESIGN.md §5.2
        valid, _ = BinDecoder._is_valid(x_or, y_or)
        if not valid:
            # Carry the raw masks so recover_efficiency_hits() can synthesize a
            # missing ribbon/fiber half (e.g. X_ribbon=0 fiber-only dropouts).
            hits.append(Hit(0.0, 0.0, 0.0, 0.0, "invalid", x_or=x_or, y_or=y_or))
            continue

        cx, sx, qx = _decode_axis(x_or, bit_counts=x_counts_col)
        cy, sy, qy = _decode_axis(y_or, bit_counts=y_counts_col)

        if qx == "unresolved" or qy == "unresolved":
            # An axis that DID resolve is kept as a one-element candidate at
            # its own centroid (width recovered from sigma) so that a plane
            # which failed on only one axis can still be recovered by
            # disambiguate_telescope_hits(): the known axis is matched
            # trivially and only the failed axis is filled from the two-plane
            # projection.  An axis that failed contributes its real candidate
            # hypotheses.  (DESIGN.md §6.4 / §6.6.)
            cands_x = (
                _axis_candidates(x_or)
                if qx == "unresolved"
                else [(cx, max(1, round(sx * math.sqrt(12) / _STRIP_MM)))]
            )
            cands_y = (
                _axis_candidates(y_or)
                if qy == "unresolved"
                else [(cy, max(1, round(sy * math.sqrt(12) / _STRIP_MM)))]
            )
            hits.append(
                Hit(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    "unresolved",
                    cands_x,
                    cands_y,
                    x_or=x_or,
                    y_or=y_or,
                )
            )
            continue

        # Channel → physical coordinate — DESIGN.md §5.4
        x_mm = (cx + 0.5) * _STRIP_MM
        y_mm = (cy + 0.5) * _STRIP_MM
        quality = "golden" if (qx == "golden" and qy == "golden") else "cluster"
        hits.append(Hit(x_mm, y_mm, sx, sy, quality))

    return hits


class PlaneCandidate(NamedTuple):
    x_mm: float
    y_mm: float
    sigma_x: float
    sigma_y: float


def reconstruct_plane_candidates(
    pos_ref: PosRef,
    pos_paths: list[Path],
    n_cols: int,
    max_per_plane: int = 8,
    tot_thresh: int = 1,
) -> list[list[PlaneCandidate]]:
    """
    Enumerate per-plane candidate (x_mm, y_mm) positions for one event,
    instead of collapsing each plane to a single resolved Hit.

    Each plane's candidate list is the Cartesian product of its X-axis and
    Y-axis candidates (`_axis_candidates`): one candidate for a golden/
    cluster axis, the full ribbon×fiber mirror-fold cross-product for an
    ambiguous one.  An invalid plane (saturated half, or zero ribbon
    channel) yields an empty list.  The product is capped at
    `max_per_plane`, keeping the most compact candidates first (smallest
    width_x + width_y, ties broken by channel) — used by the Stage 5
    combinatorial track finder in place of a single resolved Hit per plane.

    tot_thresh mirrors decode_position's OR-mask threshold so the masks fed
    to candidate enumeration match the resolved decode path exactly.
    TOT-weighting (tot_weights) is deferred — not supported here.
    """
    words = _read_block(pos_paths, pos_ref, n_cols)

    planes: list[list[PlaneCandidate]] = []
    for col in range(n_cols):
        x_or, y_or = _or_masks(words, col, n_cols, tot_thresh)

        valid, _ = BinDecoder._is_valid(x_or, y_or)
        if not valid:
            planes.append([])
            continue

        cands_x = _axis_candidates(x_or)
        cands_y = _axis_candidates(y_or)

        points = [(wx + wy, cx, cy, wx, wy) for cx, wx in cands_x for cy, wy in cands_y]
        points.sort(key=lambda p: p[:3])

        planes.append(
            [
                PlaneCandidate(
                    x_mm=(cx + 0.5) * _STRIP_MM,
                    y_mm=(cy + 0.5) * _STRIP_MM,
                    sigma_x=(_STRIP_MM * wx) / math.sqrt(12),
                    sigma_y=(_STRIP_MM * wy) / math.sqrt(12),
                )
                for _, cx, cy, wx, wy in points[:max_per_plane]
            ]
        )

    return planes


def disambiguate_telescope_hits(
    hits: list[Hit],
    z_tel: Sequence[float],
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


def _recover_axis_efficiency(
    field_or: int,
    coord_pred_corr: float,
    delta: float,
    tol: float,
) -> tuple[float, int] | None:
    """
    Synthesize a single-channel coordinate for an axis where exactly one of the
    fiber / ribbon halves fired ("option (0)" efficiency dropout recovery).

    A telescope channel is ch = 10·ribbon + fiber and needs both a fired ribbon
    bit and a fired fiber bit.  When only one half fired, _decode_axis() yields
    no candidate, but the other two planes still predict where the track crosses
    this plane.  We anchor to the half that *did* fire and read the missing half
    off the prediction:

      - ribbon-only: for each fired ribbon r, fiber f = clamp(round(ch_pred − 10r))
      - fiber-only:  for each fired fiber f, ribbon r = clamp(round((ch_pred − f)/10))

    The prediction is supplied in the alignment-corrected frame (coord_pred_corr);
    `delta` is this plane's offset so the raw channel position can be compared in
    the same frame: ch_pred_raw = (coord_pred_corr + delta)/strip − 0.5.

    Returns (coord_mm_raw, width=1) for the candidate nearest the prediction when
    it lies within `tol` (in the corrected frame), else None.  Saturated halves
    (all ten bits set) are rejected.
    """
    fiber_half = (field_or >> _N) & _FULL_HALF
    ribbon_half = field_or & _FULL_HALF
    if fiber_half == _FULL_HALF or ribbon_half == _FULL_HALF:
        return None

    fired_ribbons = [b for b in range(_N) if (ribbon_half >> b) & 1]
    fired_fibers = [b for b in range(_N) if (fiber_half >> b) & 1]

    ch_pred_raw = (coord_pred_corr + delta) / _STRIP_MM - 0.5

    if fired_ribbons and not fired_fibers:
        cands = [
            _N * r + min(9, max(0, round(ch_pred_raw - _N * r))) for r in fired_ribbons
        ]
    elif fired_fibers and not fired_ribbons:
        cands = [
            _N * min(9, max(0, round((ch_pred_raw - f) / _N))) + f for f in fired_fibers
        ]
    else:
        return None

    def _corr_dist(ch: int) -> float:
        return abs((ch + 0.5) * _STRIP_MM - delta - coord_pred_corr)

    best_ch = min(cands, key=_corr_dist)
    if _corr_dist(best_ch) <= tol:
        return (best_ch + 0.5) * _STRIP_MM, 1
    return None


def _axis_value(
    field_or: int,
    coord_pred_corr: float,
    delta: float,
    tol: float,
) -> tuple[float, int, bool] | None:
    """
    Resolve one axis of an efficiency-recovery candidate plane.

    Returns (coord_mm_raw, width, used_efficiency_fill) or None.  The axis is
    resolved by, in order: (a) rejecting a saturated half, (b) using the normal
    _decode_axis() result if it already resolves golden/cluster, else
    (c) _recover_axis_efficiency() synthesizing the missing half.
    """
    fiber_half = (field_or >> _N) & _FULL_HALF
    ribbon_half = field_or & _FULL_HALF
    if fiber_half == _FULL_HALF or ribbon_half == _FULL_HALF:
        return None

    centroid, sigma, quality = _decode_axis(field_or)
    if quality in ("golden", "cluster"):
        width = max(1, round(sigma * math.sqrt(12) / _STRIP_MM))
        return (centroid + 0.5) * _STRIP_MM, width, False

    rec = _recover_axis_efficiency(field_or, coord_pred_corr, delta, tol)
    if rec is None:
        return None
    coord_mm_raw, width = rec
    return coord_mm_raw, width, True


def recover_efficiency_hits(
    hits: list[Hit],
    z_tel: Sequence[float],
    offsets: Sequence[tuple[float, float]] | None = None,
) -> list[Hit]:
    """
    Recover single-channel efficiency dropouts ("option (0)").

    For each plane k whose quality is 'unresolved' or 'invalid' (and which still
    carries its raw OR masks), if the *other two* planes are 'golden'/'cluster',
    fit a straight line through them, project to plane k, and resolve each axis
    via _axis_value().  The plane is upgraded to quality 'efficiency' only when
    both axes resolve **and at least one of them used the synthesized fill** (a
    plane that resolves cleanly on both axes would already be golden/cluster).

    Frame handling mirrors disambiguate_telescope_hits(): predictions and the
    acceptance test use the alignment-corrected frame (coord − delta) when
    `offsets` is given; the returned hit carries the raw channel position.  Run
    this *after* disambiguate_telescope_hits() so two-plane recoveries are
    available as references.  References are only ever 'golden'/'cluster', so a
    recovered plane is never used to predict another.

    Only applied to 3-plane inputs; returns hits unchanged otherwise.
    """
    if len(hits) != 3:
        return hits

    _MATCH_TOL = 1.5 * _STRIP_MM  # acceptance window in mm (matches disambiguate)
    if offsets is None:
        dx = (0.0, 0.0, 0.0)
        dy = (0.0, 0.0, 0.0)
    else:
        dx = tuple(o[0] for o in offsets)
        dy = tuple(o[1] for o in offsets)

    result = list(hits)
    for k in range(3):
        hit_k = hits[k]
        if hit_k.quality not in ("unresolved", "invalid"):
            continue
        if hit_k.x_or is None or hit_k.y_or is None:
            continue

        j1, j2 = [j for j in range(3) if j != k]
        ref1, ref2 = hits[j1], hits[j2]
        if ref1.quality not in ("golden", "cluster"):
            continue
        if ref2.quality not in ("golden", "cluster"):
            continue

        t = (z_tel[k] - z_tel[j1]) / (z_tel[j2] - z_tel[j1])
        rx1, rx2 = ref1.x_mm - dx[j1], ref2.x_mm - dx[j2]
        ry1, ry2 = ref1.y_mm - dy[j1], ref2.y_mm - dy[j2]
        x_pred = rx1 + t * (rx2 - rx1)
        y_pred = ry1 + t * (ry2 - ry1)

        res_x = _axis_value(hit_k.x_or, x_pred, dx[k], _MATCH_TOL)
        res_y = _axis_value(hit_k.y_or, y_pred, dy[k], _MATCH_TOL)
        if res_x is None or res_y is None:
            continue
        x_mm, width_x, eff_x = res_x
        y_mm, width_y, eff_y = res_y
        if not (eff_x or eff_y):
            continue

        sigma_x = (_STRIP_MM * width_x) / math.sqrt(12)
        sigma_y = (_STRIP_MM * width_y) / math.sqrt(12)
        result[k] = Hit(x_mm, y_mm, sigma_x, sigma_y, "efficiency")

    return result
