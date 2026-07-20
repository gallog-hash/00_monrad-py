"""
Stage 3 (part) — single-hit position decoding.

decode_position(pos_ref, pos_paths, n_cols) -> list[Hit]
    one entry per column (detector plane)

Hit
    NamedTuple: (x_mm, y_mm, sigma_x, sigma_y, quality, candidates_x, candidates_y)
    quality: 'golden' | 'cluster' | 'unresolved' | 'invalid'
"""

import math
import struct
from pathlib import Path
from typing import Literal, NamedTuple

from ..timing import PosRef
from ..decoders.position import (
    BinDecoder,
    POS_COORD_MASK,
    POS_X_SHIFT,
    POS_HALF_BITS,
    split_half,
    combine_channel,
    split_channel,
)

_STRIP_MM = 10.0  # mm per channel strip — DESIGN.md §6.5

# Qualities that count as a usable hit for the pose fit.
GOOD_QUALITIES = ("golden", "cluster")


class Hit(NamedTuple):
    x_mm: float
    y_mm: float
    sigma_x: float
    sigma_y: float
    quality: Literal["golden", "cluster", "unresolved", "invalid"]
    # Candidate (centroid_ch, width) pairs for 'unresolved' hits, retained for
    # diagnostics (e.g. scripts/investigate_single_axis.py).  Stage 5
    # enumerates its own per-plane candidates via reconstruct_plane_candidates.
    # None for all other qualities.
    candidates_x: list[tuple[float, int]] | None = None
    candidates_y: list[tuple[float, int]] | None = None


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


def _bit_counts(words: list[int], col: int, n_cols: int) -> tuple[list[int], list[int]]:
    """
    Per-bit row counts (0..16) for one column's X/Y 20-bit fields across the
    16-row block.  bits 0-9 = ribbon, bits 10-19 = fiber.  Shared by
    _or_masks's tot_thresh>1 path, decode_position's tot_weights path, and
    reconstruct_plane_candidates's TOT scoring.
    """
    x_counts = [0] * 20
    y_counts = [0] * 20
    for row in range(16):
        w = words[row * n_cols + col]
        y_bits = w & POS_COORD_MASK
        x_bits = (w >> POS_X_SHIFT) & POS_COORD_MASK
        for bit in range(20):
            if (x_bits >> bit) & 1:
                x_counts[bit] += 1
            if (y_bits >> bit) & 1:
                y_counts[bit] += 1
    return x_counts, y_counts


def _or_masks(
    words: list[int],
    col: int,
    n_cols: int,
    tot_thresh: int = 1,
) -> tuple[int, int]:
    """
    OR the X/Y 20-bit fields for one column across the 16-row block,
    keeping only bits that fired in >= tot_thresh of the 16 rows.

    tot_thresh=1 is a plain bitwise OR (DESIGN.md §6.2); tot_thresh>1 filters
    single-row noise spikes via a per-bit TOT count, identical to the
    count-path in decode_position().
    """
    if tot_thresh <= 1:
        x_or = 0
        y_or = 0
        for row in range(16):
            w = words[row * n_cols + col]
            y_or |= w & POS_COORD_MASK
            x_or |= (w >> POS_X_SHIFT) & POS_COORD_MASK
        return x_or, y_or

    x_counts, y_counts = _bit_counts(words, col, n_cols)
    x_or = sum((1 << bit) for bit in range(20) if x_counts[bit] >= tot_thresh)
    y_or = sum((1 << bit) for bit in range(20) if y_counts[bit] >= tot_thresh)
    return x_or, y_or


def _tot_weighted_centroid(
    candidates: list[int],
    ribbon_counts: list[int],
    fiber_counts: list[int],
    n: int = POS_HALF_BITS,
) -> float:
    """
    Compute a TOT-weighted centroid over the candidate channel list.

    Each candidate ch = n*r + f gets weight = ribbon_counts[r] * fiber_counts[f].
    Falls back to the unweighted mean if all weights are zero.
    """
    rf = [split_channel(ch, n) for ch in candidates]
    weights = [ribbon_counts[r] * fiber_counts[f] for r, f in rf]
    total = sum(weights)
    if total == 0:
        return sum(candidates) / len(candidates)
    return sum(w * ch for w, ch in zip(weights, candidates)) / total


def _axis_candidates(field_or: int, n: int = POS_HALF_BITS) -> list[tuple[float, int]]:
    """
    Return candidate (centroid_ch, width) pairs for an axis that decoded as
    'unresolved'.  centroid_ch is in channel units; width is the number of
    combined channels in the hypothesis, used to compute sigma on selection.
    """
    fiber_half, ribbon_half = split_half(field_or)

    fcs = BinDecoder._find_clusters(fiber_half)
    rcs = BinDecoder._find_clusters(ribbon_half)

    candidates: list[tuple[float, int]] = []
    for rc in rcs:
        for fc in fcs:
            # Split the (ribbon × fiber) cross-product into maximal contiguous
            # channel runs — see _axis_candidates_with_tot for the rationale.
            # Adjacent ribbons are N apart, so e.g. fiber {3,4} × ribbon {2,3}
            # yields two candidates [23,24] and [33,34], not one width-4 blob
            # straddling the 25..32 gap.
            chs = sorted(combine_channel(r, f, n) for r in rc for f in fc)
            run = [chs[0]]
            for ch in chs[1:]:
                if ch == run[-1] + 1:
                    run.append(ch)
                else:
                    candidates.append((sum(run) / len(run), len(run)))
                    run = [ch]
            candidates.append((sum(run) / len(run), len(run)))
    return candidates


def _axis_candidates_with_tot(
    field_or: int,
    counts: list[int],
    tot_weights: bool = False,
    n: int = POS_HALF_BITS,
) -> list[tuple[float, int, int]]:
    """
    Like _axis_candidates, but also returns each candidate's TOT score:
    sum of ribbon_count * fiber_count (the _tot_weighted_centroid weighting
    convention) over its contributing (ribbon, fiber) pairs — a measure of
    how solidly each fired bit was seen across the 16-row block.

    counts is the 20-element per-bit row count for this field (ribbon 0-9,
    fiber 10-19), as returned by _bit_counts — always the raw hardware bit
    width, regardless of n (the combine factor).

    tot_weights : when True, each candidate's centroid is TOT-weighted by its
                  per-channel ribbon_count * fiber_count (same convention as
                  _tot_weighted_centroid / decode_position's tot_weights path),
                  falling back to the unweighted mean when all weights are
                  zero.  Width-1 candidates are unaffected.
    n           : fiber×ribbon combine factor for this detector (DESIGN.md
                  §2.4) — number of fiber positions actually wired per ribbon
                  channel.  Distinct from POS_HALF_BITS, the fixed raw bit
                  width used to slice counts into ribbon/fiber halves.
    """
    fiber_half, ribbon_half = split_half(field_or)
    ribbon_counts = counts[:POS_HALF_BITS]
    fiber_counts = counts[POS_HALF_BITS:]

    fcs = BinDecoder._find_clusters(fiber_half)
    rcs = BinDecoder._find_clusters(ribbon_half)

    candidates: list[tuple[float, int, int]] = []

    def _emit(run: list[int]) -> None:
        rf = [split_channel(c, n) for c in run]
        weights = [ribbon_counts[r] * fiber_counts[f] for r, f in rf]
        tot = sum(weights)
        if tot_weights and tot > 0:
            centroid = sum(w * c for w, c in zip(weights, run)) / tot
        else:
            centroid = sum(run) / len(run)
        candidates.append((centroid, len(run), tot))

    for rc in rcs:
        for fc in fcs:
            # The (ribbon × fiber) cross-product is not necessarily a single
            # hit: adjacent ribbons are N apart, so a multi-ribbon cluster only
            # forms one contiguous channel run when the fiber cluster spans the
            # full decade.  Otherwise the combined N*r+f channels break into
            # several gap-free runs, each a distinct candidate — e.g. fiber
            # {3,4} × ribbon {2,3} yields [23,24] and [33,34], not one width-4
            # blob straddling the 25..32 gap.  Split into maximal contiguous
            # runs (mirroring _reconstruct_coord's contiguity rule, but
            # enumerating each run instead of rejecting the whole pair).
            chs = sorted(combine_channel(r, f, n) for r in rc for f in fc)
            run = [chs[0]]
            for ch in chs[1:]:
                if ch == run[-1] + 1:
                    run.append(ch)
                else:
                    _emit(run)
                    run = [ch]
            _emit(run)
    return candidates


def _decode_axis(
    field_or: int,
    bit_counts: list[int] | None = None,
    n: int = POS_HALF_BITS,
    max_cluster_width: int | None = None,
) -> tuple[float, float, Literal["golden", "cluster", "unresolved"]]:
    """
    Decode one 20-bit fiber×ribbon field (already extracted from u64).

    bits  0– 9: ribbon mask
    bits 10–19: fiber mask

    Returns (centroid_ch, sigma_mm, quality).
    Implements DESIGN.md §6.4 steps 1–5.

    bit_counts : optional 20-element list of per-bit TOT counts
                 (bit_counts[0..9] = ribbon, bit_counts[10..19] = fiber).
                 When provided, cluster centroids are TOT-weighted.
    n          : fiber×ribbon combine factor for this detector (DESIGN.md
                 §2.4).
    max_cluster_width : if set, a resolved candidate whose merged-channel
                 width exceeds this cap is treated as too ambiguous and
                 reported as 'unresolved' instead of 'cluster' (None = no
                 cap, current behaviour).
    """
    fiber_half, ribbon_half = split_half(field_or)

    fcs = BinDecoder._find_clusters(fiber_half)
    rcs = BinDecoder._find_clusters(ribbon_half)

    res = BinDecoder._reconstruct_coord(fcs, rcs, n)
    if res is not None:
        centroid, candidates = res
        width = len(candidates)
        if max_cluster_width is not None and width > max_cluster_width:
            return 0.0, 0.0, "unresolved"
        sigma = (_STRIP_MM * width) / math.sqrt(12)
        quality = "golden" if width == 1 else "cluster"
        if bit_counts is not None and width > 1:
            ribbon_counts = bit_counts[:POS_HALF_BITS]
            fiber_counts = bit_counts[POS_HALF_BITS:]
            centroid = _tot_weighted_centroid(
                candidates, ribbon_counts, fiber_counts, n
            )
        return centroid, sigma, quality

    return 0.0, 0.0, "unresolved"


def decode_position(
    pos_ref: PosRef,
    pos_paths: list[Path],
    n_cols: int,
    tot_thresh: int = 1,
    tot_weights: bool = False,
    n_fibers_per_ribbon: int = POS_HALF_BITS,
    max_cluster_width: int | None = None,
) -> list[Hit]:
    """
    Decode one event's position from its PosRef.

    Implements DESIGN.md §6.1–§6.5 using the fiber×ribbon logic from
    decoders/position.py (BinDecoder._find_clusters, _reconstruct_coord,
    _is_valid).  The PosRef is received directly from the stage-1 stream
    (DESIGN.md §4.5) — no pos_map lookup.

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
    n_fibers_per_ribbon : fiber×ribbon combine factor for this detector
                   (DESIGN.md §2.4) — number of fiber positions actually
                   wired per ribbon channel.  Defaults to the raw hardware
                   width (10); probes may wire fewer.
    max_cluster_width : if set, an axis whose resolved candidate width
                   exceeds this cap is reported as 'unresolved' instead of
                   'cluster' (None = no cap, current behaviour).

    Returns
    -------
    list of length n_cols.  Each element is a Hit (always present,
    with quality 'invalid' or 'unresolved' when reconstruction fails)
    so callers can use a plain membership test on quality strings
    rather than checking for None.
    """
    words = _read_block(pos_paths, pos_ref, n_cols)

    hits: list[Hit] = []
    for col in range(n_cols):
        if not tot_weights:
            # Fast path: OR (with threshold if requested) — DESIGN.md §6.2
            x_or, y_or = _or_masks(words, col, n_cols, tot_thresh)
            x_counts_col = None
            y_counts_col = None
        else:
            # Count-path: accumulate per-bit TOT counts across 16 rows, needed
            # for weighted centroids (_tot_weighted_centroid below).
            x_counts_col, y_counts_col = _bit_counts(words, col, n_cols)
            # Apply threshold: keep bit only if count >= tot_thresh.
            x_or = sum(
                (1 << bit) for bit in range(20) if x_counts_col[bit] >= tot_thresh
            )
            y_or = sum(
                (1 << bit) for bit in range(20) if y_counts_col[bit] >= tot_thresh
            )

        # Validity prefilter — DESIGN.md §6.3
        valid, _ = BinDecoder._is_valid(x_or, y_or)
        if not valid:
            hits.append(Hit(0.0, 0.0, 0.0, 0.0, "invalid"))
            continue

        cx, sx, qx = _decode_axis(
            x_or,
            bit_counts=x_counts_col,
            n=n_fibers_per_ribbon,
            max_cluster_width=max_cluster_width,
        )
        cy, sy, qy = _decode_axis(
            y_or,
            bit_counts=y_counts_col,
            n=n_fibers_per_ribbon,
            max_cluster_width=max_cluster_width,
        )

        if qx == "unresolved" or qy == "unresolved":
            # An axis that DID resolve is kept as a one-element candidate at
            # its own centroid (width recovered from sigma); an axis that
            # failed contributes its real multi-candidate hypotheses.  These
            # per-axis lists are retained for diagnostics (DESIGN.md §6.4);
            # Stage 5 enumerates its own candidates via
            # reconstruct_plane_candidates rather than reading them here.
            cands_x = (
                _axis_candidates(x_or, n=n_fibers_per_ribbon)
                if qx == "unresolved"
                else [(cx, max(1, round(sx * math.sqrt(12) / _STRIP_MM)))]
            )
            cands_y = (
                _axis_candidates(y_or, n=n_fibers_per_ribbon)
                if qy == "unresolved"
                else [(cy, max(1, round(sy * math.sqrt(12) / _STRIP_MM)))]
            )
            hits.append(Hit(0.0, 0.0, 0.0, 0.0, "unresolved", cands_x, cands_y))
            continue

        # Channel → physical coordinate — DESIGN.md §6.5
        x_mm = (cx + 0.5) * _STRIP_MM
        y_mm = (cy + 0.5) * _STRIP_MM
        quality = "golden" if (qx == "golden" and qy == "golden") else "cluster"
        hits.append(Hit(x_mm, y_mm, sx, sy, quality))

    return hits
