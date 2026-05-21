#!/usr/bin/env python3
"""
Diagnose hit-pattern quality in telescope/probe *.bin position files.

Aggregates all *.bin files in a directory (or analyses a single file) and
reports, per plane and axis (fiber_X, ribbon_X, fiber_Y, ribbon_Y):

  mean popcount     — average number of bits set after 16-row OR;
                      a clean single-strip hit should give ~1.
  popcount dist     — fraction of events with exactly k bits set.
  per-bit rates     — fraction of events in which each of the 10 bits fired.
  fold symmetry     — ratio of firing rates for bit k vs. bit (9-k);
                      a ratio near 1.00 for all pairs is the signature of a
                      folded-fiber readout where one physical strip maps to
                      MAROC channels k and (9-k) simultaneously.
  all-bits-set rate — fraction of events with all 10 bits set; values above
                      ~1 % indicate MAROC cross-talk or electronic noise.

Usage
-----
    python scripts/diagnose_hits.py <path> [--max-groups N]

    <path>            *.bin file  or  directory containing *.bin files
    --max-groups N    Analyse at most N event groups per file (quick scan)

Examples
--------
    python scripts/diagnose_hits.py data/0_testLab_20220204/Base/
    python scripts/diagnose_hits.py data/0_testLab_20220204/Base/20220204_100253.bin
    python scripts/diagnose_hits.py data/0_testLab_20220204/Base/ --max-groups 5000
"""

import argparse
import struct
from collections import Counter
from pathlib import Path

_HDR   = 8    # 4-byte n_rows + 4-byte n_cols
_WORD  = 8    # bytes per u64
_NBITS = 10   # fiber and ribbon masks are 10 bits each


# ── per-file reading ─────────────────────────────────────────────────────────

def _read_file(
    path: Path,
    max_groups: int | None,
) -> tuple[int, int, list[int]]:
    """
    Read one *.bin file.

    Returns (n_cols, n_groups, words) where words is a flat list of u64
    integers in the order they appear on disk (row-major within each group).
    """
    with open(path, 'rb') as fh:
        n_rows_total = struct.unpack_from('<I', fh.read(4))[0]
        n_cols       = struct.unpack_from('<I', fh.read(4))[0]
        raw = fh.read()

    n_groups = n_rows_total // 16
    if max_groups is not None:
        n_groups = min(n_groups, max_groups)

    n_words = n_groups * 16 * n_cols
    words = [
        struct.unpack_from('<Q', raw, i * _WORD)[0]
        for i in range(n_words)
    ]
    return n_cols, n_groups, words


# ── per-column accumulation ──────────────────────────────────────────────────

def _accumulate(
    words:    list[int],
    n_groups: int,
    n_cols:   int,
    col:      int,
    bit_freq: list[Counter],
    pop_hist: list[Counter],
) -> None:
    """
    OR the 16 rows for each group in the given column and accumulate
    per-bit firing counts and popcount histograms into bit_freq / pop_hist.

    Indices: 0=fiber_X, 1=ribbon_X, 2=fiber_Y, 3=ribbon_Y.

    All-bits-set events (popcount == 10, cross-talk artefacts) are counted
    in pop_hist but excluded from bit_freq so they do not bias the
    fold-symmetry calculation.
    """
    mask10 = (1 << _NBITS) - 1

    for g in range(n_groups):
        x_or = y_or = 0
        base = g * 16 * n_cols + col
        for row in range(16):
            w = words[base + row * n_cols]
            y_or |= w & 0xFFFFF
            x_or |= (w >> 32) & 0xFFFFF

        fx = (x_or >> _NBITS) & mask10
        rx =  x_or             & mask10
        fy = (y_or >> _NBITS) & mask10
        ry =  y_or             & mask10

        for i, m in enumerate([fx, rx, fy, ry]):
            pc = bin(m).count('1')
            pop_hist[i][pc] += 1
            # Exclude all-bits-set events from bit_freq: they set every bit
            # equally and would artificially pull fold-symmetry ratios toward
            # 1.00, masking or exaggerating the folded-fiber signal.
            if pc < _NBITS:
                for bit in range(_NBITS):
                    if (m >> bit) & 1:
                        bit_freq[i][bit] += 1


# ── reporting helpers ────────────────────────────────────────────────────────

_INACTIVE_THRESHOLD = 0.02   # bits firing < 2 % of events are treated as unwired


def _fold_symmetry(bit_freq: Counter, n_groups: int) -> list[float]:
    """
    Compute the (k, 9-k) mirror-symmetry ratio for each of the 5 bit pairs.

    A ratio of 1.00 means bits k and 9-k fire at exactly the same rate.
    For a folded-fiber readout where one physical strip maps to channels k
    and 9-k, all five ratios should be close to 1.00.

    Returns nan for pairs where either bit fires below _INACTIVE_THRESHOLD
    (unwired or disabled channel — the pairing is meaningless there).
    """
    ratios = []
    for k in range(_NBITS // 2):
        lo = bit_freq.get(k, 0)
        hi = bit_freq.get(_NBITS - 1 - k, 0)
        if lo / n_groups < _INACTIVE_THRESHOLD or hi / n_groups < _INACTIVE_THRESHOLD:
            ratios.append(float('nan'))
        elif max(lo, hi) == 0:
            ratios.append(float('nan'))
        else:
            ratios.append(min(lo, hi) / max(lo, hi))
    return ratios


def _active_bits(bit_freq: Counter, n_groups: int, threshold: float = 2.0) -> list[int]:
    """Return bit indices whose firing rate exceeds threshold percent."""
    return [b for b in range(_NBITS) if 100 * bit_freq.get(b, 0) / n_groups > threshold]


def _print_axis(
    label:     str,
    bit_freq:  Counter,
    pop_hist:  Counter,
    n_groups:  int,
) -> None:
    # True mean popcount from pop_hist (includes all-bits-set events).
    # bit_freq excludes all-bits-set, so using it here would give a
    # downward-biased mean.
    mean_pop = sum(k * pop_hist.get(k, 0) for k in range(_NBITS + 1)) / n_groups

    pop_parts = []
    for k in range(_NBITS + 1):
        c = pop_hist.get(k, 0)
        if c:
            pop_parts.append(f'{k}:{100*c/n_groups:.1f}%')
    pop_str = '  '.join(pop_parts)

    # Per-bit rates exclude all-bits-set events (see _accumulate).
    rates = [100 * bit_freq.get(b, 0) / n_groups for b in range(_NBITS)]
    rates_str = '[' + ', '.join(f'{r:.1f}' for r in rates) + ']'

    sym    = _fold_symmetry(bit_freq, n_groups)
    sym_str = '[' + ', '.join('nan' if s != s else f'{s:.2f}' for s in sym) + ']'
    sym_mean = [s for s in sym if s == s]
    sym_mean_val = sum(sym_mean) / len(sym_mean) if sym_mean else float('nan')

    n_all  = pop_hist.get(_NBITS, 0)
    n_zero = pop_hist.get(0, 0)
    active = _active_bits(bit_freq, n_groups)

    print(f'    {label}')
    print(f'      mean popcount    : {mean_pop:.2f}')
    print(f'      popcount dist    : {pop_str}')
    print(f'      per-bit rates %  : {rates_str}')
    print(f'      active bits (>2%): {active}')
    print(f'      fold sym (k,9-k) : {sym_str}  mean={sym_mean_val:.2f}  '
          f'[1.00 = perfect mirror → folded readout]')
    if n_all:
        flag = '  *** CROSS-TALK SUSPECTED' if 100 * n_all / n_groups > 1.0 else ''
        print(f'      all-bits-set     : {n_all} ({100*n_all/n_groups:.2f}%){flag}')
    if n_zero:
        print(f'      zero-bits (no hit): {n_zero} ({100*n_zero/n_groups:.2f}%)')


def _print_summary_table(
    all_stats: list[tuple[int, Counter, Counter, int]],
    n_groups:  int,
) -> None:
    """
    Print a compact one-line-per-(plane, axis) summary table.

    all_stats: list of (col, bit_freq_list[4], pop_hist_list[4], ...)
    """
    labels = ['fiber_X', 'ribbon_X', 'fiber_Y', 'ribbon_Y']
    print(f'\n  {"Plane":>5}  {"Axis":>10}  {"mean pop":>9}  '
          f'{"golden%":>8}  {"all-bits%":>10}  {"fold-sym":>9}')
    print('  ' + '-' * 60)
    for col, bit_freqs, pop_hists in all_stats:
        for i, lbl in enumerate(labels):
            bf  = bit_freqs[i]
            ph  = pop_hists[i]
            mp  = sum(k * ph.get(k, 0) for k in range(_NBITS + 1)) / n_groups
            g1  = 100 * ph.get(1, 0) / n_groups
            n_all = 100 * ph.get(_NBITS, 0) / n_groups
            sym = _fold_symmetry(bf, n_groups)
            sm  = [s for s in sym if s == s]
            sm_val = sum(sm) / len(sm) if sm else float('nan')
            print(f'  {col:>5}  {lbl:>10}  {mp:>9.2f}  '
                  f'{g1:>8.1f}  {n_all:>10.2f}  {sm_val:>9.2f}')


# ── main entry point ─────────────────────────────────────────────────────────

def diagnose(paths: list[Path], max_groups: int | None) -> None:
    """Aggregate statistics across all paths and print the full report."""

    # Determine n_cols from the first file.
    n_cols_ref, _, _ = _read_file(paths[0], max_groups=1)

    # Per-plane accumulators.
    bit_freqs = [[Counter() for _ in range(4)] for _ in range(n_cols_ref)]
    pop_hists = [[Counter() for _ in range(4)] for _ in range(n_cols_ref)]
    total_groups = 0

    for path in paths:
        n_cols, n_groups, words = _read_file(path, max_groups)
        if n_cols != n_cols_ref:
            print(f'  WARNING: {path.name} has {n_cols} planes '
                  f'(expected {n_cols_ref}) — skipped')
            continue
        for col in range(n_cols):
            _accumulate(words, n_groups, n_cols, col,
                        bit_freqs[col], pop_hists[col])
        total_groups += n_groups

    if total_groups == 0:
        print('No events found.')
        return

    print(f'\n{"="*65}')
    src = paths[0].parent if len(paths) > 1 else paths[0]
    print(f'Source  : {src}')
    print(f'Files   : {len(paths)}  |  Total groups: {total_groups}'
          + (f'  (capped at {max_groups}/file)' if max_groups else ''))
    print(f'Planes  : {n_cols_ref}')
    print(f'{"="*65}')

    # Summary table first for quick scanning.
    all_stats = [(col, bit_freqs[col], pop_hists[col]) for col in range(n_cols_ref)]
    _print_summary_table(all_stats, total_groups)

    # Detailed per-plane, per-axis breakdown.
    labels = [('fiber_X', 0), ('ribbon_X', 1), ('fiber_Y', 2), ('ribbon_Y', 3)]
    for col in range(n_cols_ref):
        print(f'\n  Plane {col}')
        for lbl, idx in labels:
            _print_axis(lbl, bit_freqs[col][idx], pop_hists[col][idx], total_groups)


def main() -> None:
    p = argparse.ArgumentParser(
        description='Diagnose hit-pattern quality in position *.bin files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('path', type=Path,
                   help='*.bin file or directory containing *.bin files')
    p.add_argument('--max-groups', type=int, default=None, metavar='N',
                   help='Analyse at most N event groups per file (quick scan)')
    args = p.parse_args()

    if args.path.is_dir():
        paths = sorted(
            f for f in args.path.glob('*.bin') if '_GPS' not in f.name
        )
        if not paths:
            p.error(f'No *.bin files found in {args.path}')
    elif args.path.is_file():
        paths = [args.path]
    else:
        p.error(f'Path not found: {args.path}')

    diagnose(paths, args.max_groups)


if __name__ == '__main__':
    main()
