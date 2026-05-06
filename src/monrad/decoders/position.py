#!/usr/bin/env python3
"""
Decoder for BuS_Tracker binary data files (*.bin).

File format:
  - bytes 0-3:  u32, number of rows
  - bytes 4-7:  u32, number of columns
  - bytes 8-end: n_rows * n_cols little-endian u64 values (row-major)
    Each u64 is structured as:
      bits  0-19  : Y-coordinate (20 bits): bits 0-9 = ribbon channels, bits 10-19 = fiber channels
      bits 20-31  : empty        (12 bits)
      bits 32-51  : X-coordinate (20 bits): bits 32-41 = ribbon channels, bits 42-51 = fiber channels
      bits 52-62  : GEN counter  (11 bits, LSB at bit 52; bit 63 unused)

  Rows are grouped in blocks of 16 (80 ns window sampled at 5 ns).  All 16
  rows in a block share the same GEN value.  The OR across the 16 samples is
  used for hit reconstruction.  Channel positions are numbered from LSB = 0,
  consistent with the GEN counter convention.
"""

import struct
import numpy as np
import sys


class BinDecoder:
    """Decoder for .bin event data files."""

    def __init__(self, filename: str):
        self.filename = filename

    def read(self) -> tuple[int, int, np.ndarray]:
        """
        Read file and return (n_cols, n_rows, data).

        data shape: (n_rows, n_cols), dtype uint64
        """
        with open(self.filename, 'rb') as f:
            raw = f.read()

        if len(raw) < 8:
            raise ValueError(f"File too short: {len(raw)} bytes")

        n_rows = struct.unpack_from('<I', raw, 0)[0]
        n_cols = struct.unpack_from('<I', raw, 4)[0]

        expected = 8 + n_rows * n_cols * 8
        if len(raw) != expected:
            raise ValueError(
                f"Size mismatch: expected {expected} bytes "
                f"({n_cols} cols × {n_rows} rows × 8 bytes + 8-byte header), "
                f"got {len(raw)}"
            )

        data = np.frombuffer(raw[8:], dtype='<u8').reshape(n_rows, n_cols)
        return n_cols, n_rows, data

    @staticmethod
    def _parse_u64(value: int) -> dict:
        return {
            'Y':   value        & 0xFFFFF,   # bits  0-19  (20 bits)
            'X':  (value >> 32) & 0xFFFFF,   # bits 32-51  (20 bits)
            'GEN':(value >> 52) & 0x7FF,     # bits 52-62  (11 bits; bit 63 unused)
        }

    def analyze(self):
        n_cols, n_rows, data = self.read()
        print(f"File:    {self.filename}")
        print(f"Columns: {n_cols}")
        print(f"Rows:    {n_rows}")
        for r in range(n_rows):
            parsed = [self._parse_u64(int(data[r, c])) for c in range(n_cols)]
            gens = [p['GEN'] for p in parsed]
            gen_ok = len(set(gens)) == 1
            print(f"  row {r}:  GEN={gens}  {'OK' if gen_ok else 'MISMATCH'}")
            for c, p in enumerate(parsed):
                print(f"    u64[{c}]:  Y={p['Y']}  X={p['X']}  GEN={p['GEN']}")

    @staticmethod
    def _fmt_bits(value: int, width: int = 20) -> str:
        s = format(value, f'0{width}b')
        h = width // 2
        return s[:h] + '|' + s[h:]

    @staticmethod
    def _fmt_counts(vals: list, width: int = 20) -> str:
        def ch(n):
            return str(n) if n <= 9 else chr(ord('a') + n - 10)
        s = ''.join(ch(sum((v >> bit) & 1 for v in vals))
                    for bit in range(width - 1, -1, -1))
        h = width // 2
        return s[:h] + '|' + s[h:]

    @staticmethod
    def _single_bit_pos(value: int) -> int | None:
        """Return bit position (0=LSB) if exactly one bit is set, else None."""
        if value == 0 or (value & (value - 1)) != 0:
            return None
        return value.bit_length() - 1

    @staticmethod
    def _find_clusters(value: int, width: int = 10) -> list[list[int]]:
        """Find contiguous groups of set bits. Position 0 = LSB."""
        clusters, current = [], []
        for pos in range(width):
            if (value >> pos) & 1:
                current.append(pos)
            elif current:
                clusters.append(current)
                current = []
        if current:
            clusters.append(current)
        return clusters

    @staticmethod
    def _reconstruct_coord(fiber_clusters: list, ribbon_clusters: list, N: int = 10):
        """
        Return (centroid, candidates) if the single fiber × single ribbon cluster
        produces a contiguous range of X = N*r + f values, else None.
        Contiguity is checked in the combined coordinate, not in fiber/ribbon space
        separately — adjacent ribbons are N apart, not 1.
        """
        if len(fiber_clusters) != 1 or len(ribbon_clusters) != 1:
            return None
        fc, rc = fiber_clusters[0], ribbon_clusters[0]
        candidates = sorted(N * r + f for r in rc for f in fc)
        if candidates != list(range(candidates[0], candidates[0] + len(candidates))):
            return None
        return sum(candidates) / len(candidates), candidates

    @staticmethod
    def _is_valid(x_or: int, y_or: int) -> tuple[bool, list[str]]:
        """Return (valid, reasons).
        Invalid if any fiber/ribbon half of OR equals 1023 (all bits set),
        or if the ribbon half of OR is 0 (no ribbon channel fired).
        """
        FULL = 0x3FF
        x_fiber, x_ribbon = (x_or >> 10) & FULL, x_or & FULL
        y_fiber, y_ribbon = (y_or >> 10) & FULL, y_or & FULL
        reasons = []
        if x_fiber == FULL:
            reasons.append('X_fiber=1023')
        if x_ribbon == FULL:
            reasons.append('X_ribbon=1023')
        if y_fiber == FULL:
            reasons.append('Y_fiber=1023')
        if y_ribbon == FULL:
            reasons.append('Y_ribbon=1023')
        if x_ribbon == 0:
            reasons.append('X_ribbon=0')
        if y_ribbon == 0:
            reasons.append('Y_ribbon=0')
        return not reasons, reasons

    def or_visual(self, max_groups: int = None):
        """
        Group rows by GEN (16 consecutive samples), bitwise-OR the X and Y
        fields across the group, and print a visual bitstream table.

        Validity: a column is INVALID if any fiber/ribbon half of the OR equals
        1023 (all bits set) or if the ribbon half is 0 (no ribbon channel fired).

        Hit reconstruction for valid columns:
          - Fiber and ribbon bits are clustered into contiguous groups (LSB=ch.0).
          - A hit is reconstructed when each half yields exactly one cluster whose
            cross-product N*r + f forms a contiguous range in the combined coord.
          - Golden: both fiber and ribbon have a single bit (unique channel pair).
          - Cluster: adjacent bits produce an interpolated (half-integer) position.
          - Unresolved: multiple clusters or non-contiguous candidates.
        A reconstruction summary is printed after all groups.
        """
        GROUP_SIZE = 16
        n_cols, n_rows, data = self.read()
        n_groups = n_rows // GROUP_SIZE
        remainder = n_rows % GROUP_SIZE

        print(f"File:    {self.filename}")
        print(f"Columns: {n_cols},  Rows: {n_rows},  Groups: {n_groups}"
              + (f"  (+{remainder} leftover rows ignored)" if remainder else ""))

        limit = n_groups if max_groups is None else min(max_groups, n_groups)
        stats = {'golden': 0, 'cluster': 0, 'unresolved': 0, 'invalid': 0}

        for g in range(limit):
            row_start = g * GROUP_SIZE
            rows = range(row_start, row_start + GROUP_SIZE)

            group = [[self._parse_u64(int(data[r, c])) for c in range(n_cols)]
                     for r in rows]

            gen_vals = [group[s][0]['GEN'] for s in range(GROUP_SIZE)]
            gen_ok = len(set(gen_vals)) == 1
            gen_label = str(gen_vals[0]) if gen_ok else f"MISMATCH {gen_vals}"

            # Compute OR and validity for all columns before printing the header
            results = []
            for c in range(n_cols):
                x_vals = [group[s][c]['X'] for s in range(GROUP_SIZE)]
                y_vals = [group[s][c]['Y'] for s in range(GROUP_SIZE)]
                x_or = 0
                y_or = 0
                for v in x_vals:
                    x_or |= v
                for v in y_vals:
                    y_or |= v
                valid, reasons = self._is_valid(x_or, y_or)
                results.append((x_vals, y_vals, x_or, y_or, valid, reasons))

            group_valid = all(r[4] for r in results)
            group_tag = 'VALID' if group_valid else 'INVALID'

            print(f"\n{'='*64}")
            print(f"Group {g}  (rows {row_start}-{row_start+GROUP_SIZE-1})  "
                  f"GEN={gen_label}  [{group_tag}]")

            sep = '-' * 58
            for c, (x_vals, y_vals, x_or, y_or, valid, reasons) in enumerate(results):
                col_tag = 'VALID' if valid else f"INVALID ({', '.join(reasons)})"
                print(f"\n  Column {c}  [{col_tag}]")
                print(f"  {'Sample':>8}  {'X (fiber|ribbon)':^21}  {'Y (fiber|ribbon)':^21}")
                print(f"  {sep}")
                for s in range(GROUP_SIZE):
                    print(f"  {s:>8}  {self._fmt_bits(x_vals[s])}  "
                          f"{self._fmt_bits(y_vals[s])}")
                print(f"  {sep}")
                print(f"  {'Count':>8}  {self._fmt_counts(x_vals)}  "
                      f"{self._fmt_counts(y_vals)}")
                print(f"  {'OR':>8}  {self._fmt_bits(x_or)}  "
                      f"{self._fmt_bits(y_or)}")
                print(f"  {'':>8}  X={x_or:<8d}  Y={y_or:<8d}")

                if not valid:
                    stats['invalid'] += 1
                    continue

                N = 10
                xfc = self._find_clusters((x_or >> N) & 0x3FF)
                xrc = self._find_clusters(x_or & 0x3FF)
                yfc = self._find_clusters((y_or >> N) & 0x3FF)
                yrc = self._find_clusters(y_or & 0x3FF)

                res_x = self._reconstruct_coord(xfc, xrc, N)
                res_y = self._reconstruct_coord(yfc, yrc, N)

                if res_x is not None and res_y is not None:
                    cx, cands_x = res_x
                    cy, cands_y = res_y
                    golden = len(cands_x) == 1 and len(cands_y) == 1
                    tag = 'golden' if golden else 'cluster'
                    stats[tag] += 1

                    def fmt_coord(centroid, cands):
                        v = f"{centroid:.0f}" if centroid == int(centroid) else f"{centroid:.1f}"
                        if len(cands) == 1:
                            r, f = divmod(cands[0], N)
                            return f"{v} (r={r}, f={f})"
                        return f"{v} {{{cands[0]}..{cands[-1]}}}"

                    print(f"  {'Hit':>8}  [{tag}]  "
                          f"X={fmt_coord(cx, cands_x)}    Y={fmt_coord(cy, cands_y)}")
                else:
                    stats['unresolved'] += 1

                    def diag(fc, rc):
                        if len(fc) != 1 or len(rc) != 1:
                            return f"{len(fc)} fiber cluster(s), {len(rc)} ribbon cluster(s)"
                        cands = sorted(N * r + f for r in rc[0] for f in fc[0])
                        return f"non-contiguous {{{','.join(map(str, cands))}}}"

                    print(f"  {'Hit':>8}  [unresolved]  "
                          f"X: {diag(xfc, xrc)}  Y: {diag(yfc, yrc)}")

        n_valid = stats['golden'] + stats['cluster'] + stats['unresolved']
        n_total = n_valid + stats['invalid']
        n_restored = stats['golden'] + stats['cluster']
        pct = lambda n, d: 100 * n / d if d else 0.0
        print(f"\n{'='*64}")
        print(f"Reconstruction summary  ({limit} groups × {n_cols} columns = {n_total} total)")
        print(f"  Invalid:     {stats['invalid']:>6}  ({pct(stats['invalid'], n_total):5.1f}%)")
        print(f"  Golden:      {stats['golden']:>6}  ({pct(stats['golden'],  n_total):5.1f}%)  "
              f"[1 fiber bit + 1 ribbon bit]")
        print(f"  Cluster:     {stats['cluster']:>6}  ({pct(stats['cluster'], n_total):5.1f}%)  "
              f"[single contiguous cluster each]")
        print(f"  Unresolved:  {stats['unresolved']:>6}  ({pct(stats['unresolved'], n_total):5.1f}%)  "
              f"[multiple clusters or no fiber hit]")
        print(f"  ---")
        print(f"  Cluster approach: {n_restored} / {n_valid} valid  ({pct(n_restored, n_valid):.1f}%)")

    def export_csv(self, output_file: str = None):
        """Export data as CSV with Y, X, GEN fields extracted from each u64."""
        if output_file is None:
            output_file = self.filename.replace('.bin', '_decoded.csv')

        n_cols, n_rows, data = self.read()

        header = 'row_id,' + ','.join(
            f'col_{i}_Y,col_{i}_X,col_{i}_GEN' for i in range(n_cols)
        )

        with open(output_file, 'w') as f:
            f.write(header + '\n')
            for r in range(n_rows):
                fields = []
                for c in range(n_cols):
                    p = self._parse_u64(int(data[r, c]))
                    fields += [str(p['Y']), str(p['X']), str(p['GEN'])]
                f.write(f"{r}," + ','.join(fields) + '\n')

        print(f"Exported {n_rows} rows × {n_cols} u64s (Y/X/GEN) to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python decode_bin.py <bin_file> [--csv [output.csv]] [--or [N]]")
        print("  --or [N]   bitwise-OR visual for each GEN group (optionally limit to N groups)")
        sys.exit(1)

    bin_file = sys.argv[1]
    decoder = BinDecoder(bin_file)

    if '--csv' in sys.argv:
        idx = sys.argv.index('--csv')
        out = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('--') else None
        decoder.export_csv(out)
    elif '--or' in sys.argv:
        idx = sys.argv.index('--or')
        nxt = sys.argv[idx + 1] if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith('--') else None
        max_groups = int(nxt) if nxt is not None else None
        decoder.or_visual(max_groups)
    else:
        decoder.analyze()


if __name__ == "__main__":
    main()
